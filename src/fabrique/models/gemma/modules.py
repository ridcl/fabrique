from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn import _modules

from fabrique.models.gemma.layers import GemmaRMSNorm

K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

# A dictionary with the following array shapes as keys:
# v: [batch_size, cache_size, num_heads, head_dim]
# k: [batch_size, cache_size, num_heads, head_dim]
# end_index: [batch_size]
LayerCache = dict[str, jax.Array]


def _create_sliding_mask(
    segment_pos: jnp.ndarray,
    end_index: int,
    cache_len: int,
    sliding_window_size: int,
):
    """Creates mask for sliding window attention."""
    total_tokens = end_index + segment_pos.shape[1]  # cached + processing tokens

    def _reconstruct_rotated_cache_positions():
        cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
        cache_positions = (
            jnp.zeros_like(cache_positions)
            # kv were placed at index (position_id % cache_len) in the cache.
            .at[cache_positions % cache_len].set(cache_positions)
        )
        return cache_positions

    # Reconstruct position_ids for cached kv.
    cache_positions = jax.lax.cond(
        total_tokens <= cache_len,
        lambda: jnp.arange(cache_len),
        _reconstruct_rotated_cache_positions,
    )

    cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
    segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]
    sliding_mask = cache_positions > segment_pos - sliding_window_size
    sliding_mask *= cache_positions < segment_pos + sliding_window_size
    return sliding_mask


AttentionType = _modules.AttentionType


class Embedder(nnx.Module):
    """Embedder module."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        vision_proj_dim: int | None = None,
        *,
        param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.vision_proj_dim = vision_proj_dim
        # Embedding matrix of shape [vocab_size, embed_dim]
        # self.input_embedding_table = self.param(
        #     'input_embedding',
        #     nn.initializers.normal(),
        #     (self.vocab_size, self.embed_dim),
        # )
        self.input_embedding_table = nnx.Param(
            jax.random.normal(rngs.params(), (vocab_size, embed_dim), dtype=param_dtype)
        )

        # For the multi-modal models, the encoder has additional parameters:
        # * `mm_soft_embedding_norm` and `mm_input_projection`: Those weights
        #   serve to project the soft tokens from the image encoder into the
        #   embedding space of the text encoder. Those tokens are then merged with
        #   the text tokens inside `Transformer._include_vision_embeddings`.
        if self.vision_proj_dim:
            # note: keeping the params in float32
            self.mm_soft_embedding_norm = GemmaRMSNorm(self.vision_proj_dim, rngs=rngs)
            self.mm_input_projection = nnx.Einsum(
                "...tm,md->...td",
                kernel_shape=(self.vision_proj_dim, self.embed_dim),
                rngs=rngs,
            )

    def encode(self, x: jax.Array) -> jax.Array:
        """Encodes the input tokens.

        Args:
            x: Input tokens of shape [seq_len] or [batch_size, seq_len], where
                each token is an integer in [0, vocab_size).

        Returns:
            Encoded tokens of shape [seq_len, embed_dim] or [batch_size, seq_len,
            embed_dim].
        """
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        """Decodes the input vectors.

        Args:
            x: Array of shape [seq_len, embed_dim] or [batch_size, seq_len,
                embed_dim].

        Returns:
            Array of shape [seq_len, vocab_size] or [batch_size, seq_len, vocab_size].
        """
        return jnp.dot(x, self.input_embedding_table.T)

    def encode_vision(self, x: jax.Array) -> jax.Array:
        """Projects siglip embeddings to the embedding space of the text encoder."""
        x = self.mm_soft_embedding_norm(x)
        x = self.mm_input_projection(x)
        return x


class Attention(nnx.Module):
    """Attention module."""

    @property
    def use_qkv_einsum(self):
        return self.num_kv_heads == self.num_heads

    @property
    def use_gqa(self):
        return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        features: int,
        head_dim: int,
        attn_type: AttentionType,
        query_pre_attn_scalar: float,
        rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY,
        rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR,
        attn_logits_soft_cap: float | None = None,
        sliding_window_size: int | None = None,
        use_qk_norm: bool = False,
        param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.features = features
        self.head_dim = head_dim
        self.attn_type = attn_type
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.rope_base_frequency = rope_base_frequency
        self.rope_scale_factor = rope_scale_factor
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.sliding_window_size = sliding_window_size
        self.use_qk_norm = use_qk_norm

        einsum = partial(nnx.Einsum, param_dtype=param_dtype, rngs=rngs)
        init_fn = nnx.initializers.lecun_normal()

        self.attn_vec_einsum = einsum(
            "BTNH,NHD->BTD",
            kernel_shape=(self.num_heads, self.head_dim, self.features),
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
        )
        if self.use_qkv_einsum:
            self.qkv_einsum = einsum(
                "BTD,SNDH->SBTNH",
                kernel_shape=(3, self.num_heads, self.features, self.head_dim),
                kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            )
        else:
            self.q_einsum = einsum(
                "BTD,NDH->BTNH",
                kernel_shape=(self.num_heads, self.features, self.head_dim),
                kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            )
            self.kv_einsum = einsum(
                "BSD,CKDH->CBSKH",
                kernel_shape=(2, self.num_kv_heads, self.features, self.head_dim),
                kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            )
        if self.use_qk_norm:
            self._query_norm = GemmaRMSNorm(
                self.head_dim, param_dtype=param_dtype, rngs=rngs
            )
            self._key_norm = GemmaRMSNorm(
                self.head_dim, param_dtype=param_dtype, rngs=rngs
            )

    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        cache: LayerCache | None,
        attn_mask: jax.Array,
    ) -> tuple[LayerCache | None, jax.Array]:
        """Applies multi-head attention to the inputs.

        Args:
            x: Input sequence of shape [batch_size, seq_len, embed_dim].
            segment_pos: Input absolute positions of shape [batch_size, seq_len].
            cache: KV cache or None.
            attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

        Returns:
           cache: Updated attention KV cache.
            outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
        """
        if self.use_qkv_einsum:
            # [batch_size, seq_len, num_heads, head_dim]
            query_proj, key_proj, value_proj = self.qkv_einsum(x)
        else:
            query_proj = self.q_einsum(x)
            key_proj, value_proj = self.kv_einsum(x)

        if self.use_qk_norm:
            query_proj = self._query_norm(query_proj)
            key_proj = self._key_norm(key_proj)

        query_proj = _positional_embeddings.apply_rope(
            query_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )
        query_scaled = query_proj * self.query_pre_attn_scalar

        key_proj = _positional_embeddings.apply_rope(
            key_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )

        # Cache is left aligned.
        # Save the KV values to the cache.
        if cache is not None:
            end_index = cache["end_index"][0]
            cache_size = cache["v"].shape[1]
            slice_indices = (0, end_index % cache_size, 0, 0)

            # [batch_size, cache_size, num_heads, head_dim]
            value_proj = jax.lax.dynamic_update_slice(
                cache["v"],
                value_proj,
                slice_indices,
            )

            # [batch_size, cache_size, num_heads, head_dim]
            key_proj = jax.lax.dynamic_update_slice(cache["k"], key_proj, slice_indices)

        if self.use_gqa:
            # Reshape matrices to enable einsums over groups.
            b, t, kg, h = query_scaled.shape
            query_scaled = query_scaled.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
            )
            logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_scaled, key_proj)
            b, t, k, g, s = logits.shape
            logits = logits.reshape((b, t, k * g, s))
        else:
            # [batch_size, seq_len, num_heads, cache_size]
            # If cache is None, then cache_size = seq_len.
            logits = jnp.einsum("BTNH,BSNH->BTNS", query_scaled, key_proj)

        if self.attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / self.attn_logits_soft_cap)
            logits = logits * self.attn_logits_soft_cap

        if self.attn_type == AttentionType.LOCAL_SLIDING:
            if self.sliding_window_size is None:
                raise ValueError(
                    "Sliding_window_size must be set if Local Sliding attention type"
                )
            sliding_mask = _create_sliding_mask(
                segment_pos,
                end_index=cache["end_index"][0] if cache is not None else 0,
                # Derive cache length from attn_mask shape in case cache is None
                cache_len=attn_mask.shape[-1],
                sliding_window_size=self.sliding_window_size,
            )
            # [batch_size, seq_len, cache_size]
            attn_mask *= sliding_mask

        # [batch_size, seq_len, num_heads, cache_size]
        padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

        # Multi-head attention matrices.
        # [batch_size, seq_len, num_heads, cache_size]
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

        if self.use_gqa:
            # Reshape matrices to enable einsums over groups.
            b, t, kg, h = probs.shape
            probs = probs.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
            )
            encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs, value_proj)
            b, t, k, g, h = encoded.shape
            encoded = encoded.reshape((b, t, k * g, h))
        else:
            # [batch_size, seq_len, num_heads, head_dim]
            encoded = jnp.einsum("BTNS,BSNH->BTNH", probs, value_proj)

        # [batch_size, seq_len, features]
        attn_output = self.attn_vec_einsum(encoded)

        if cache is not None:
            seq_len = x.shape[1]
            new_cache = {
                # [batch_size, cache_size, num_heads, head_dim]
                "v": value_proj,
                # [batch_size, cache_size, num_heads, head_dim]
                "k": key_proj,
                # [batch_size]
                "end_index": cache["end_index"] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, attn_output

    @classmethod
    def init_cache(
        cls,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> LayerCache:
        del cls  # not used
        return {
            "v": jnp.zeros((batch_size, cache_size, num_heads, head_dim), dtype=dtype),
            "k": jnp.zeros((batch_size, cache_size, num_heads, head_dim), dtype=dtype),
            "end_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        }


class FeedForward(nnx.Module):
    """Feed forward module."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        transpose_gating_einsum: bool,
        *,
        param_dtype: jax.typing.DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.features = features  # features = embed_dim
        self.hidden_dim = hidden_dim
        self.transpose_gating_einsum = transpose_gating_einsum

        einsum = partial(nnx.Einsum, param_dtype=param_dtype, rngs=rngs)
        init_fn = nnx.initializers.lecun_normal()
        # Some versions use an alternate parameter ordering that
        # transposes hidden_dim and features.
        if self.transpose_gating_einsum:
            self.gating = einsum(
                "...F,NHF->...NH",
                kernel_shape=(2, self.hidden_dim, self.features),
                kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            )
        else:
            self.gating = einsum(
                "...F,NFH->...NH",
                kernel_shape=(2, self.features, self.hidden_dim),
                kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            )
        # Use the same scope for backwards compatibility with existing checkpoints
        # created before using `_layers.Einsum` here.
        # import flax.linen as nn
        # nn.share_scope(self, gating)

        # Project back from hidden_dim to features.
        self.linear = einsum(
            "...H,HF->...F",
            kernel_shape=(self.hidden_dim, self.features),
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
        )
        # nn.share_scope(self, linear)

    def __call__(self, x):
        """Applies the feed forward module.

        Args:
            x: Input sequence of shape [batch_size, seq_len, features].

        Returns:
            Output sequence of shape [batch_size, seq_len, features].
        """
        # [batch_size, seq_len, 2, hidden_dim]
        gate = self.gating(x)
        # [batch_size, seq_len, hidden_dim]
        activations = nnx.gelu(gate[..., 0, :]) * gate[..., 1, :]
        # [batch_size, seq_len, features]
        outputs = self.linear(activations)
        return outputs


class Block(nnx.Module):
    """Transformer block."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        use_post_attn_norm: bool,
        use_post_ffw_norm: bool,
        attn_type: AttentionType,
        query_pre_attn_scalar: float,
        transpose_gating_einsum: bool,
        rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY,
        rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR,
        attn_logits_soft_cap: float | None = None,
        sliding_window_size: int | None = None,
        use_qk_norm: bool = False,
        *,
        param_dtype: jax.typing.DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.use_post_attn_norm = use_post_attn_norm
        self.use_post_ffw_norm = use_post_ffw_norm
        self.attn_type = attn_type
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.transpose_gating_einsum = transpose_gating_einsum
        self.rope_base_frequency = rope_base_frequency
        self.rope_scale_factor = rope_scale_factor
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.sliding_window_size = sliding_window_size
        self.use_qk_norm = use_qk_norm

        # norm = partial()
        self.pre_attention_norm = GemmaRMSNorm(
            embed_dim, param_dtype=param_dtype, rngs=rngs
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            features=self.embed_dim,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            attn_type=self.attn_type,
            query_pre_attn_scalar=self.query_pre_attn_scalar,
            rope_base_frequency=self.rope_base_frequency,
            rope_scale_factor=self.rope_scale_factor,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            sliding_window_size=self.sliding_window_size,
            use_qk_norm=self.use_qk_norm,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if self.use_post_attn_norm:
            self.post_attention_norm = GemmaRMSNorm(
                embed_dim, param_dtype=param_dtype, rngs=rngs
            )
        else:
            self.post_attention_norm = None

        self.pre_ffw_norm = GemmaRMSNorm(embed_dim, param_dtype=param_dtype, rngs=rngs)

        self.mlp = FeedForward(
            features=self.embed_dim,
            hidden_dim=self.hidden_dim,
            transpose_gating_einsum=self.transpose_gating_einsum,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if self.use_post_ffw_norm:
            self.post_ffw_norm = GemmaRMSNorm(
                embed_dim, param_dtype=param_dtype, rngs=rngs
            )
        else:
            self.post_ffw_norm = None

    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        cache: LayerCache | None,
        attn_mask: jax.Array,
    ) -> tuple[LayerCache | None, jax.Array]:
        """Applies the block to the inputs.

        Args:
            x: Input sequence of shape [batch_size, seq_len, embed_dim].
            segment_pos: Input absolute positions of shape [batch_size, seq_len].
            cache: KV cache or None.
            attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

        Returns:
            cache: Updated attention KV cache.
            outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
        """
        inputs_normalized = self.pre_attention_norm(x)

        # attn_output.shape = [batch_size, seq_len, embed_dim]
        # cache["k"].shape = [batch_size, cache_size, num_heads, head_dim]
        # cache["v"].shape = [batch_size, cache_size, num_heads, head_dim]
        # cache["end_index"].shape = [batch_size]
        cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )

        if self.post_attention_norm is not None:
            attn_output = self.post_attention_norm(attn_output)

        attn_output += x

        outputs = self.pre_ffw_norm(attn_output)

        outputs = self.mlp(outputs)

        if self.post_ffw_norm is not None:
            outputs = self.post_ffw_norm(outputs)

        outputs += attn_output

        return cache, outputs


###############################################################################


def example():
    rngs = nnx.Rngs(params=0)
    batch_size = 1
    seq_len = 5
    # cache_len = 5
    num_heads: int = 2
    num_kv_heads: int = 2
    features: int = 12
    head_dim: int = 8
    attn_type: AttentionType = AttentionType.LOCAL_SLIDING
    query_pre_attn_scalar: float = 1.0
    sliding_window_size = 3
    self = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=features,
        head_dim=head_dim,
        attn_type=attn_type,
        query_pre_attn_scalar=query_pre_attn_scalar,
        sliding_window_size=sliding_window_size,
        rngs=rngs,
    )

    x: jax.Array = jax.random.normal(rngs.params(), (batch_size, seq_len, features))
    segment_pos: jax.Array = jnp.arange(seq_len).reshape(batch_size, -1)
    cache: LayerCache | None = None
    attn_mask: jax.Array = nnx.make_causal_mask(jnp.ones(5))

    self(x, segment_pos, cache, attn_mask)

    self = Embedder(100, 8, 12, rngs=rngs)
    self.encode_vision(jnp.ones((5, 12))).shape
