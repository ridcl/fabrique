"""Gemma 4 vision encoder implementation."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass
class VisionConfig:
    """Configuration for the Gemma 4 vision encoder."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    head_dim: int = 64
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    in_channels: int = 3
    max_soft_tokens: int = 280
    image_token_id: int = 258_880


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def compute_2d_rope(
    pixel_position_ids: jax.Array,
    head_dim: int,
    theta: float = 100.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute 2D RoPE cos/sin for vision patches.

    Each spatial dimension (x, y) gets its own set of frequencies computed over
    spatial_dim = head_dim // 2 = 32 channels, then concatenated.

    Args:
      pixel_position_ids: [B, N, 2] integer (x, y) position ids; -1 for padding.
      head_dim: Per-head dimension (64 for Gemma4 vision).
      theta: RoPE base frequency (100.0).

    Returns:
      cos, sin: each [B, N, head_dim] float32.
    """
    spatial_dim = head_dim // 2  # = 32
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, spatial_dim, 2, dtype=jnp.float32) / spatial_dim)
    )  # [spatial_dim // 2 = 16]

    all_cos, all_sin = [], []
    for i in range(2):  # x=0, y=1
        pos = pixel_position_ids[..., i].astype(jnp.float32)  # [B, N]
        freqs = jnp.einsum("bn,d->bnd", pos, inv_freq)  # [B, N, 16]
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # [B, N, 32]
        all_cos.append(jnp.cos(emb))
        all_sin.append(jnp.sin(emb))

    cos = jnp.concatenate(all_cos, axis=-1)  # [B, N, 64]
    sin = jnp.concatenate(all_sin, axis=-1)  # [B, N, 64]
    return cos, sin


def apply_2d_rope(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply 2D RoPE to q or k.

    Splits head_dim into x-half and y-half, applies standard RoPE to each half
    independently using the pre-split cos/sin, then concatenates.

    Args:
      x: [B, N, H, head_dim].
      cos: [B, N, head_dim] — first half is for x, second half for y.
      sin: [B, N, head_dim].

    Returns:
      [B, N, H, head_dim] with 2D RoPE applied.
    """
    half_dim = x.shape[-1] // 2  # = 32
    x_x = x[..., :half_dim]  # [B, N, H, 32]
    x_y = x[..., half_dim:]  # [B, N, H, 32]
    cos_x = cos[..., :half_dim][:, :, None, :]  # [B, N, 1, 32]
    cos_y = cos[..., half_dim:][:, :, None, :]
    sin_x = sin[..., :half_dim][:, :, None, :]
    sin_y = sin[..., half_dim:][:, :, None, :]
    x_x_rot = x_x * cos_x + rotate_half(x_x) * sin_x
    x_y_rot = x_y * cos_y + rotate_half(x_y) * sin_y
    return jnp.concatenate([x_x_rot, x_y_rot], axis=-1)


class RMSNorm(nnx.Module):
    """RMSNorm with optional learnable scale (with_scale=False for v_norm)."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        with_scale: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.eps = eps
        self.with_scale = with_scale
        self.dtype = dtype
        if with_scale:
            self.scale = nnx.Param(
                nnx.initializers.ones_init()(rngs.params(), (dim,)).astype(param_dtype)
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f32 = x.astype(jnp.float32)
        rms_inv = jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        out = (x_f32 * rms_inv).astype(x.dtype)
        if self.with_scale:
            return out * self.scale.value.astype(x.dtype)
        return out


class VisionPatchEmbedder(nnx.Module):
    """Embeds image patches with 2D learned positional embeddings."""

    def __init__(
        self,
        config: VisionConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        patch_channels = config.in_channels * config.patch_size**2  # 3*16^2 = 768
        self.input_proj = nnx.Linear(
            patch_channels,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # 2D position embedding table: one row per spatial axis (x, y).
        # Initialized to ones, matching HF: nn.Parameter(torch.ones(2, pos_size, H)).
        self.position_embedding_table = nnx.Param(
            jnp.ones(
                (2, config.position_embedding_size, config.hidden_size),
                dtype=param_dtype,
            )
        )

    def __call__(
        self,
        pixel_values: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> jax.Array:
        """
        Args:
          pixel_values: [B, N, patch_channels] float32 in [0, 1].
          pixel_position_ids: [B, N, 2] int32 (x, y); -1 for padding patches.

        Returns:
          [B, N, hidden_size] embedded patches.
        """
        # Normalize [0, 1] → [-1, 1] matching HF: pixel_values = 2*(v - 0.5).
        pixel_values = 2.0 * pixel_values.astype(jnp.float32) - 1.0

        # Linear patch embedding.
        hidden = self.input_proj(pixel_values.astype(self.dtype))  # [B, N, H]

        # 2D positional embeddings via one-hot lookup over the table.
        pos_size = self.config.position_embedding_size
        clamped = jnp.clip(pixel_position_ids, 0, pos_size - 1)  # [B, N, 2]
        table = self.position_embedding_table.value.astype(self.dtype)  # [2, P, H]

        one_hot_x = jax.nn.one_hot(clamped[..., 0], pos_size, dtype=self.dtype)
        one_hot_y = jax.nn.one_hot(clamped[..., 1], pos_size, dtype=self.dtype)
        pe = jnp.einsum("bnp,ph->bnh", one_hot_x, table[0]) + jnp.einsum(
            "bnp,ph->bnh", one_hot_y, table[1]
        )  # [B, N, H]

        # Zero out positional embeddings for padding patches.
        padding = (pixel_position_ids == -1).all(axis=-1, keepdims=True)  # [B,N,1]
        pe = jnp.where(padding, 0.0, pe)

        return hidden + pe


class VisionAttention(nnx.Module):
    """Multi-head attention for the Gemma4 vision encoder.

    Key differences from standard attention:
    - scaling = 1.0 (no head_dim^{-0.5} scaling)
    - q_norm and k_norm have learnable scale; v_norm does NOT
    - 2D RoPE applied to q and k (split head_dim into x-half and y-half)
    - Bidirectional attention (no causal mask; padding masked out)
    """

    def __init__(
        self,
        config: VisionConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        h = config.hidden_size
        n = config.num_attention_heads
        d = config.head_dim

        # All projections use ClippableLinear matching HF's Gemma4ClippableLinear.
        # Checkpoint keys have a `.linear.weight` suffix; clip scalars are separate.
        self.q_proj = ClippableLinear(
            h, n * d, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.k_proj = ClippableLinear(
            h, n * d, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.v_proj = ClippableLinear(
            h, n * d, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.o_proj = ClippableLinear(
            n * d, h, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.q_norm = RMSNorm(
            d,
            eps=config.rms_norm_eps,
            with_scale=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.k_norm = RMSNorm(
            d,
            eps=config.rms_norm_eps,
            with_scale=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # v_norm has no learnable weight (with_scale=False → no checkpoint tensor).
        self.v_norm = RMSNorm(
            d,
            eps=config.rms_norm_eps,
            with_scale=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> jax.Array:
        """
        Args:
          x: [B, N, hidden_size].
          cos, sin: [B, N, head_dim] from compute_2d_rope.
          pixel_position_ids: [B, N, 2] for building the padding mask.

        Returns:
          [B, N, hidden_size].
        """
        B, N, _ = x.shape
        H = self.config.num_attention_heads
        D = self.config.head_dim

        q = self.q_proj(x).reshape(B, N, H, D)
        k = self.k_proj(x).reshape(B, N, H, D)
        v = self.v_proj(x).reshape(B, N, H, D)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Apply 2D RoPE in float32 then cast back.
        q_f = apply_2d_rope(
            q.astype(jnp.float32), cos.astype(jnp.float32), sin.astype(jnp.float32)
        ).astype(self.dtype)
        k_f = apply_2d_rope(
            k.astype(jnp.float32), cos.astype(jnp.float32), sin.astype(jnp.float32)
        ).astype(self.dtype)

        # Bidirectional attention mask: valid[i] ∧ valid[j].
        # Padding patches get -inf so they are excluded from softmax.
        valid = ~(pixel_position_ids == -1).all(axis=-1)  # [B, N]
        attn_bias = jnp.where(
            valid[:, :, None] & valid[:, None, :], 0.0, jnp.finfo(jnp.float32).min
        )  # [B, N, N]
        attn_bias = attn_bias[:, None, :, :]  # [B, 1, N, N]

        # Gemma4 vision uses scaling = 1.0 (not head_dim^{-0.5}).
        q_t = jnp.transpose(q_f, (0, 2, 1, 3))  # [B, H, N, D]
        k_t = jnp.transpose(k_f, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))

        scores = jnp.einsum("bhnd,bhmd->bhnm", q_t, k_t)  # [B, H, N, N], scale=1.0
        scores = scores + attn_bias

        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(self.dtype)
        out = jnp.einsum("bhnm,bhmd->bhnd", weights, v_t)  # [B, H, N, D]

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, N, H * D)
        return self.o_proj(out)


class ClippableLinear(nnx.Module):
    """Linear layer with optional per-tensor input/output clipping.

    Mirrors HF's ``Gemma4ClippableLinear``.  The four scalar clip buffers
    (``input_min``, ``input_max``, ``output_min``, ``output_max``) default to
    ±inf so that they act as identity operations when not loaded from a
    checkpoint.  When a quantized checkpoint is loaded the loader fills them
    with calibrated bfloat16 scalars.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(
            in_features,
            out_features,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Scalar clip values; default to ±inf (pass-through).
        self.input_min = nnx.Param(jnp.full((), -jnp.inf, dtype=param_dtype))
        self.input_max = nnx.Param(jnp.full((), jnp.inf, dtype=param_dtype))
        self.output_min = nnx.Param(jnp.full((), -jnp.inf, dtype=param_dtype))
        self.output_max = nnx.Param(jnp.full((), jnp.inf, dtype=param_dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.clip(x, self.input_min.value, self.input_max.value)
        x = self.linear(x)
        x = jnp.clip(x, self.output_min.value, self.output_max.value)
        return x


class VisionMLP(nnx.Module):
    """SwiGLU MLP for the Gemma4 vision encoder (gelu_pytorch_tanh activation)."""

    def __init__(
        self,
        config: VisionConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        h, ih = config.hidden_size, config.intermediate_size
        self.gate_proj = ClippableLinear(
            h, ih, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.up_proj = ClippableLinear(
            h, ih, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.down_proj = ClippableLinear(
            ih, h, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # gelu_pytorch_tanh ≡ jax.nn.gelu(..., approximate=True)
        return self.down_proj(
            jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x)
        )


class VisionEncoderLayer(nnx.Module):
    """Gemma4 vision encoder layer with 4 RMSNorms.

    Unlike the 2-norm Qwen3-VL block, Gemma4 applies separate pre/post norms for
    both attention and FFW, matching the HF Gemma4VisionEncoderLayer pattern.
    """

    def __init__(
        self,
        config: VisionConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        h, eps = config.hidden_size, config.rms_norm_eps
        kw = dict(with_scale=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.input_layernorm = RMSNorm(h, eps=eps, **kw)
        self.post_attention_layernorm = RMSNorm(h, eps=eps, **kw)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps, **kw)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps, **kw)
        self.self_attn = VisionAttention(
            config, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.mlp = VisionMLP(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> jax.Array:
        # Attention block: pre_norm → attn → post_norm → residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, pixel_position_ids)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # FFW block: pre_norm → mlp → post_norm → residual
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x


class VisionPooler(nnx.Module):
    """2D average-pooling over k×k windows plus sqrt(hidden_size) scaling.

    The pooling uses pixel_position_ids to assign each patch to a pool slot:
      kernel_idx = floor(x / k) + (max_x // k) * floor(y / k)

    This groups patches into non-overlapping k×k spatial windows.  The output
    is scaled by sqrt(hidden_size), matching HF Gemma4VisionPooler.
    """

    def __init__(self, config: VisionConfig):
        self.k = config.pooling_kernel_size
        self.scale = float(config.hidden_size) ** 0.5

    def __call__(
        self,
        hidden_states: jax.Array,
        pixel_position_ids: jax.Array,
        output_length: int,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
          hidden_states: [B, N, H].
          pixel_position_ids: [B, N, 2] int (x, y); -1 for padding.
          output_length: static number of pool slots = N // k^2.

        Returns:
          pooled: [B, output_length, H].
          valid_mask: [B, output_length] bool — True for non-zero (valid) slots.
        """
        k = self.k
        B, N, H = hidden_states.shape

        # Clamp padding coords (−1) to 0; padding patches get zero weight anyway.
        clamped = jnp.clip(pixel_position_ids, 0, None)  # [B, N, 2]

        # max_x per image = highest x-coord + 1.
        valid = ~(pixel_position_ids == -1).all(axis=-1)  # [B, N]
        max_x = (clamped[..., 0] * valid).max(axis=-1, keepdims=True) + 1  # [B, 1]

        kx = clamped[..., 0] // k  # [B, N]
        ky = clamped[..., 1] // k  # [B, N]
        kernel_idx = kx + (max_x // k) * ky  # [B, N]

        # Average pool: weight = 1/k^2 for valid patches, 0 for padding.
        weights = jax.nn.one_hot(kernel_idx, output_length, dtype=jnp.float32) / float(
            k * k
        )  # [B, N, L]
        weights = jnp.where(valid[:, :, None], weights, 0.0)

        pooled = jnp.einsum(
            "bnl,bnh->blh", weights, hidden_states.astype(jnp.float32)
        ).astype(
            hidden_states.dtype
        )  # [B, L, H]

        pooled = pooled * self.scale  # scale by sqrt(hidden_size)
        valid_mask = weights.sum(axis=1) > 0.0  # [B, L]
        return pooled, valid_mask


class VisionEncoder(nnx.Module):

    def __init__(
        self,
        config: VisionConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List(
            [
                VisionEncoderLayer(
                    config, dtype=dtype, param_dtype=param_dtype, rngs=rngs
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(x, cos, sin, pixel_position_ids)
        return x


class VisionProjector(nnx.Module):
    """Projects vision features into language-model embedding space.

    Matches HF Gemma4MultimodalEmbedder:
      1. Scaleless RMSNorm (no learnable weight — no checkpoint tensor).
      2. Linear projection (no bias).
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        rms_norm_eps: float,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.embedding_pre_projection_norm = RMSNorm(
            vision_hidden_size,
            eps=rms_norm_eps,
            with_scale=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.embedding_projection = nnx.Linear(
            vision_hidden_size,
            text_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding_pre_projection_norm(x)
        return self.embedding_projection(x)


class VisionModel(nnx.Module):
    """Full Gemma4 vision encoder: patch embed → transformer → pool."""

    def __init__(
        self,
        config: VisionConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.patch_embedder = VisionPatchEmbedder(
            config, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.encoder = VisionEncoder(
            config, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.pooler = VisionPooler(config)

    def __call__(
        self,
        pixel_values: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Encode a batch of images.

        Args:
          pixel_values: [B, N, patch_channels] float32 in [0, 1].  N must equal
            some integer multiple of pooling_kernel_size^2 so that output_length
            = N // k^2 is an integer.
          pixel_position_ids: [B, N, 2] int32 (x, y) patch positions; -1 for
            padding patches added to align images to a common N.

        Returns:
          pooled: [B, output_length, hidden_size] — the pooled patch features.
            Valid (non-padding) slots contain real features; the rest are zero.
          valid_mask: [B, output_length] bool — True for valid output slots.
        """
        k = self.config.pooling_kernel_size
        N = pixel_values.shape[1]
        output_length = N // (k * k)

        # Embed patches and add 2D positional embeddings.
        x = self.patch_embedder(pixel_values, pixel_position_ids)  # [B, N, H]

        # Compute 2D RoPE cosine/sine for q/k.
        cos, sin = compute_2d_rope(
            pixel_position_ids, self.config.head_dim, self.config.rope_theta
        )

        # Run the vision transformer.
        x = self.encoder(x, cos, sin, pixel_position_ids)  # [B, N, H]

        # 2D average pooling.
        pooled, valid_mask = self.pooler(x, pixel_position_ids, output_length)

        return pooled, valid_mask
