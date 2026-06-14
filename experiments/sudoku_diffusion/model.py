"""A small, dense, DiffusionGemma-like model in JAX / Flax NNX.

This re-implements the *architecture* of Google's DiffusionGemma
(https://github.com/huggingface/transformers/blob/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py)
for the Sudoku toy task, with these deliberate departures from the original:

* **Framework:** JAX / Flax NNX instead of PyTorch.
* **Dense, not MoE:** DiffusionGemma is a 26B mixture-of-experts model; here
  every layer is a single dense GeGLU MLP.
* **From scratch:** no pretrained weights and a custom 12-token vocabulary
  (see ``tokenizer.py``); the embedding table is therefore tiny.
* **Small by default:** the default config is ~45M params (~90 MB in bf16), but
  every dimension is configurable.
* **No encoder / vision / KV cache:** Sudoku has no separate prompt and no
  images.  The whole board is a single bidirectional "canvas" that is denoised
  in place, so we keep only the decoder and run full forward passes.

What we *keep* from DiffusionGemma / the Gemma family, because it is what makes
the architecture what it is:

* **Bidirectional attention.**  Diffusion denoises every position at once, so
  there is no causal mask -- each token attends to all (non-pad) tokens.
* **Gemma "sandwich" normalization:** an RMSNorm before *and* after both the
  attention and the feed-forward sub-blocks.
* **QK-norm:** RMSNorm applied to per-head query/key vectors.
* **Rotary position embeddings (RoPE)**, with optional sliding-window layers.
* **GeGLU** feed-forward with the tanh-approximated GELU activation.
* **Input-embedding scaling** by ``sqrt(embed_dim)`` and **final-logit
  softcapping** (``tanh`` clamp).
* **Self-conditioning:** the model can be fed its own soft predictions from the
  previous denoising step, mixed back into the input embeddings.

The *training/sampling* diffusion loop lives elsewhere; this file only provides
the network and is written so a single forward pass is all the loop needs::

    logits = model(input_tokens, padding_mask=..., self_cond_logits=...)
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jaxtyping
from flax import nnx
from jax import numpy as jnp

# Large negative value used to mask out attention scores before softmax.
K_MASK = -2.3819763e38


@dataclasses.dataclass(slots=True)
class ModelConfig:
    """Configuration for :class:`SudokuDiffusion`.

    Sizes are intentionally explicit (no "auto" head_dim) so the parameter
    count is easy to reason about.
    """

    vocab_size: int
    embed_dim: int
    hidden_dim: int  # GeGLU intermediate size
    num_layers: int
    num_heads: int
    head_dim: int
    num_kv_heads: int  # < num_heads enables grouped-query attention
    max_seq_len: int = 128
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    # Optional alternating local/global attention, Gemma-3 style.  ``None`` means
    # every layer is global (full) attention -- the sensible default for the
    # ~81-token Sudoku canvas.  Otherwise a window size in tokens; every
    # ``global_attn_every`` layer stays global.
    sliding_window: int | None = None
    global_attn_every: int = 6
    # Softcapping (``cap * tanh(x / cap)``); ``None`` disables.
    final_logit_softcapping: float | None = 30.0
    attn_logit_softcapping: float | None = None
    use_tied_embedding: bool = True
    use_self_conditioning: bool = True
    param_dtype: jnp.dtype = jnp.bfloat16

    @classmethod
    def sudoku_tiny(cls, vocab_size: int = 12) -> "ModelConfig":
        """~7M params -- fast smoke tests."""
        return cls(
            vocab_size=vocab_size,
            embed_dim=256,
            hidden_dim=1024,
            num_layers=6,
            num_heads=4,
            head_dim=64,
            num_kv_heads=4,
        )

    @classmethod
    def sudoku_small(cls, vocab_size: int = 12) -> "ModelConfig":
        """~45M params (~90 MB in bf16) -- the default."""
        return cls(
            vocab_size=vocab_size,
            embed_dim=512,
            hidden_dim=2048,
            num_layers=10,
            num_heads=8,
            head_dim=64,
            num_kv_heads=8,
        )

    @classmethod
    def sudoku_base(cls, vocab_size: int = 12) -> "ModelConfig":
        """~150M params -- when the small model isn't enough."""
        return cls(
            vocab_size=vocab_size,
            embed_dim=768,
            hidden_dim=3072,
            num_layers=18,
            num_heads=12,
            head_dim=64,
            num_kv_heads=4,
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nnx.Module):
    """Gemma-style RMSNorm: normalize, then scale by ``(1 + weight)``.

    The ``+1`` (with a zero-initialized weight) means the layer starts as a pure
    normalization and learns a *deviation* from unit scale.
    """

    def __init__(
        self,
        dim: int,
        *,
        norm_eps: float,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.w = nnx.Param(
            nnx.initializers.zeros_init()(rngs.params(), (dim,)).astype(param_dtype)
        )
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        rms_inv = jax.lax.rsqrt(
            jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.norm_eps
        )
        normed = (x_f32 * rms_inv).astype(dtype)
        return normed * (1.0 + self.w.astype(dtype))


def apply_rope(
    x: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [B, L]
    *,
    rope_theta: float,
) -> jaxtyping.Array:
    """Rotary position embedding (GPT-NeoX "rotate-half" layout)."""
    head_dim = x.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction  # [H/2]
    sinusoid = positions[..., None].astype(jnp.float32) / timescale  # [B, L, H/2]
    sin = jnp.sin(sinusoid)[:, :, None, :]  # [B, L, 1, H/2]
    cos = jnp.cos(sinusoid)[:, :, None, :]
    first, second = jnp.split(x, 2, axis=-1)
    out = jnp.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )
    return out.astype(x.dtype)


class Embedder(nnx.Module):
    """Token embedding table with Gemma-style ``sqrt(embed_dim)`` scaling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.input_embedding = nnx.Param(
            nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)).astype(
                param_dtype
            )
        )
        self.embed_scale = math.sqrt(embed_dim)

    def encode(self, tokens: jaxtyping.Array) -> jaxtyping.Array:
        x = self.input_embedding[(tokens,)]
        return x * jnp.astype(self.embed_scale, x.dtype)

    def decode(self, x: jaxtyping.Array) -> jaxtyping.Array:
        """Project hidden states back to vocabulary logits (tied weights)."""
        return jnp.dot(x, self.input_embedding.T)

    def soft_encode(self, probs: jaxtyping.Array) -> jaxtyping.Array:
        """Embed a *distribution* over the vocabulary (for self-conditioning)."""
        x = jnp.dot(probs, self.input_embedding.astype(probs.dtype))
        return x * jnp.astype(self.embed_scale, x.dtype)


def repeat_kv(x: jaxtyping.Array, n_rep: int) -> jaxtyping.Array:
    """Expand [B, L, n_kv, H] to [B, L, n_kv * n_rep, H] for grouped-query attn."""
    if n_rep == 1:
        return x
    b, l, n_kv, h = x.shape
    x = jnp.broadcast_to(x[:, :, :, None, :], (b, l, n_kv, n_rep, h))
    return x.reshape(b, l, n_kv * n_rep, h)


class Attention(nnx.Module):
    """Bidirectional multi-head attention with QK-norm and RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        q_out = config.num_heads * config.head_dim
        kv_out = config.num_kv_heads * config.head_dim
        linear = lambda i, o: nnx.Linear(
            i, o, use_bias=False, rngs=rngs, param_dtype=config.param_dtype
        )
        self.q_proj = linear(config.embed_dim, q_out)
        self.k_proj = linear(config.embed_dim, kv_out)
        self.v_proj = linear(config.embed_dim, kv_out)
        self.o_proj = linear(q_out, config.embed_dim)
        self.q_norm = RMSNorm(
            config.head_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )
        self.n_rep = config.num_heads // config.num_kv_heads
        self.scale = config.head_dim**-0.5

    @jax.named_scope("attention")
    def __call__(
        self,
        x: jaxtyping.Array,  # [B, L, D]
        positions: jaxtyping.Array,  # [B, L]
        attn_mask: jaxtyping.Array,  # [B, L, L] bool, True = attend
    ) -> jaxtyping.Array:
        b, l, _ = x.shape
        cfg = self.config

        q = self.q_norm(self.q_proj(x).reshape(b, l, cfg.num_heads, cfg.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, l, cfg.num_kv_heads, cfg.head_dim))
        v = self.v_proj(x).reshape(b, l, cfg.num_kv_heads, cfg.head_dim)

        q = apply_rope(q, positions, rope_theta=cfg.rope_theta)
        k = apply_rope(k, positions, rope_theta=cfg.rope_theta)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # [B, N, L, H]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = jnp.einsum("bntd,bnsd->bnts", q, k) * self.scale
        if cfg.attn_logit_softcapping is not None:
            cap = cfg.attn_logit_softcapping
            scores = cap * jnp.tanh(scores / cap)
        scores = jnp.where(attn_mask[:, None, :, :], scores, K_MASK)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)

        out = jnp.einsum("bnts,bnsd->bntd", weights, v)  # [B, N, L, H]
        out = out.transpose(0, 2, 1, 3).reshape(b, l, cfg.num_heads * cfg.head_dim)
        return self.o_proj(out)


class MLP(nnx.Module):
    """GeGLU feed-forward (tanh-approx GELU), as in Gemma."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        linear = lambda i, o: nnx.Linear(
            i, o, use_bias=False, rngs=rngs, param_dtype=config.param_dtype
        )
        self.gate_proj = linear(config.embed_dim, config.hidden_dim)
        self.up_proj = linear(config.embed_dim, config.hidden_dim)
        self.down_proj = linear(config.hidden_dim, config.embed_dim)

    @jax.named_scope("mlp")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        gate = nnx.gelu(self.gate_proj(x), approximate=True)
        return self.down_proj(gate * self.up_proj(x))


class DecoderLayer(nnx.Module):
    """One transformer block with Gemma sandwich normalization."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        norm = lambda: RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )
        self.input_layernorm = norm()
        self.attn = Attention(config, rngs=rngs)
        self.post_attention_layernorm = norm()
        self.pre_feedforward_layernorm = norm()
        self.mlp = MLP(config, rngs=rngs)
        self.post_feedforward_layernorm = norm()

    def __call__(
        self,
        x: jaxtyping.Array,
        positions: jaxtyping.Array,
        attn_mask: jaxtyping.Array,
    ) -> jaxtyping.Array:
        # Attention sub-block (norm before and after, then residual add).
        attn_out = self.attn(self.input_layernorm(x), positions, attn_mask)
        x = x + self.post_attention_layernorm(attn_out)
        # Feed-forward sub-block (same sandwich pattern).
        ffn_out = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(ffn_out)
        return x


class SelfConditioning(nnx.Module):
    """Mixes the previous denoising step's soft predictions into the input.

    Given input token embeddings and a "self-conditioning signal" (the soft
    embedding of the previous step's predicted distribution), produce refined
    input embeddings.  Disabling self-conditioning is just passing a zero signal.
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        linear = lambda i, o: nnx.Linear(
            i, o, use_bias=False, rngs=rngs, param_dtype=config.param_dtype
        )
        self.pre_norm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )
        self.gate_proj = linear(config.embed_dim, config.hidden_dim)
        self.up_proj = linear(config.embed_dim, config.hidden_dim)
        self.down_proj = linear(config.hidden_dim, config.embed_dim)
        self.post_norm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )

    def __call__(
        self, inputs_embeds: jaxtyping.Array, sc_signal: jaxtyping.Array
    ) -> jaxtyping.Array:
        normed = self.pre_norm(sc_signal)
        gate = nnx.gelu(self.gate_proj(normed), approximate=True)
        sc = self.down_proj(gate * self.up_proj(normed))
        return self.post_norm(inputs_embeds + sc)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SudokuDiffusion(nnx.Module):
    """Dense, DiffusionGemma-like denoiser for the Sudoku canvas."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embedder = Embedder(
            config.vocab_size,
            config.embed_dim,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )
        if config.use_self_conditioning:
            self.self_conditioning = SelfConditioning(config, rngs=rngs)
        self.layers = nnx.List(
            [DecoderLayer(config, rngs=rngs) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            param_dtype=config.param_dtype,
        )
        if not config.use_tied_embedding:
            self.lm_head = nnx.Linear(
                config.embed_dim,
                config.vocab_size,
                use_bias=False,
                rngs=rngs,
                param_dtype=config.param_dtype,
            )

    def _layer_mask(
        self,
        layer_idx: int,
        base_mask: jaxtyping.Array,  # [B, L, L]
        positions: jaxtyping.Array,  # [B, L]
    ) -> jaxtyping.Array:
        """Add a (bidirectional) sliding window for local-attention layers."""
        cfg = self.config
        if cfg.sliding_window is None or (layer_idx + 1) % cfg.global_attn_every == 0:
            return base_mask
        dist = jnp.abs(positions[:, :, None] - positions[:, None, :])  # [B, L, L]
        return base_mask & (dist < cfg.sliding_window)

    @jax.named_scope("sudoku_diffusion")
    def __call__(
        self,
        input_tokens: jaxtyping.Array,  # [B, L] int
        *,
        padding_mask: jaxtyping.Array | None = None,  # [B, L] bool/int, 1 = real token
        positions: jaxtyping.Array | None = None,  # [B, L] int
        self_cond_logits: jaxtyping.Array | None = None,  # [B, L, V] float
    ) -> jaxtyping.Array:
        """Run one denoising forward pass.

        Returns logits of shape ``[B, L, vocab_size]`` -- a predicted
        distribution over tokens for *every* position (masked positions are the
        ones the diffusion loop actually resamples).
        """
        cfg = self.config
        b, l = input_tokens.shape

        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(l, dtype=jnp.int32), (b, l))

        # Bidirectional base mask: attend everywhere except padded keys.
        base_mask = jnp.ones((b, l, l), dtype=jnp.bool_)
        if padding_mask is not None:
            base_mask = base_mask & padding_mask.astype(jnp.bool_)[:, None, :]

        x = self.embedder.encode(input_tokens)  # [B, L, D]

        if cfg.use_self_conditioning:
            if self_cond_logits is not None:
                probs = jax.nn.softmax(self_cond_logits.astype(jnp.float32), axis=-1)
                sc_signal = self.embedder.soft_encode(probs.astype(x.dtype))
            else:
                sc_signal = jnp.zeros_like(x)
            x = self.self_conditioning(x, sc_signal)

        for i, layer in enumerate(self.layers):
            x = layer(x, positions, self._layer_mask(i, base_mask, positions))

        x = self.final_norm(x)
        logits = self.embedder.decode(x) if cfg.use_tied_embedding else self.lm_head(x)
        logits = logits.astype(jnp.float32)

        if cfg.final_logit_softcapping is not None:
            cap = cfg.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits

    def get_model_input(self) -> dict:
        """A dummy input for tracing / sharding initialization."""
        b, l = 2, self.config.max_seq_len
        return {
            "input_tokens": jnp.ones((b, l), dtype=jnp.int32),
            "padding_mask": jnp.ones((b, l), dtype=jnp.bool_),
        }


def count_params(model: nnx.Module) -> int:
    """Total number of scalar parameters in ``model``."""
    params = nnx.state(model, nnx.Param)
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
