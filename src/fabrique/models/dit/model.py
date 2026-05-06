"""Diffusion Transformer (DiT) for pixel-space image generation.

Architecture follows DiT (Peebles & Xie 2023) adapted from the Qwen3-VL ViT:
- 2D RoPE positional encoding (same construction as Qwen3-VL vision encoder)
- Timestep conditioning via zero-initialized gate injection per block
- Batch-first tensor layout throughout: [B, N, D]

Zero-init gate injection:
  At initialisation cond_proj weights/biases are zero, so
  tanh(cond_proj(cond)) == 0 and the block is a pure ViT residual stream.
  This means pretrained Qwen3-VL ViT weights can be loaded into the
  corresponding norm/attn/mlp sub-modules with no disruption.

Pretrained weight compatibility (DiTBlock ↔ Qwen3-VL VisionBlock):
  norm1, norm2  → identical LayerNorm shapes
  attn.qkv_proj → same shape (hidden → 3*hidden, bias=True)
  attn.out_proj → same shape (hidden → hidden, bias=True)
  mlp.linear1   → same shape (hidden → intermediate, bias=True)
  mlp.linear2   → same shape (intermediate → hidden, bias=True)
  cond_proj     → NEW, zero-init, no pretrained counterpart
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotate the second half of the last dimension, used by RoPE."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def patchify(images: jax.Array, patch_size: int) -> jax.Array:
    """Split images into non-overlapping patches.

    Args:
      images: float array [B, H, W, C].
      patch_size: spatial patch side length P. H and W must be divisible by P.

    Returns:
      Float array [B, (H/P)*(W/P), P*P*C].
    """
    B, H, W, C = images.shape
    P = patch_size
    ph, pw = H // P, W // P
    x = images.reshape(B, ph, P, pw, P, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, ph, pw, P, P, C]
    return x.reshape(B, ph * pw, P * P * C)


def unpatchify(
    patches: jax.Array,
    patch_size: int,
    height: int,
    width: int,
    channels: int,
) -> jax.Array:
    """Reconstruct images from patches (inverse of patchify).

    Args:
      patches: float array [B, (H/P)*(W/P), P*P*C].
      patch_size: spatial patch side length P.
      height, width, channels: target image shape.

    Returns:
      Float array [B, H, W, C].
    """
    B = patches.shape[0]
    P = patch_size
    ph, pw = height // P, width // P
    x = patches.reshape(B, ph, pw, P, P, channels)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, ph, P, pw, P, C]
    return x.reshape(B, height, width, channels)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def compute_rope(
    height: int,
    width: int,
    head_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Compute 2D RoPE cos/sin tables for a height×width patch grid.

    Call **outside** the JIT boundary; pass the results into the model's
    __call__ each forward step (they are static w.r.t. a fixed image size).

    Construction matches the Qwen3-VL vision encoder so that pretrained
    positional encodings transfer directly when height == width ==
    sqrt(num_position_embeddings) and patch_size matches.

    Args:
      height: number of patches along the vertical axis (H // patch_size).
      width:  number of patches along the horizontal axis (W // patch_size).
      head_dim: per-head feature dimension (hidden_size // num_heads).

    Returns:
      cos, sin: float32 JAX arrays of shape [height*width, head_dim].
    """
    rotary_dim = head_dim // 2  # e.g. 32 for head_dim=64
    inv_freq = 1.0 / (
        10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )  # [rotary_dim//2]
    max_len = max(height, width)
    table = np.outer(
        np.arange(max_len, dtype=np.float32), inv_freq
    )  # [max_len, rotary_dim//2]

    h_idx = np.repeat(np.arange(height), width)  # [H*W]
    w_idx = np.tile(np.arange(width), height)  # [H*W]

    # Gather h and w frequencies then concatenate → [H*W, rotary_dim]
    rotary_emb = np.concatenate([table[h_idx], table[w_idx]], axis=-1)
    # Double for rotate_half compatibility: [H*W, head_dim]
    emb = np.concatenate([rotary_emb, rotary_emb], axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return jnp.asarray(cos), jnp.asarray(sin)


# ---------------------------------------------------------------------------
# Config and presets
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DiTConfig:
    """Configuration for the Diffusion Transformer.

    Attributes:
      hidden_size: token embedding dimension.
      depth: number of transformer blocks.
      num_heads: number of attention heads.
      intermediate_size: MLP hidden dimension.
      patch_size: spatial patch side length in pixels.
      in_channels: number of image channels (3 for RGB).
      freq_embed_size: sinusoidal frequency embedding size fed into the
        timestep MLP (must be even).
    """

    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 4
    in_channels: int = 3
    freq_embed_size: int = 256


def dit_small() -> DiTConfig:
    """ViT-S/4 sized model — fast to iterate on."""
    return DiTConfig(hidden_size=384, depth=12, num_heads=6, intermediate_size=1536)


def dit_base() -> DiTConfig:
    """ViT-B/4 sized model."""
    return DiTConfig(hidden_size=768, depth=12, num_heads=12, intermediate_size=3072)


def dit_qwen() -> DiTConfig:
    """Matches Qwen3-VL-2B/4B vision encoder dimensions.

    Use this when initialising from pretrained Qwen3-VL ViT weights.
    Set patch_size=16 to match the Qwen3-VL patch size.
    """
    return DiTConfig(hidden_size=1024, depth=24, num_heads=16, intermediate_size=4096)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class TimestepEmbedding(nnx.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP.

    Maps a scalar timestep t ∈ [0, 1] to a conditioning vector of shape
    [hidden_size] per batch element.
    """

    def __init__(
        self,
        hidden_size: int,
        freq_embed_size: int = 256,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.freq_embed_size = freq_embed_size
        self.linear1 = nnx.Linear(
            freq_embed_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @staticmethod
    def sinusoidal(t: jax.Array, dim: int) -> jax.Array:
        """Compute sinusoidal embedding for a batch of scalars.

        Args:
          t: float array [B], values in [0, 1].
          dim: output dimension (must be even).

        Returns:
          float32 array [B, dim].
        """
        half = dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None, :]  # [B, half]
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    def __call__(self, t: jax.Array) -> jax.Array:
        """t: float [B] ∈ [0,1]  →  [B, hidden_size]"""
        x = self.sinusoidal(t, self.freq_embed_size)  # float32 [B, freq_embed_size]
        x = self.linear1(x)
        x = jax.nn.silu(x)
        x = self.linear2(x)
        return x


class PatchEmbed(nnx.Module):
    """Linear projection of flattened image patches to hidden_size.

    Weight-compatible with Qwen3-VL VisionPatchEmbed when patch_volume
    = in_channels * patch_size² (temporal_patch_size=1 for still images).
    """

    def __init__(
        self,
        hidden_size: int,
        patch_volume: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.dtype = dtype
        self.proj = nnx.Linear(
            patch_volume,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [..., patch_volume]  →  [..., hidden_size]"""
        return self.proj(x.astype(self.dtype))


class Attention(nnx.Module):
    """Multi-head self-attention with 2D RoPE.

    Operates on batch-first tensors [B, N, D]. Weight-compatible with
    Qwen3-VL VisionAttention (same qkv_proj and out_proj shapes).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.dtype = dtype
        self.qkv_proj = nnx.Linear(
            hidden_size,
            3 * hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
    ) -> jax.Array:
        """
        Args:
          x:   [B, N, D]
          cos: [N, head_dim]  float32, from compute_rope
          sin: [N, head_dim]  float32, from compute_rope
        Returns:
          [B, N, D]
        """
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, N, 3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [B, N, D]

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)

        # Apply 2D RoPE in float32 then cast back — matches Qwen3-VL convention
        cos_ = cos[None, :, None, :].astype(jnp.float32)  # [1, N, 1, head_dim]
        sin_ = sin[None, :, None, :].astype(jnp.float32)
        q_f, k_f = q.astype(jnp.float32), k.astype(jnp.float32)
        q = (q_f * cos_ + rotate_half(q_f) * sin_).astype(self.dtype)
        k = (k_f * cos_ + rotate_half(k_f) * sin_).astype(self.dtype)

        # Rearrange to [B, H, N, head_dim] for the attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.astype(self.dtype).transpose(0, 2, 1, 3)

        # Scaled dot-product attention — full, no mask (single image per slot)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale  # [B, H, N, N]
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(self.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)  # [B, H, N, head_dim]

        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.hidden_size)
        return self.out_proj(out)


class MLP(nnx.Module):
    """Two-layer feed-forward network with GELU activation.

    Weight-compatible with Qwen3-VL VisionMLP (linear1/linear2 same shapes).
    nnx.Linear handles arbitrary leading batch dims, so this works for both
    [N, D] (Qwen style) and [B, N, D] (DiT style).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear2(jax.nn.gelu(self.linear1(x), approximate=True))


class DiTBlock(nnx.Module):
    """Transformer block with zero-init gate conditioning.

    Conditioning signal (e.g. timestep embedding) is injected after the
    attention and MLP sub-layers via a learned gate initialised to zero:

        gate_attn, gate_mlp = split(cond_proj(cond), 2)
        x = x + tanh(gate_attn) * Attention(norm1(x))
        x = x + tanh(gate_mlp)  * MLP(norm2(x))

    Because cond_proj is zero-initialised, tanh(0) = 0 at the start of
    training: the block is an exact ViT residual block and pretrained Qwen3-VL
    weights for norm1/norm2/attn/mlp transfer without any numerical disruption.

    Pretrained weight compatibility:
      norm1, norm2  ← VisionBlock.norm1/norm2  (LayerNorm, same dim)
      attn          ← VisionBlock.attn         (qkv_proj + out_proj, same shapes)
      mlp           ← VisionBlock.mlp          (linear1 + linear2, same shapes)
      cond_proj     ← NEW, zero-init
    """

    def __init__(
        self,
        config: DiTConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            use_fast_variance=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            use_fast_variance=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attn = Attention(
            config.hidden_size,
            config.num_heads,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = MLP(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Zero-init: gate == 0 at init → tanh(0) == 0 → no conditioning at init
        self.cond_proj = nnx.Linear(
            config.hidden_size,
            2 * config.hidden_size,
            use_bias=True,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        """
        Args:
          x:    [B, N, D]  patch token sequence
          cos:  [N, head_dim]  float32 RoPE cosines
          sin:  [N, head_dim]  float32 RoPE sines
          cond: [B, D]  conditioning vector (e.g. timestep embedding)
        Returns:
          [B, N, D]
        """
        gate = self.cond_proj(cond)  # [B, 2D]
        gate_attn, gate_mlp = jnp.split(gate, 2, axis=-1)  # each [B, D]
        x = x + jnp.tanh(gate_attn)[:, None, :] * self.attn(self.norm1(x), cos, sin)
        x = x + jnp.tanh(gate_mlp)[:, None, :] * self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class DiT(nnx.Module):
    """Diffusion Transformer for pixel-space image generation.

    The forward pass maps (noisy_images, timesteps) → predicted_velocity.
    Under flow matching the training target is:
        velocity = x_data - x_noise   (constant along the linear interpolant)

    The noise_head is zero-initialised so the network predicts zero velocity
    at initialisation, which is a stable starting point.

    Example::

        config = dit_small()
        model = DiT(config, rngs=nnx.Rngs(0))

        # Precompute RoPE tables once (outside JIT) for a fixed image size
        H, W = 64, 64
        cos, sin = compute_rope(
            H // config.patch_size,
            W // config.patch_size,
            config.hidden_size // config.num_heads,
        )

        # Training step
        v_pred = model(noisy_images, t, cos, sin)   # [B, H, W, C]
        loss = jnp.mean((v_pred - target_velocity) ** 2)
    """

    def __init__(
        self,
        config: DiTConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        patch_volume = config.in_channels * config.patch_size**2

        self.patch_embed = PatchEmbed(
            config.hidden_size,
            patch_volume,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.time_embed = TimestepEmbedding(
            config.hidden_size,
            config.freq_embed_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.blocks = nnx.List(
            [
                DiTBlock(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            use_fast_variance=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Zero-init: model predicts zero velocity at the start of training
        self.noise_head = nnx.Linear(
            config.hidden_size,
            patch_volume,
            use_bias=True,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
    ) -> jax.Array:
        """Predict flow-matching velocity for a batch of noisy images.

        Args:
          x:   noisy images,  float [B, H, W, C]
          t:   timesteps,     float [B], values in [0, 1]
          cos: RoPE cosines,  float32 [N, head_dim]  (from compute_rope)
          sin: RoPE sines,    float32 [N, head_dim]  (from compute_rope)

        Returns:
          Predicted velocity, float [B, H, W, C], same shape as x.
        """
        cfg = self.config
        B, H, W, C = x.shape
        P = cfg.patch_size

        # Patchify and embed: [B, N, D]
        tokens = self.patch_embed(patchify(x, P))

        # Timestep conditioning: [B, D]
        cond = self.time_embed(t)

        for block in self.blocks:
            tokens = block(tokens, cos, sin, cond)

        tokens = self.final_norm(tokens)

        # Velocity per patch: [B, N, patch_volume]
        v_patches = self.noise_head(tokens)

        # Reconstruct spatial layout: [B, H, W, C]
        return unpatchify(v_patches, P, H, W, C)
