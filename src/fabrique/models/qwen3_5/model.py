"""Qwen3.5 (dense) model in JAX/Flax NNX.

Text-only, matching HuggingFace ``Qwen3_5ForCausalLM``.  Qwen3.5 interleaves two
kinds of token mixer -- ``[linear_attention x3, full_attention]`` repeated -- so
three quarters of the layers are Gated DeltaNet (linear attention with a delta
rule and a forget gate) and one quarter is gated GQA softmax attention.

Deliberately out of scope: the vision tower (`model.visual.*` in the
checkpoints), the Multi-Token-Prediction head, and the MoE variants
(`qwen3_5_moe`).  Structure follows the sibling ``qwen3vl`` package.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import json
import os

import flax
import jax
import jax.sharding as shd
import jaxtyping
from flax import nnx
from jax import numpy as jnp
from jax.interpreters import pxla
from tunix.utils import compat, env_utils

# The cuDNN-probing helper lives in the qwen3vl package; it is model-agnostic and
# subtle enough (a mismatched cuDNN fails only at runtime) that duplicating it
# would be worse than the cross-package import.  Candidate for extraction into a
# shared module if a third model needs it.
from fabrique.models.qwen3vl.vision import attention_impl_kwargs

env_utils.setup_sharding_environment()

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


class RematConfig(enum.Enum):
    NONE = "none"
    BLOCK = "block"


@dataclasses.dataclass(frozen=True)
class ShardingConfig:
    """Sharding configuration for Qwen3.5."""

    emb_vd: tuple[str | None, ...]
    emb_dv: tuple[str | None, ...]
    q_weight_dnh: tuple[str | None, ...]
    kv_weight_dnh: tuple[str | None, ...]
    o_weight_nhd: tuple[str | None, ...]
    ffw_weight_df: tuple[str | None, ...]
    ffw_weight_fd: tuple[str | None, ...]
    rms_norm_weight: tuple[str | None, ...]
    act_btd: tuple[str | None, ...]
    act_btf: tuple[str | None, ...]
    act_btnh: tuple[str | None, ...]
    # Gated DeltaNet
    gdn_weight_dz: tuple[str | None, ...]
    gdn_weight_zd: tuple[str | None, ...]
    gdn_conv_weight: tuple[str | None, ...]
    gdn_head_scalar: tuple[str | None, ...]

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None
        return ShardingConfig(
            emb_vd=("tp", fsdp),
            emb_dv=(fsdp, "tp"),
            q_weight_dnh=(fsdp, "tp", None),
            kv_weight_dnh=(fsdp, "tp", None),
            o_weight_nhd=("tp", None, fsdp),
            ffw_weight_df=(fsdp, "tp"),
            ffw_weight_fd=("tp", fsdp),
            rms_norm_weight=("tp",),
            act_btd=("fsdp", None, None if is_sampling else "tp"),
            act_btf=("fsdp", None, "tp"),
            act_btnh=("fsdp", None, "tp", None),
            gdn_weight_dz=(fsdp, "tp"),
            gdn_weight_zd=("tp", fsdp),
            gdn_conv_weight=("tp", None),
            gdn_head_scalar=("tp",),
        )


def _hybrid_layer_types(
    num_layers: int, full_attention_interval: int
) -> tuple[str, ...]:
    """``[linear, linear, linear, full]``-style pattern, as HF builds it."""
    return tuple(
        FULL_ATTENTION if (i + 1) % full_attention_interval == 0 else LINEAR_ATTENTION
        for i in range(num_layers)
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
    """Configuration for the dense Qwen3.5 text model."""

    num_layers: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    # Gated DeltaNet
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int = 4
    # layout
    full_attention_interval: int = 4
    layer_types: tuple[str, ...] = ()
    # attention
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, ...] = (11, 11, 10)
    use_tied_embedding: bool = True
    # execution
    mamba_ssm_dtype: jnp.dtype = jnp.float32
    gdn_chunk_size: int = 64
    param_dtype: jnp.dtype = jnp.bfloat16
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
    remat_config: RematConfig = RematConfig.NONE

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = _hybrid_layer_types(
                self.num_layers, self.full_attention_interval
            )
        if len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_layers is {self.num_layers}"
            )

    @property
    def rotary_dim(self) -> int:
        """Number of head dims that RoPE actually rotates (partial RoPE)."""
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_hf_config(cls, path: str, **overrides) -> ModelConfig:
        """Build a config from a HuggingFace ``config.json``.

        Preferred over the hardcoded size classmethods: the checkpoint is the
        source of truth, and a silent mismatch between a hardcoded config and
        the weights is hard to debug.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            raw = json.load(f)
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        kwargs = dict(
            num_layers=text["num_hidden_layers"],
            vocab_size=text["vocab_size"],
            embed_dim=text["hidden_size"],
            hidden_dim=text["intermediate_size"],
            num_heads=text["num_attention_heads"],
            head_dim=text["head_dim"],
            num_kv_heads=text["num_key_value_heads"],
            rope_theta=int(rope.get("rope_theta", text.get("rope_theta", 10_000_000))),
            norm_eps=text["rms_norm_eps"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
            full_attention_interval=text.get("full_attention_interval", 4),
            layer_types=tuple(text.get("layer_types") or ()),
            attn_output_gate=text.get("attn_output_gate", True),
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            mrope_section=tuple(rope.get("mrope_section", (11, 11, 10))),
            # HF omits the key entirely for tied checkpoints in some exports,
            # so treat a missing value as the top-level flag then default True.
            use_tied_embedding=bool(
                text.get("tie_word_embeddings", raw.get("tie_word_embeddings", True))
            ),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # Convenience factories for the sizes whose configs were verified locally.
    # Prefer from_hf_config() -- these are here to match the sibling package's
    # shape and to document the architecture.
    @classmethod
    def qwen3_5_0_8b(cls):
        return cls(
            num_layers=24,
            vocab_size=248320,
            embed_dim=1024,
            hidden_dim=3584,
            num_heads=8,
            head_dim=256,
            num_kv_heads=2,
            rope_theta=10_000_000,
            norm_eps=1e-6,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            use_tied_embedding=True,
        )

    @classmethod
    def qwen3_5_4b(cls):
        return cls(
            num_layers=32,
            vocab_size=248320,
            embed_dim=2560,
            hidden_dim=9216,
            num_heads=16,
            head_dim=256,
            num_kv_heads=4,
            rope_theta=10_000_000,
            norm_eps=1e-6,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            use_tied_embedding=True,
        )

    @classmethod
    def qwen3_5_9b(cls):
        return cls(
            num_layers=32,
            vocab_size=248320,
            embed_dim=4096,
            hidden_dim=12288,
            num_heads=16,
            head_dim=256,
            num_kv_heads=4,
            rope_theta=10_000_000,
            norm_eps=1e-6,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            use_tied_embedding=False,
        )


def shard(x: jnp.ndarray, s: tuple[str, ...]):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty or jax.devices()[0].platform == "cpu":
        return x
    return jax.lax.with_sharding_constraint(
        x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
    )


class Einsum(nnx.Module):
    """Weight tensor applied through a fixed einsum string."""

    def __init__(
        self,
        einsum_str: str,
        shape: flax.typing.Shape,
        *,
        rngs: nnx.Rngs,
        sharding: tuple[str | None, ...],
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = nnx.Param(
            nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
            sharding=sharding,
        )

    @jax.named_scope("einsum")
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.einsum(self.einsum_str, x, self.w)


class Embedder(nnx.Module):
    """Token embedding, optionally reused as the output projection."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.input_embedding = nnx.Param(
            nnx.initializers.normal(dtype=param_dtype)(
                rngs.params(), (vocab_size, embed_dim)
            ),
            sharding=shd_config.emb_vd,
        )

    @jax.named_scope("embedder_encode")
    def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return self.input_embedding[(x,)]

    @jax.named_scope("embedder_decode")
    def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.dot(x, self.input_embedding.T)


class RMSNorm(nnx.Module):
    """RMSNorm with a learned scale, computed in float32 like HF."""

    def __init__(
        self,
        dim: int,
        *,
        rngs: nnx.Rngs,
        norm_eps: float = 1e-6,
        shd_config: ShardingConfig,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        # Zero-initialised: Qwen3.5 scales by (1 + weight), Gemma-style, not by
        # weight directly.  ``RMSNormGated`` below is the *other* convention
        # (ones-init, plain `w * x`) -- the two coexist in this model, so do not
        # unify them.
        self.w = nnx.Param(
            nnx.initializers.zeros_init()(rngs.params(), dim).astype(param_dtype),
            sharding=shd_config.rms_norm_weight,
        )
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        rms_inv = jax.lax.rsqrt(
            jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.norm_eps
        )
        out = x_f32 * rms_inv * (1.0 + self.w.value.astype(jnp.float32))
        return out.astype(dtype)


class RMSNormGated(nnx.Module):
    """RMSNorm followed by a SiLU gate, as used on the DeltaNet output.

    Mirrors HF ``Qwen3_5RMSNormGated``: normalise, scale, *then* multiply by
    ``silu(gate)`` -- the gate is applied after the norm, not before.
    """

    def __init__(
        self,
        dim: int,
        *,
        rngs: nnx.Rngs,
        norm_eps: float = 1e-6,
        shd_config: ShardingConfig,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.w = nnx.Param(
            nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),
            sharding=shd_config.rms_norm_weight,
        )
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm_gated")
    def __call__(self, x: jaxtyping.Array, gate: jaxtyping.Array) -> jaxtyping.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        rms_inv = jax.lax.rsqrt(
            jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.norm_eps
        )
        out = self.w * (x_f32 * rms_inv).astype(dtype)
        return (out * jax.nn.silu(gate.astype(jnp.float32))).astype(dtype)


# ---------------------------------------------------------------------------
# Rotary embedding (interleaved M-RoPE, partial)
# ---------------------------------------------------------------------------


def rotate_half(x: jaxtyping.Array) -> jaxtyping.Array:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rope_cos_sin(
    positions: jaxtyping.Array,  # [3, B, L] -- the (T, H, W) axes
    *,
    rotary_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, ...],
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
    """cos/sin of shape ``[B, L, rotary_dim]`` for interleaved M-RoPE.

    Matches HF ``Qwen3_5TextRotaryEmbedding``: build per-axis frequencies, then
    *recompose* them by overwriting interleaved slots of the T axis with the H
    and W axes, and finally duplicate to the full rotary width.
    """
    inv_freq = 1.0 / (
        rope_theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
    )  # [rotary_dim/2]
    # [3, B, L, rotary_dim/2]
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq

    recomposed = freqs[0]
    for axis, offset in ((1, 1), (2, 2)):  # H then W
        length = mrope_section[axis] * 3
        idx = jnp.arange(offset, length, 3)
        recomposed = recomposed.at[..., idx].set(freqs[axis][..., idx])

    emb = jnp.concatenate([recomposed, recomposed], axis=-1)  # [B, L, rotary_dim]
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(
    x: jaxtyping.Array,  # [B, L, N, H]
    cos: jaxtyping.Array,  # [B, L, rotary_dim]
    sin: jaxtyping.Array,
) -> jaxtyping.Array:
    """Rotate the leading ``rotary_dim`` head dims and pass the rest through."""
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    rotated = x_rot * cos + rotate_half(x_rot) * sin
    return jnp.concatenate([rotated, x_pass], axis=-1).astype(x.dtype)


def make_causal_mask(
    seq_len: int,
    batch_size: int,
    padding_mask: jaxtyping.Array | None = None,
) -> jaxtyping.Array:
    """Lower-triangular mask by *sequence index*, plus optional padding.

    Causality is by position in the sequence, matching HF's
    ``create_causal_mask`` (which derives it from ``cache_position``).  See the
    sibling qwen3vl package for why keying this off M-RoPE positions instead is
    a bug.
    """
    idx = jnp.arange(seq_len)
    mask = idx[None, None, :] <= idx[None, :, None]
    mask = jnp.broadcast_to(mask, (batch_size, seq_len, seq_len))
    if padding_mask is not None:
        mask = mask & padding_mask[:, None, :].astype(jnp.bool_)
    return mask


# ---------------------------------------------------------------------------
# Gated DeltaNet
# ---------------------------------------------------------------------------


def l2norm(x: jaxtyping.Array, eps: float = 1e-6) -> jaxtyping.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_conv1d(
    x: jaxtyping.Array,  # [B, C, L]
    weight: jaxtyping.Array,  # [C, K]
    activation: bool = True,
) -> jaxtyping.Array:
    """Depthwise causal conv1d, left-padded by ``K - 1``."""
    kernel = weight.shape[-1]
    padded = jnp.pad(x, ((0, 0), (0, 0), (kernel - 1, 0)))
    out = jax.lax.conv_general_dilated(
        padded,
        weight[:, None, :],  # [C, 1, K]
        window_strides=(1,),
        padding="VALID",
        feature_group_count=x.shape[1],
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return jax.nn.silu(out) if activation else out


def chunk_gated_delta_rule(
    query: jaxtyping.Array,  # [B, L, Hv, Dk]
    key: jaxtyping.Array,  # [B, L, Hv, Dk]
    value: jaxtyping.Array,  # [B, L, Hv, Dv]
    g: jaxtyping.Array,  # [B, L, Hv] -- log decay, <= 0
    beta: jaxtyping.Array,  # [B, L, Hv]
    *,
    chunk_size: int = 64,
    initial_state: jaxtyping.Array | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
    """Chunkwise gated delta rule -- the prefill/training path.

    Port of HF ``torch_chunk_gated_delta_rule``.  The sequential dependency runs
    over *chunks* (L / chunk_size steps) rather than tokens, and is expressed as
    a single ``lax.scan``, so the emitted jaxpr has a fixed size regardless of
    sequence length -- nothing is unrolled per token.

    Returns ``(output [B, L, Hv, Dv], final_state [B, Hv, Dk, Dv])``.
    """
    in_dtype = query.dtype
    batch, seq_len, _, k_dim = key.shape
    num_v_heads, v_dim = value.shape[-2:]

    # -> [B, H, L, D] in float32; the recurrence is numerically delicate.
    q, k, v, beta_t, decay = (
        jnp.swapaxes(t, 1, 2).astype(jnp.float32) for t in (query, key, value, beta, g)
    )
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    q = q * (k_dim**-0.5)

    pad = (-seq_len) % chunk_size
    if pad:
        q, k, v = (jnp.pad(t, ((0, 0), (0, 0), (0, pad), (0, 0))) for t in (q, k, v))
        beta_t, decay = (
            jnp.pad(t, ((0, 0), (0, 0), (0, pad))) for t in (beta_t, decay)
        )
    num_chunks = (seq_len + pad) // chunk_size

    v_beta, k_beta = v * beta_t[..., None], k * beta_t[..., None]

    def to_chunks(t):
        return t.reshape(batch, num_v_heads, num_chunks, chunk_size, t.shape[-1])

    q, k, k_beta, v_beta = (to_chunks(t) for t in (q, k, k_beta, v_beta))
    decay = decay.reshape(batch, num_v_heads, num_chunks, chunk_size)

    strictly_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), jnp.bool_), 1)
    cum_decay = jnp.cumsum(decay, axis=3)
    # pairwise[..., i, j] = decay accumulated from j to i; masked above the
    # diagonal before exp so it cannot overflow.
    pairwise = cum_decay[..., :, None] - cum_decay[..., None, :]
    pairwise = jnp.exp(jnp.where(strictly_upper, -jnp.inf, pairwise))

    ut_system = (k_beta @ jnp.swapaxes(k, -1, -2)) * pairwise
    intra_chunk_attn = (q @ jnp.swapaxes(k, -1, -2)) * pairwise
    decayed_k_beta = k_beta * jnp.exp(cum_decay)[..., None]

    # The UT transform: solve the unit lower-triangular system (the diagonal of
    # ut_system is ignored, as in HF's solve_triangular(unitriangular=True)).
    #
    # Do NOT substitute the "matmul inverse" L^-1 = prod_j (I + M^(2^j)):
    # although algebraically exact for nilpotent M, the intermediate powers
    # M^(2^j) of a 64x64 strictly-lower-triangular matrix with O(1) entries grow
    # combinatorially and destroy the result in float32.  It looks fine at small
    # chunk sizes, which makes it a trap.
    solve = functools.partial(
        jax.lax.linalg.triangular_solve,
        left_side=True,
        lower=True,
        unit_diagonal=True,
        transpose_a=False,
    )
    new_values = solve(ut_system, v_beta)
    k_cumdecay = solve(ut_system, decayed_k_beta)

    q = q * jnp.exp(cum_decay)[..., None]
    k = k * jnp.exp(cum_decay[..., -1:] - cum_decay)[..., None]
    chunk_decay = jnp.exp(cum_decay[..., -1])[..., None, None]

    state0 = (
        jnp.zeros((batch, num_v_heads, k_dim, v_dim), jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )

    def step(state, xs):
        nv, kcd, q_i, intra_i, k_i, decay_i = xs
        v_new = nv - kcd @ state
        out = q_i @ state + intra_i @ v_new
        state = state * decay_i + jnp.swapaxes(k_i, -1, -2) @ v_new
        return state, out

    # Move the chunk axis to the front for scan; everything else stays batched.
    def to_scan(t):
        return jnp.moveaxis(t, 2, 0)

    final_state, out = jax.lax.scan(
        step,
        state0,
        tuple(
            to_scan(t)
            for t in (new_values, k_cumdecay, q, intra_chunk_attn, k, chunk_decay)
        ),
    )
    out = jnp.moveaxis(out, 0, 2).reshape(batch, num_v_heads, -1, v_dim)[:, :, :seq_len]
    return jnp.swapaxes(out, 1, 2).astype(in_dtype), final_state


def recurrent_gated_delta_rule(
    query: jaxtyping.Array,  # [B, L, Hv, Dk]
    key: jaxtyping.Array,
    value: jaxtyping.Array,  # [B, L, Hv, Dv]
    g: jaxtyping.Array,
    beta: jaxtyping.Array,
    *,
    initial_state: jaxtyping.Array | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
    """Token-by-token gated delta rule -- the cached-decode path.

    Port of HF ``torch_recurrent_gated_delta_rule``.  A ``lax.scan`` over the
    sequence axis, so again nothing unrolls; for single-token decode the scan
    has length 1.
    """
    in_dtype = query.dtype
    batch, seq_len = key.shape[0], key.shape[1]
    num_v_heads, v_dim = value.shape[-2:]
    k_dim = key.shape[-1]

    q, k, v, beta_t, decay = (
        jnp.swapaxes(t, 1, 2).astype(jnp.float32) for t in (query, key, value, beta, g)
    )
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    q = q / (k_dim**0.5)

    state0 = (
        jnp.zeros((batch, num_v_heads, k_dim, v_dim), jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )

    def step(state, xs):
        q_t, k_t, v_t, g_t, b_t = xs
        state = state * jnp.exp(g_t)[..., None, None]
        kv_mem = jnp.sum(state * k_t[..., None], axis=-2)
        delta = (v_t - kv_mem) * b_t[..., None]
        state = state + k_t[..., None] * delta[..., None, :]
        return state, jnp.sum(state * q_t[..., None], axis=-2)

    def mv(t):
        return jnp.moveaxis(t, 2, 0)

    final_state, out = jax.lax.scan(
        step,
        state0,
        (mv(q), mv(k), mv(v), mv(decay), mv(beta_t)),
    )
    out = jnp.moveaxis(out, 0, 2)  # [B, Hv, L, Dv]
    del seq_len
    return jnp.swapaxes(out, 1, 2).astype(in_dtype), final_state


class GatedDeltaNet(nnx.Module):
    """Linear-attention token mixer: short conv + gated delta rule + gated norm."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig,
    ):
        self.config = config
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.shd_config = shd_config

        dt, pdt = config.param_dtype, config.param_dtype
        lin = functools.partial(
            nnx.Linear, use_bias=False, dtype=dt, param_dtype=pdt, rngs=rngs
        )
        self.in_proj_qkv = lin(config.embed_dim, self.conv_dim)
        self.in_proj_z = lin(config.embed_dim, self.value_dim)
        self.in_proj_b = lin(config.embed_dim, self.num_v_heads)
        self.in_proj_a = lin(config.embed_dim, self.num_v_heads)
        self.out_proj = lin(self.value_dim, config.embed_dim)

        # Depthwise conv kernel, stored as [conv_dim, K] (torch keeps a
        # singleton in-channel axis: [conv_dim, 1, K]).
        self.conv1d = nnx.Param(
            nnx.initializers.normal(dtype=pdt)(
                rngs.params(), (self.conv_dim, self.conv_kernel_size)
            ),
            sharding=shd_config.gdn_conv_weight,
        )
        # Kept in float32: A_log is exponentiated and dt_bias goes through
        # softplus, both of which lose too much in bfloat16.
        self.dt_bias = nnx.Param(
            nnx.initializers.ones_init()(rngs.params(), self.num_v_heads).astype(
                jnp.float32
            ),
            sharding=shd_config.gdn_head_scalar,
        )
        self.A_log = nnx.Param(
            nnx.initializers.zeros_init()(rngs.params(), self.num_v_heads).astype(
                jnp.float32
            ),
            sharding=shd_config.gdn_head_scalar,
        )
        self.norm = RMSNormGated(
            self.head_v_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
            param_dtype=pdt,
        )

    @jax.named_scope("gated_delta_net")
    def __call__(
        self,
        x: jaxtyping.Array,  # [B, L, D]
        padding_mask: jaxtyping.Array | None = None,
    ) -> jaxtyping.Array:
        batch, seq_len, _ = x.shape

        if padding_mask is not None:
            # Zero padded positions before the conv so they cannot leak into
            # the recurrence (cf. HF apply_mask_to_padding_states).
            x = x * padding_mask[..., None].astype(x.dtype)

        mixed_qkv = self.in_proj_qkv(x)  # [B, L, conv_dim]
        mixed_qkv = causal_conv1d(
            jnp.swapaxes(mixed_qkv, 1, 2), self.conv1d.value, activation=True
        )
        mixed_qkv = jnp.swapaxes(mixed_qkv, 1, 2)[:, :seq_len]

        q, k, v = jnp.split(mixed_qkv, [self.key_dim, self.key_dim * 2], axis=-1)
        q = q.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        z = self.in_proj_z(x).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        beta = jax.nn.sigmoid(self.in_proj_b(x).astype(jnp.float32))
        a = self.in_proj_a(x).astype(jnp.float32)
        g = -jnp.exp(self.A_log.value.astype(jnp.float32)) * jax.nn.softplus(
            a + self.dt_bias.value.astype(jnp.float32)
        )

        repeats = self.num_v_heads // self.num_k_heads
        if repeats > 1:
            q = jnp.repeat(q, repeats, axis=2)
            k = jnp.repeat(k, repeats, axis=2)

        core_out, _ = chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=self.config.gdn_chunk_size, use_qk_l2norm=True
        )

        core_out = self.norm(core_out, z).reshape(batch, seq_len, self.value_dim)
        return self.out_proj(core_out)


class Attention(nnx.Module):
    """Gated GQA attention with per-head q/k norms and partial RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig,
    ):
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.shd_config = shd_config
        pdt = config.param_dtype

        # q_proj emits the query *and* its output gate, hence head_dim * 2.
        q_out = self.head_dim * (2 if config.attn_output_gate else 1)
        self.q_proj = Einsum(
            einsum_str="BTD,DNH->BTNH",
            shape=(config.embed_dim, self.num_heads, q_out),
            rngs=rngs,
            sharding=shd_config.q_weight_dnh,
            param_dtype=pdt,
        )
        self.k_proj = Einsum(
            einsum_str="BSD,DKH->BSKH",
            shape=(config.embed_dim, self.num_kv_heads, self.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_dnh,
            param_dtype=pdt,
        )
        self.v_proj = Einsum(
            einsum_str="BSD,DKH->BSKH",
            shape=(config.embed_dim, self.num_kv_heads, self.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_dnh,
            param_dtype=pdt,
        )
        self.o_proj = Einsum(
            einsum_str="BTNH,NHD->BTD",
            shape=(self.num_heads, self.head_dim, config.embed_dim),
            rngs=rngs,
            sharding=shd_config.o_weight_nhd,
            param_dtype=pdt,
        )
        self.q_norm = RMSNorm(
            self.head_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
            param_dtype=pdt,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
            param_dtype=pdt,
        )

    @jax.named_scope("attention")
    def __call__(
        self,
        x: jaxtyping.Array,  # [B, L, D]
        cos: jaxtyping.Array,
        sin: jaxtyping.Array,
        attn_mask: jaxtyping.Array | None,  # [B, L, S] bool
    ) -> jaxtyping.Array:
        qg = self.q_proj(x)  # [B, L, N, H*2] (or H)
        if self.config.attn_output_gate:
            q, gate = jnp.split(qg, 2, axis=-1)
            gate = gate.reshape(*x.shape[:-1], -1)
        else:
            q, gate = qg, None

        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = shard(q, self.shd_config.act_btnh)
        k = shard(k, self.shd_config.act_btnh)
        v = shard(v, self.shd_config.act_btnh)

        mask = jnp.expand_dims(attn_mask, -3) if attn_mask is not None else None
        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            scale=self.scale,
            **attention_impl_kwargs(q.dtype, self.head_dim),
        )
        if gate is not None:
            out = out.reshape(*x.shape[:-1], -1) * jax.nn.sigmoid(gate)
            out = out.reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        return self.o_proj(out)


class MLP(nnx.Module):
    """SwiGLU feed-forward block."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig,
    ):
        dt = config.param_dtype
        lin = functools.partial(
            nnx.Linear, use_bias=False, dtype=dt, param_dtype=dt, rngs=rngs
        )
        self.gate_proj = lin(config.embed_dim, config.hidden_dim)
        self.up_proj = lin(config.embed_dim, config.hidden_dim)
        self.down_proj = lin(config.hidden_dim, config.embed_dim)
        self.shd_config = shd_config

    @jax.named_scope("mlp")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        h = jax.nn.silu(self.gate_proj(x)) * self.up_proj(x)
        h = shard(h, self.shd_config.act_btf)
        return self.down_proj(h)


class DecoderLayer(nnx.Module):
    """One hybrid block: either GDN or gated attention, then an MLP."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig,
    ):
        self.block_type = config.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
            param_dtype=config.param_dtype,
        )
        if self.block_type == LINEAR_ATTENTION:
            self.linear_attn = GatedDeltaNet(config, rngs=rngs, shd_config=shd_config)
        elif self.block_type == FULL_ATTENTION:
            self.attn = Attention(config, rngs=rngs, shd_config=shd_config)
        else:
            raise ValueError(f"unknown layer type {self.block_type!r}")
        self.post_attention_layernorm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
            param_dtype=config.param_dtype,
        )
        self.mlp = MLP(config, rngs=rngs, shd_config=shd_config)

    def __call__(
        self,
        x: jaxtyping.Array,
        cos: jaxtyping.Array,
        sin: jaxtyping.Array,
        attn_mask: jaxtyping.Array | None,
        padding_mask: jaxtyping.Array | None,
    ) -> jaxtyping.Array:
        residual = x
        h = self.input_layernorm(x)
        if self.block_type == LINEAR_ATTENTION:
            h = self.linear_attn(h, padding_mask)
        else:
            h = self.attn(h, cos, sin, attn_mask)
        x = residual + h
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3_5(nnx.Module):
    """Dense Qwen3.5 text model (equivalent of HF ``Qwen3_5ForCausalLM``)."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig | None = None,
    ):
        self.config = config
        shd_config = shd_config or config.shd_config
        self.embedder = Embedder(
            config.vocab_size,
            config.embed_dim,
            rngs=rngs,
            shd_config=shd_config,
            param_dtype=config.param_dtype,
        )
        self.layers = compat.ModuleList(
            [
                DecoderLayer(config, i, rngs=rngs, shd_config=shd_config)
                for i in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
            param_dtype=config.param_dtype,
        )
        self.lm_head = (
            None
            if config.use_tied_embedding
            else Einsum(
                einsum_str="BTD,DV->BTV",
                shape=(config.embed_dim, config.vocab_size),
                rngs=rngs,
                sharding=shd_config.emb_dv,
                param_dtype=config.param_dtype,
            )
        )

    def __call__(
        self,
        input_tokens: jaxtyping.Array,  # [B, L]
        positions: jaxtyping.Array | None = None,  # [4, B, L] or [B, L] or None
        padding_mask: jaxtyping.Array | None = None,  # [B, L]
        cache: Cache | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[jaxtyping.Array, list[jaxtyping.Array] | None]:
        """Returns ``(logits, hidden_states_or_None)``.

        ``positions`` follows HF's 4-row layout: row 0 is the text position used
        for masking, rows 1..3 are the (T, H, W) axes fed to M-RoPE.  For
        text-only input all four rows are identical, and passing ``None`` builds
        them from a plain arange.
        """
        if cache is not None:
            raise NotImplementedError(
                "cached decode is not implemented yet; call with cache=None"
            )
        batch, seq_len = input_tokens.shape
        if positions is None:
            idx = jnp.arange(seq_len)[None, :]
            positions = jnp.broadcast_to(idx, (4, batch, seq_len))
        elif positions.ndim == 2:
            positions = jnp.broadcast_to(positions[None], (4, *positions.shape))
        rope_positions = positions[1:]  # (T, H, W)

        cos, sin = rope_cos_sin(
            rope_positions,
            rotary_dim=self.config.rotary_dim,
            rope_theta=self.config.rope_theta,
            mrope_section=self.config.mrope_section,
        )
        attn_mask = make_causal_mask(seq_len, batch, padding_mask)

        x = self.embedder.encode(input_tokens)
        hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            if self.config.remat_config == RematConfig.BLOCK:
                x = nnx.remat(lambda m, *a: m(*a))(
                    layer, x, cos, sin, attn_mask, padding_mask
                )
            else:
                x = layer(x, cos, sin, attn_mask, padding_mask)
            if output_hidden_states:
                hidden_states.append(x)
        x = self.final_norm(x)
        logits = self.embedder.decode(x) if self.lm_head is None else self.lm_head(x)
        return logits, hidden_states

    def get_model_input(self):
        """Dummy inputs, for LoRA/graph initialisation (cf. qwen3vl)."""
        return {
            "input_tokens": jnp.zeros((1, 8), dtype=jnp.int32),
            "positions": None,
            "padding_mask": jnp.ones((1, 8), dtype=jnp.bool_),
            "cache": None,
        }

    def init_cache(self, *args, **kwargs):
        raise NotImplementedError("cached decode is not implemented yet")
