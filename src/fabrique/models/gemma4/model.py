"""Gemma 4 multimodal model.

Extends the Tunix Gemma4 text model with a vision tower and projector so that
pixel_values / pixel_position_ids can be passed alongside text tokens.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
from functools import partial
from typing import Optional, Tuple

import flax
import jax
import jax.sharding as shd
import jaxtyping
from flax import nnx
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma4 import moe
from tunix.models.gemma4.model import GEMMA4_ATTENTION_PATTERN, AttentionType
from tunix.utils import compat, env_utils
from tunix.utils.sharding_utils import shard

from fabrique.models.gemma4 import vision as vision_lib

env_utils.setup_sharding_environment()


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
    NONE = enum.auto()
    BLOCK = enum.auto()
    DECODER = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for gemma transformer."""

    emb_vd: Tuple[str | None, ...]
    q_weight_ndh: Tuple[str | None, ...]
    kv_weight_cndh: Tuple[str | None, ...]
    qkv_weight_cndh: Tuple[str | None, ...]
    o_weight_nhd: Tuple[str | None, ...]
    ffw_weight_df: Tuple[str | None, ...]
    ffw_weight_fd: Tuple[str | None, ...]
    rms_norm_weight: Tuple[str | None, ...]
    act_btd: Tuple[str | None, ...]
    act_btf: Tuple[str | None, ...]
    act_btnh: Tuple[str | None, ...]
    vision_proj: Tuple[str | None, ...]
    vision_soft_emb_norm_weight: Tuple[str | None, ...]
    # MoE sharding
    exp_weight_edf: Tuple[str | None, ...]
    exp_weight_efd: Tuple[str | None, ...]
    # PLE sharding
    per_layer_model_projection: Tuple[str | None, ...]
    per_layer_input_gate: Tuple[str | None, ...]
    per_layer_projection: Tuple[str | None, ...]
    per_layer_input_embedding: Tuple[str | None, ...]

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None

        return ShardingConfig(
            emb_vd=("tp", fsdp),
            q_weight_ndh=("tp", fsdp, None),
            kv_weight_cndh=(None, "tp", fsdp, None),
            qkv_weight_cndh=(None, "tp", fsdp, None),
            o_weight_nhd=("tp", None, fsdp),
            ffw_weight_df=(fsdp, "tp"),
            ffw_weight_fd=("tp", fsdp),
            rms_norm_weight=("tp",),
            act_btd=("fsdp", None, None if is_sampling else "tp"),
            act_btf=("fsdp", None, "tp"),
            act_btnh=("fsdp", None, "tp", None),
            vision_proj=(fsdp, "tp"),
            vision_soft_emb_norm_weight=("tp",),
            exp_weight_edf=(fsdp, None, None, "tp"),
            exp_weight_efd=(fsdp, "tp", None),
            per_layer_model_projection=(fsdp, None, "tp"),
            per_layer_input_gate=(fsdp, "tp"),
            per_layer_projection=("tp", fsdp),
            per_layer_input_embedding=("tp", None, fsdp),
        )


@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
    """Transformer config."""

    num_layers: int
    num_embed: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    final_logit_softcap: float = 30.0
    sliding_window_size: int | None = None
    per_layer_input_dim: int = 0
    num_global_kv_heads: int | None = None
    global_key_size: int = 512
    attention_pattern: tuple["AttentionType", ...] | None = None
    frac_shared_layers: float = 0.0
    global_rope_proportion: float = 0.25
    local_rope_proportion: float = 1.0
    k_eq_v_global: bool = False
    override_kv_shared_ffw_hidden: int | None = None

    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000
    local_scale_factor: float = 1.0
    global_scale_factor: float = 1.0

    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
    remat_config: RematConfig = RematConfig.NONE
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32
    use_flash_attention: bool = False
    flash_attention_block_size: int = 1024

    # MoE config
    enable_moe: bool = False
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    expert_dim: int | None = None
    moe_dense_hidden_dim: int | None = None

    # Vision config (None = text-only)
    vision_config: Optional[vision_lib.VisionConfig] = None

    @classmethod
    def gemma4_e4b(
        cls,
        sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ) -> "ModelConfig":
        return cls(
            num_layers=42,
            num_embed=262144,
            embed_dim=2560,
            hidden_dim=2560 * 4,
            num_heads=8,
            head_dim=256,
            num_kv_heads=2,
            sliding_window_size=512,
            shd_config=sharding_config,
            per_layer_input_dim=256,
            frac_shared_layers=18.0 / 42,
            attention_pattern=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ),
        )

    @classmethod
    def gemma4_e4b_it(
        cls,
        sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ) -> "ModelConfig":
        """Gemma 4 E4B multimodal instruct config."""
        return cls(
            num_layers=42,
            num_embed=262144,
            embed_dim=2560,
            hidden_dim=2560 * 4,
            num_heads=8,
            head_dim=256,
            num_kv_heads=2,
            sliding_window_size=512,
            shd_config=sharding_config,
            per_layer_input_dim=256,
            frac_shared_layers=18.0 / 42,
            attention_pattern=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ),
            vision_config=vision_lib.VisionConfig(),
        )

    @classmethod
    def gemma4_e2b(
        cls,
        sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ) -> "ModelConfig":
        return cls(
            num_layers=35,
            num_embed=262144,
            embed_dim=1536,
            hidden_dim=1536 * 4,
            num_heads=8,
            head_dim=256,
            num_kv_heads=1,
            sliding_window_size=512,
            shd_config=sharding_config,
            per_layer_input_dim=256,
            frac_shared_layers=20.0 / 35,
            override_kv_shared_ffw_hidden=int(1536 * 4 * 2),
            attention_pattern=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ),
        )

    @classmethod
    def gemma4_31b(
        cls,
        sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ) -> "ModelConfig":
        return cls(
            num_layers=60,
            num_embed=262144,
            embed_dim=5376,
            hidden_dim=5376 * 4,
            num_heads=32,
            head_dim=256,
            num_kv_heads=16,
            num_global_kv_heads=4,
            sliding_window_size=1024,
            shd_config=sharding_config,
            k_eq_v_global=True,
            attention_pattern=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ),
        )

    @classmethod
    def gemma4_26b_a4b(
        cls,
        sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ) -> "ModelConfig":
        return cls(
            num_layers=30,
            num_embed=262144,
            embed_dim=2816,
            hidden_dim=2112,
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            num_global_kv_heads=2,
            sliding_window_size=1024,
            shd_config=sharding_config,
            enable_moe=True,
            num_experts=128,
            expert_dim=704,
            num_experts_per_tok=8,
            moe_dense_hidden_dim=2112,
            k_eq_v_global=True,
            global_rope_proportion=0.25,
            attention_pattern=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ),
        )


class Embedder(nnx.Module):
    """Embedder module."""

    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.vocab_size = config.num_embed
        self.embed_dim = config.embed_dim
        self.param_dtype = config.param_dtype

        # When True, the two large embedding tables live on CPU and lookups are
        # routed through explicit CPU→accelerator device transfers.  Set by the
        # params loader when cpu_embed=True (e.g. for the E4B model).
        self._cpu_offload: bool = False

        self.input_embedding = nnx.Param(
            nnx.initializers.normal(dtype=self.param_dtype)(
                rngs.params(), (self.vocab_size, self.embed_dim)
            ),
        )

        if config.per_layer_input_dim > 0:
            self.per_layer_model_projection = Einsum(
                einsum_str="BTD,DNP->BTNP",
                shape=(self.embed_dim, config.num_layers, config.per_layer_input_dim),
                sharding=config.shd_config.per_layer_model_projection,
                w_scale=(float(self.embed_dim) ** -0.5),
                rngs=rngs,
                dtype=self.config.dtype,
                param_dtype=self.param_dtype,
            )

            self.per_layer_projection_norm = RMSNorm(
                config.per_layer_input_dim,
                rngs=rngs,
                sharding=config.shd_config,
                dtype=self.config.dtype,
                param_dtype=self.param_dtype,
            )

            self.per_layer_input_embedding = nnx.Param(
                nnx.initializers.normal(dtype=self.param_dtype)(
                    rngs.params(),
                    (self.vocab_size, config.num_layers, config.per_layer_input_dim),
                ),
                sharding=config.shd_config.per_layer_input_embedding,
            )

    def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        if self._cpu_offload:
            # Embedding table is on CPU: move token IDs there, gather, bring back.
            cpu = jax.devices("cpu")[0]
            gpu = jax.devices()[0]
            x = jax.device_put(self.input_embedding.value[jax.device_put(x, cpu)], gpu)
        else:
            x = self.input_embedding[(x,)]
        x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
        x = jnp.astype(x, self.config.dtype)
        x = shard(x, self.config.shd_config.act_btd)
        return x

    def encode_per_layer_input(
        self, x: jaxtyping.ArrayLike, t: jaxtyping.ArrayLike
    ) -> jaxtyping.Array:
        t = jnp.where(
            jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)
        )
        x = self.per_layer_model_projection(x)
        x = self.per_layer_projection_norm(x)
        if self._cpu_offload:
            cpu = jax.devices("cpu")[0]
            gpu = jax.devices()[0]
            y = jax.device_put(
                self.per_layer_input_embedding.value[jax.device_put(t, cpu)], gpu
            )
        else:
            y = self.per_layer_input_embedding.value[t]
        y *= jnp.sqrt(self.config.per_layer_input_dim).astype(y.dtype)
        return (x + y) * jax.lax.rsqrt(2.0).astype(x.dtype)

    def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        if self._cpu_offload:
            # Embedding table is on CPU: move hidden state there, matmul, bring back.
            cpu = jax.devices("cpu")[0]
            gpu = jax.devices()[0]
            x_cpu = jax.device_put(jnp.astype(x, self.config.dtype), cpu)
            w_cpu = jnp.astype(self.input_embedding.value, self.config.dtype)
            return jax.device_put(jnp.dot(x_cpu, w_cpu.T), gpu)
        x = jnp.astype(x, self.config.dtype)
        w = jnp.astype(self.input_embedding.value, self.config.dtype)
        return jnp.dot(x, w.T)


class Einsum(nnx.Module):
    """Einsum module."""

    def __init__(
        self,
        einsum_str: str,
        shape: flax.typing.Shape,
        *,
        rngs: nnx.Rngs,
        sharding: Tuple[str | None, ...],
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        w_scale: float | None = None,
    ):
        self.einsum_str = einsum_str
        self.dtype = dtype
        self.w_scale = w_scale

        self.shape = shape
        self.w = nnx.Param(
            nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
            sharding=sharding,
        )

    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        w = self.w.value
        if self.w_scale is not None:
            w = w * self.w_scale
        x = jnp.astype(x, self.dtype)
        w = jnp.astype(w, self.dtype)
        return jnp.einsum(self.einsum_str, x, w)


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
    """Finds the index of the last (rightmost) '1' from attn_mask."""
    cache_len = attn_mask.shape[-1]
    all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)
    reversed_matrix = attn_mask[:, :, ::-1]
    first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)
    last_one_index_original = cache_len - 1 - first_one_from_right
    return jnp.where(all_zeros_mask, 0, last_one_index_original).squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jnp.ndarray,
    sliding_window_size: int,
) -> jnp.ndarray:
    """Helper function to create sliding window mask for local attention."""
    upper_index = find_last_one_index(attn_mask)
    window_start_pos = upper_index - sliding_window_size + 1
    abs_pos = jnp.arange(attn_mask.shape[-1])
    window_mask = abs_pos[None, :] >= window_start_pos[:, None]
    causal_mask = abs_pos[None, :] <= upper_index[:, None]
    return (window_mask & causal_mask)[:, None, :]


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(
        self,
        dim: int,
        *,
        rngs: nnx.Rngs,
        sharding: ShardingConfig = ShardingConfig.get_default_sharding(),
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
    ):
        self.scale = nnx.Param(
            nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),
            sharding=sharding.rms_norm_weight,
        )
        self.dtype = dtype

    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        x = jnp.astype(x, jnp.float32)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06).astype(x.dtype)
        scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * scale
        return normed_inputs.astype(self.dtype)


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jax.Array:
    """Applies RoPE to inputs of shape [B, L, N, H]."""
    head_dim = inputs.shape[-1]
    rope_angles = int(rope_proportion * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    freq_exponents = (2.0 / head_dim) * jnp.arange(0, rope_angles, dtype=jnp.float32)
    timescale = jnp.pad(
        base_frequency**freq_exponents,
        (0, nope_angles),
        mode="constant",
        constant_values=(0, jnp.inf),
    )
    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return jnp.concatenate([first_part, second_part], axis=-1).astype(inputs.dtype)


K_MASK = -2.3819763e38


def create_kv_cache_sharing_patterns(
    num_layers: int,
    frac_shared_layers: float,
    share_global: bool,
    share_local: bool,
    attention_types: tuple[AttentionType, ...],
) -> list[int]:
    kv_cache_sharing_patterns = []
    num_unshared_layers = int(num_layers - frac_shared_layers * num_layers)
    for i in range(num_layers):
        if i < num_unshared_layers:
            kv_cache_sharing_patterns.append(i)
        else:
            if attention_types[i] == AttentionType.GLOBAL and share_global:
                kv_cache_sharing_patterns.append(num_unshared_layers - 1)
            elif attention_types[i] == AttentionType.LOCAL_SLIDING and share_local:
                kv_cache_sharing_patterns.append(num_unshared_layers - 2)
            else:
                kv_cache_sharing_patterns.append(i)
    return kv_cache_sharing_patterns


class Attention(nnx.Module):
    """Attention module."""

    def __init__(
        self,
        config: ModelConfig,
        attn_type: AttentionType,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.rope_proportion = (
            config.global_rope_proportion
            if attn_type == AttentionType.GLOBAL
            else config.local_rope_proportion
        )
        self.attn_type = attn_type
        self.rope_base_frequency = (
            config.local_base_frequency
            if attn_type == AttentionType.LOCAL_SLIDING
            else config.global_base_frequency
        )
        self.rope_scale_factor = (
            config.local_scale_factor
            if attn_type == AttentionType.LOCAL_SLIDING
            else config.global_scale_factor
        )

        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        if attn_type == AttentionType.GLOBAL:
            if config.num_global_kv_heads is not None:
                self.num_kv_heads = config.num_global_kv_heads
            if config.global_key_size is not None:
                self.head_dim = config.global_key_size

        self.attn_vec_einsum = Einsum(
            einsum_str="BTNH,NHD->BTD",
            shape=(config.num_heads, self.head_dim, config.embed_dim),
            rngs=rngs,
            sharding=config.shd_config.o_weight_nhd,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self.q_einsum = Einsum(
            einsum_str="BTD,NDH->BTNH",
            shape=(config.num_heads, config.embed_dim, self.head_dim),
            rngs=rngs,
            sharding=config.shd_config.q_weight_ndh,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

        k_eq_v = config.k_eq_v_global if attn_type == AttentionType.GLOBAL else False
        if k_eq_v:
            self.k_einsum = Einsum(
                einsum_str="BSD,KDH->BSKH",
                shape=(self.num_kv_heads, config.embed_dim, self.head_dim),
                rngs=rngs,
                sharding=config.shd_config.q_weight_ndh,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
        else:
            self.kv_einsum = Einsum(
                einsum_str="BSD,CKDH->CBSKH",
                shape=(2, self.num_kv_heads, config.embed_dim, self.head_dim),
                rngs=rngs,
                sharding=(
                    (None, None, "fsdp", None)
                    if self.num_kv_heads == 1
                    else config.shd_config.kv_weight_cndh
                ),
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
        self._query_norm = RMSNorm(
            self.head_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self._key_norm = RMSNorm(
            self.head_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

    def block(
        self,
        x: jaxtyping.Array,
        segment_pos: jaxtyping.Array,
        cache: LayerCache | None,
        attn_mask: jaxtyping.Array,
        kv_shared_cache: LayerCache | None = None,
    ) -> tuple[LayerCache | None, jaxtyping.Array]:
        x = x.astype(self.config.dtype)
        seq_len = x.shape[1]
        query_proj = self.q_einsum(x)
        query_proj = shard(query_proj, self.config.shd_config.act_btnh)
        query_proj = self._query_norm(query_proj)
        query_proj = apply_rope(
            query_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
            rope_proportion=self.rope_proportion,
        )

        if kv_shared_cache is not None:
            key_proj = kv_shared_cache["k"]
            value_proj = kv_shared_cache["v"]
        else:
            if hasattr(self, "k_einsum"):
                key_proj = self.k_einsum(x)
                value_proj = key_proj
            else:
                key_proj, value_proj = self.kv_einsum(x)

            key_proj = shard(key_proj, self.config.shd_config.act_btnh)
            value_proj = shard(value_proj, self.config.shd_config.act_btnh)

            value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
            value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
            key_proj = self._key_norm(key_proj)
            key_proj = apply_rope(
                key_proj,
                segment_pos,
                base_frequency=self.rope_base_frequency,
                scale_factor=self.rope_scale_factor,
                rope_proportion=self.rope_proportion,
            )

        if cache is not None:
            end_index = cache["end_index"][0]
            slice_indices = (0, end_index % cache["v"].shape[1], 0, 0)
            value_proj = jax.lax.dynamic_update_slice(
                cache["v"], value_proj, slice_indices
            )
            key_proj = jax.lax.dynamic_update_slice(cache["k"], key_proj, slice_indices)

        b, t, qh, d = query_proj.shape
        _, _, kh, _ = key_proj.shape

        if self.config.use_flash_attention and seq_len > 1:
            query_proj = query_proj.transpose(0, 2, 1, 3)
            key_proj = key_proj.transpose(0, 2, 1, 3)
            value_proj = value_proj.transpose(0, 2, 1, 3)

            mesh = pxla.thread_resources.env.physical_mesh
            if self.attn_type == AttentionType.LOCAL_SLIDING:
                mask = mask_lib.LocalMask(
                    (seq_len, seq_len),
                    window_size=(self.config.sliding_window_size - 1, 0),
                    offset=0,
                )
            else:
                mask = mask_lib.CausalMask((seq_len, seq_len))

            multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

            block_sizes = splash.BlockSizes(
                block_q=self.config.flash_attention_block_size,
                block_kv=self.config.flash_attention_block_size,
                block_q_dkv=self.config.flash_attention_block_size,
                block_kv_dkv=self.config.flash_attention_block_size,
                block_kv_dkv_compute=self.config.flash_attention_block_size,
                block_q_dq=self.config.flash_attention_block_size,
                block_kv_dq=self.config.flash_attention_block_size,
            )

            shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
            head_shards = (
                mesh.shape[shd_n] if shd_n is not None and shd_n in mesh.shape else 1
            )
            q_seq_shards = (
                mesh.shape[shd_t] if shd_t is not None and shd_t in mesh.shape else 1
            )

            splash_attn_kernel = splash.make_splash_mha(
                multi_head_mask,
                block_sizes=block_sizes,
                head_shards=head_shards,
                q_seq_shards=q_seq_shards,
            )

            shd_spec = P(shd_b, shd_n, shd_t, shd_h)
            unsharded_seq = P(shd_b, shd_n, None, shd_h)
            kernel_spec = splash_attn_kernel.manual_sharding_spec(
                shd.NamedSharding(mesh, P(shd_n, shd_t))
            )

            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
                out_specs=shd_spec,
                check_rep=False,
            )
            def sharded_splash_attn(kernel, q_block, k_block, v_block):
                return jax.vmap(kernel)(q_block, k_block, v_block)

            qkv = sharded_splash_attn(
                splash_attn_kernel, query_proj, key_proj, value_proj
            )
            encoded = qkv.transpose(0, 2, 1, 3)
        else:
            if self.use_gqa:
                b, t, kg, h = query_proj.shape
                n_groups = kg // self.num_kv_heads
                query_reshaped = query_proj.reshape(
                    (b, t, self.num_kv_heads, n_groups, h)
                )
                logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_reshaped, key_proj)
                b, t, k, g, s = logits.shape
                logits = logits.reshape((b, t, k * g, s))
            else:
                logits = jnp.einsum("BTNH,BSNH->BTNS", query_proj, key_proj)

            if self.attn_type == AttentionType.LOCAL_SLIDING:
                if segment_pos.shape[1] == 1:
                    sliding_mask = create_sliding_window_mask(
                        attn_mask,
                        sliding_window_size=self.config.sliding_window_size,
                    )
                else:
                    all_ones = jnp.ones_like(attn_mask)
                    sliding_mask = jnp.triu(
                        all_ones, -1 * self.config.sliding_window_size + 1
                    ) * jnp.tril(all_ones, self.config.sliding_window_size - 1)
                attn_mask = sliding_mask * attn_mask

            attn = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
            attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
                key_proj.dtype
            )

            if self.use_gqa:
                b, t, kg, s = attn.shape
                n_groups = kg // self.num_kv_heads
                probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
                encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs_reshaped, value_proj)
                b, t, k, g, h = encoded.shape
                encoded = encoded.reshape((b, t, k * g, h))
            else:
                encoded = jnp.einsum("BTNS,BSNH->BTNH", attn, value_proj)

        attn_output = self.attn_vec_einsum(encoded)
        attn_output = shard(attn_output, self.config.shd_config.act_btd)

        if cache is not None:
            new_cache = {
                "v": value_proj,
                "k": key_proj,
                "end_index": cache["end_index"] + seq_len,
            }
        else:
            # Always return K/V so that dedicated layers can supply them to shared
            # layers during the full-sequence (non-cached) forward pass.
            new_cache = {"k": key_proj, "v": value_proj}

        return new_cache, attn_output

    @property
    def use_gqa(self):
        return self.num_kv_heads != self.config.num_heads and self.num_kv_heads > 1

    def __call__(self, x, segment_pos, cache, attn_mask, kv_shared_cache=None):
        return self.block(
            x, segment_pos, cache, attn_mask, kv_shared_cache=kv_shared_cache
        )

    def init_cache(self, batch_size, max_seq_len, dtype):
        return {
            "k": jnp.zeros(
                (batch_size, max_seq_len, self.num_kv_heads, self.head_dim), dtype
            ),
            "v": jnp.zeros(
                (batch_size, max_seq_len, self.num_kv_heads, self.head_dim), dtype
            ),
            "end_index": jnp.zeros((batch_size,), jnp.int32),
        }


class FeedForward(nnx.Module):
    """Feed forward module."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ):
        self.config = config
        h_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
        self.gate_proj = nnx.Linear(
            config.embed_dim,
            h_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), config.shd_config.ffw_weight_df
            ),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self.up_proj = nnx.Linear(
            config.embed_dim,
            h_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), config.shd_config.ffw_weight_df
            ),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self.down_proj = nnx.Linear(
            h_dim,
            config.embed_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), config.shd_config.ffw_weight_fd
            ),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

    def __call__(self, x):
        return self.down_proj(nnx.gelu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nnx.Module):
    """Decoder layer."""

    def __init__(
        self,
        config: ModelConfig,
        attn_type: AttentionType,
        *,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.pre_attention_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self.attn = Attention(config=config, attn_type=attn_type, rngs=rngs)
        self.post_attention_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self.pre_ffw_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )
        self.mlp = FeedForward(config=config, hidden_dim=hidden_dim, rngs=rngs)

        if config.enable_moe:
            self.moe_pre_ffw_norm = RMSNorm(
                config.embed_dim,
                rngs=rngs,
                sharding=config.shd_config,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
            self.moe = moe.MoERagged(config=config, rngs=rngs)
            self.moe_post_ffw_norm = RMSNorm(
                config.embed_dim,
                rngs=rngs,
                sharding=config.shd_config,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
            self.dense_post_ffw_norm = RMSNorm(
                config.embed_dim,
                rngs=rngs,
                sharding=config.shd_config,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
        self.post_ffw_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

        if config.per_layer_input_dim > 0:
            self.per_layer_input_gate = Einsum(
                einsum_str="BTD,DP->BTP",
                shape=(config.embed_dim, config.per_layer_input_dim),
                sharding=config.shd_config.per_layer_input_gate,
                rngs=rngs,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
            self.per_layer_projection = Einsum(
                einsum_str="BTP,PD->BTD",
                shape=(config.per_layer_input_dim, config.embed_dim),
                sharding=config.shd_config.per_layer_projection,
                rngs=rngs,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.embed_dim,
                rngs=rngs,
                sharding=config.shd_config,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
            )

        self.skip_scale = nnx.Param(jnp.ones((1,), dtype=config.param_dtype))

    def __call__(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
        per_layer_input=None,
        kv_shared_cache=None,
    ):
        norm = self.pre_attention_norm(x)
        cache, attn = self.attn(
            norm, segment_pos, cache, attn_mask, kv_shared_cache=kv_shared_cache
        )
        attn = self.post_attention_norm(attn)
        attn += x

        norm_ffw = self.pre_ffw_norm(attn)
        ffw = self.mlp(norm_ffw)
        if self.config.enable_moe:
            ffw = self.dense_post_ffw_norm(ffw)
            moe_norm_ffw = self.moe_pre_ffw_norm(attn)
            moe_out = self.moe(moe_norm_ffw)
            moe_out = self.moe_post_ffw_norm(moe_out)
            ffw += moe_out
        ffw = self.post_ffw_norm(ffw)
        ffw += attn

        if self.config.per_layer_input_dim > 0 and per_layer_input is not None:
            gating_input = ffw
            mapped = self.per_layer_input_gate(gating_input)
            mapped = jax.nn.gelu(mapped) * per_layer_input
            mapped = self.per_layer_projection(mapped)
            mapped = self.post_per_layer_input_norm(mapped)
            ffw += mapped

        ffw = ffw * self.skip_scale.value
        return cache, ffw

    def init_cache(self, batch_size, max_seq_len, dtype):
        return self.attn.init_cache(batch_size, max_seq_len, dtype)


class Gemma4(BackendMappingMixin, nnx.Module):
    """Gemma 4 multimodal model.

    Extends the Tunix text-only Gemma4 with an optional vision tower
    (vision_tower) and projector (embed_vision) for image input.

    If config.vision_config is None this behaves identically to the
    text-only Tunix Gemma4.
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embedder = Embedder(config, rngs=rngs)

        pattern = (
            config.attention_pattern
            if config.attention_pattern
            else GEMMA4_ATTENTION_PATTERN
        )
        attention_types = [
            attn_type
            for _, attn_type in zip(range(config.num_layers), itertools.cycle(pattern))
        ]
        self.kv_cache_sharing_patterns = create_kv_cache_sharing_patterns(
            num_layers=config.num_layers,
            frac_shared_layers=config.frac_shared_layers,
            share_global=True,
            share_local=True,
            attention_types=tuple(attention_types),
        )

        self.layers = compat.ModuleList()
        for i in range(config.num_layers):
            attn_type = attention_types[i]
            h_dim = config.hidden_dim
            if (
                self.kv_cache_sharing_patterns[i] != i
                and config.override_kv_shared_ffw_hidden is not None
            ):
                h_dim = config.override_kv_shared_ffw_hidden
            self.layers.append(
                DecoderLayer(
                    config=config, attn_type=attn_type, hidden_dim=h_dim, rngs=rngs
                )
            )

        self.final_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            sharding=config.shd_config,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

        # Vision components (only instantiated when vision_config is set).
        if config.vision_config is not None:
            vcfg = config.vision_config
            self.vision_tower = vision_lib.VisionModel(
                vcfg,
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
            self.embed_vision = vision_lib.VisionProjector(
                vision_hidden_size=vcfg.hidden_size,
                text_hidden_size=config.embed_dim,
                rms_norm_eps=vcfg.rms_norm_eps,
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )

    @nnx.jit
    def __call__(
        self,
        tokens: jaxtyping.Array,  # [B, L]
        positions: jaxtyping.Array | None = None,  # [B, L]
        pixel_values: jaxtyping.Array | None = None,  # [B, N, C]
        pixel_position_ids: jaxtyping.Array | None = None,  # [B, N, 2]
        cache: Cache | None = None,
        attention_mask: jaxtyping.Array | None = None,  # [B, L, cache_len]
    ) -> tuple[jaxtyping.Array, Cache | None]:
        """Forward pass.

        Args:
          tokens: Input token ids [B, L].
          positions: 1-D positions [B, L]; defaults to arange.
          pixel_values: [B, N, patch_channels] image patches (float32 [0,1]).
            N must equal max_soft_tokens * pooling_kernel_size^2 from the HF
            processor, OR be trimmed to valid_patches_per_image (no padding).
          pixel_position_ids: [B, N, 2] (x, y) patch coords; -1 = padding.
          cache: KV cache dict.
          attention_mask: [B, L, cache_len] bool (True = attend).

        Returns:
          logits [B, L, vocab_size] and updated cache.
        """
        if positions is None:
            B, T = tokens.shape
            positions = jnp.tile(jnp.arange(T)[None, :], (B, 1))

        new_cache = None if cache is None else {}
        # KV store for sharing during full-sequence (non-cached) forward pass.
        # Attention always returns {'k', 'v'} now; dedicated layers save here so
        # that shared layers (kv_cache_sharing_patterns[i] != i) can look them up.
        _kv_share_store: dict[str, dict] = {}
        x = self.embedder.encode(tokens)
        bsz = x.shape[0]

        per_layer_inputs = None
        if self.config.per_layer_input_dim > 0:
            per_layer_inputs = self.embedder.encode_per_layer_input(x, tokens)

        # Inject vision features at image_token_id positions.
        if (
            self.config.vision_config is not None
            and pixel_values is not None
            and pixel_position_ids is not None
        ):
            image_token_id = self.config.vision_config.image_token_id
            k = self.config.vision_config.pooling_kernel_size

            # Encode images: [B, output_length, hidden_size]
            pooled, _ = self.vision_tower(pixel_values, pixel_position_ids)

            # Project to LM embedding space: [B, output_length, embed_dim]
            proj = self.embed_vision(pooled.astype(jnp.bfloat16))
            proj = proj.astype(self.config.dtype)

            # Flatten vision tokens per image: [B, output_length, embed_dim]
            # output_length = pixel_values.shape[1] // k^2 = num_vis_tokens_per_image
            num_vis = proj.shape[1]  # static at JIT-compile time

            def _inject(h, tok, vis):
                """Scatter `vis` [num_vis, D] into `h` [L, D] at image_token_id locs."""
                pos = jnp.where(
                    tok == jnp.int32(image_token_id), size=num_vis, fill_value=-1
                )[0]
                valid = pos >= 0
                pos = jnp.where(valid, pos, 0)
                updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[pos])
                return h.at[pos].set(updates)

            x = jax.vmap(_inject)(x, tokens, proj)

        for i, layer in enumerate(self.layers):
            layer_name = f"layer_{i}"
            layer_cache = cache[layer_name] if cache else None

            shared_idx = self.kv_cache_sharing_patterns[i]
            if shared_idx != i:
                shared_layer_name = f"layer_{shared_idx}"
                kv_shared_cache = (
                    new_cache.get(shared_layer_name)
                    if new_cache is not None
                    else _kv_share_store.get(shared_layer_name)
                )
            else:
                kv_shared_cache = None

            layer_cache, x = layer(
                x,
                positions,
                layer_cache,
                attention_mask,
                per_layer_input=(
                    per_layer_inputs[:, :, i, :]
                    if per_layer_inputs is not None
                    else None
                ),
                kv_shared_cache=kv_shared_cache,
            )

            if new_cache is not None:
                new_cache[layer_name] = layer_cache
            else:
                # During full-sequence forward pass, save K/V for all layers so
                # that later shared layers can look them up.
                _kv_share_store[layer_name] = layer_cache

        x = self.final_norm(x)
        logits = self.embedder.decode(x).astype(jnp.float32)

        if self.config.final_logit_softcap is not None:
            logits /= self.config.final_logit_softcap
            logits = jnp.tanh(logits) * self.config.final_logit_softcap

        return logits, new_cache

    def init_cache(self, batch_size, max_seq_len, dtype):
        cache = {}
        for i, layer in enumerate(self.layers):
            cache[f"layer_{i}"] = layer.init_cache(batch_size, max_seq_len, dtype)
        return cache

    def get_model_input(self):
        dummy_batch_size = 2
        dummy_seq_len = 1
        return {
            "tokens": jnp.ones((dummy_batch_size, dummy_seq_len), dtype=jnp.int32),
            "positions": jnp.ones((dummy_batch_size, dummy_seq_len), dtype=jnp.int32),
            "pixel_values": None,
            "pixel_position_ids": None,
            "cache": None,
            "attention_mask": jnp.ones(
                (dummy_batch_size, dummy_seq_len), dtype=jnp.bool_
            ),
        }
