# adapted from:
# https://github.com/google-deepmind/gemma/blob/22130bffc1e0fb4255de9758426865cf7e9430a8/gemma/peft/_lora.py
from typing import Sequence, Optional

import jax
import jax.numpy as jnp
from multimethod import multimethod
from flax import nnx
from flax.nnx.filterlib import Filter, OfType, Any as AnyOf
from gemma.peft import _einsum_utils


# ==================
# LoRA wrappers
# ==================


class LoRAEinsumAdapter(nnx.Module):
    """LoRA einsum module.

    This module only do the x @ A @ B computation.
    Use `LoRAEinsum` to wrap a `nn.Einsum` layer.

    Attributes:
        rank: The rank of the LoRA decomposition.
        einsum_str: The einsum string of the original einsum op. Should be
        `inputs,weights->outputs` (this will be internally rewritten as
        `inputs,a,b->outputs`)
        shape: The shape of the original weights before the low-rank adaptation.
        Should match the `weights` shape from the `einsum_str`.
        dtype: The dtype to use for the LoRA weights.
        rngs: Instance of nnx.Rngs for parameter initialization.
    """

    def __init__(
        self,
        rank: int,
        einsum_str: str,
        shape: Sequence[int],
        *,
        dtype: jnp.dtype = jnp.float32,
        sharding: jax.sharding.Sharding | None = None,
        rngs: nnx.Rngs,
    ):
        self.rank = rank
        self.einsum_str = einsum_str
        self.shape = shape
        # Get the einsum decomposition given the original einsum op.
        # e.g. `BTNH,NHD->BTD` becomes `BTNH,NHr,rD->BTD`
        out = _einsum_utils.get_lora_einsum_str_and_shapes(
            einsum_str=self.einsum_str,
            weights_shape=self.shape,
            rank=self.rank,
        )
        (lora_einsum_str, a_shape, b_shape) = out

        self.lora_einsum_str = lora_einsum_str
        a_init = nnx.initializers.kaiming_uniform()
        a_value = a_init(key=rngs.params(), shape=a_shape, dtype=dtype)
        b_init = nnx.initializers.zeros_init()
        b_value = b_init(key=rngs.params(), shape=b_shape, dtype=dtype)
        if sharding:
            a_value = jax.device_put(a_value, sharding)
            b_value = jax.device_put(b_value, sharding)
        self.lora_a = nnx.Param(a_value)
        self.lora_b = nnx.Param(b_value)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return jnp.einsum(self.lora_einsum_str, inputs, self.lora_a, self.lora_b)


class LoRAEinsum(nnx.Module):
    """Wrapper around `nn.Einsum` which adds a LoRA adapter."""

    def __init__(
        self,
        rank: int,
        base_module: nnx.Einsum,
        *,
        dtype: jnp.dtype | None = None,
        sharding: jax.sharding.Sharding | None = None,
        rngs: nnx.Rngs,
    ):
        self.rank = rank
        self.base_module = base_module
        self.dtype = dtype or self.base_module.kernel.dtype
        self.adapter = LoRAEinsumAdapter(
            rank=self.rank,
            einsum_str=self.base_module.einsum_str,
            shape=self.base_module.kernel_shape,
            dtype=self.dtype,
            sharding=sharding,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.base_module(inputs) + self.adapter(inputs)


# ==================
# Helper functions
# ==================


@multimethod
def _wrap_compatible_module(base_module: nnx.Einsum, rank: int, *, sharding: Optional[jax.sharding.Sharding] = None, rngs: nnx.Rngs):
    return LoRAEinsum(rank=rank, base_module=base_module, sharding=sharding, rngs=rngs)


# TODO: add methods for other LoRA layers


@multimethod
def _wrap_compatible_module(base_module, rank: int, *, sharding: Optional[jax.sharding.Sharding] = None, rngs: nnx.Rngs):
    raise ValueError(
        f"Module of type {base_module} doesn't have a compatible LoRA adapter"
    )


# TODO: add and test LoRALinear
LORA_COMPATIBLE_MODULE = AnyOf(OfType(nnx.Einsum))
LORA_MODULE = AnyOf(OfType(LoRAEinsum))
ALL_LORA_PARAMS = nnx.All(
    nnx.Param, nnx.Any(nnx.PathContains("lora_a"), nnx.PathContains("lora_b"))
)


def apply(
    root: nnx.Module,
    rank: int,
    filter: Filter = LORA_COMPATIBLE_MODULE,
    *,
    sharding: jax.sharding.Sharding | None = None,
    rngs: nnx.Rngs,
):
    matching = []  # list of (parent_module, lora_compatible_attr_name)
    for path, module in root.iter_modules():
        for attr_name, child in module.iter_children():
            # if child passes filter and is not LoRA module yet
            if filter(path, child) and not LORA_MODULE(path, child):
                matching.append((module, attr_name))
    for module, attr_name in matching:
        base_module = getattr(module, attr_name)
        lora_module = _wrap_compatible_module(base_module, rank, sharding=sharding, rngs=rngs)
        setattr(module, attr_name, lora_module)


def merge(root: nnx.Module):
    raise NotImplementedError("Merging LoRA parameters is not implemented yet")
    for path, module in root.iter_modules():
        for attr_name, child in module.iter_children():
            # if child passes filter and is not LoRA module yet
            if LORA_MODULE(path, child):
                base_module, adapter = child.base_module, child.adapter
                # TODO: this doesn't work for Einsum. Instead, we
                # need smth like (assuming lora_einsum_str = 'BTD,Dr,rNH->BTNH')
                # adapter_kernel = jnp.einsum("Dr,rNH->NDH")
                # base_module.kernel += adapter.lora_a @ adapter.lora_b
                setattr(module, attr_name, base_module)
