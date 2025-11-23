import os
from typing import Optional, Sequence
from multimethod import multimethod

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.filterlib import Any as AnyOf
from flax.nnx.filterlib import Filter, OfType
from gemma.peft import _einsum_utils


# ==================
# LoRA wrappers
# ==================

# adapted from:
# https://github.com/google-deepmind/gemma/blob/22130bffc1e0fb4255de9758426865cf7e9430a8/gemma/peft/_lora.py

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
def _wrap_compatible_module(
    base_module: nnx.Einsum,
    rank: int,
    *,
    sharding: Optional[jax.sharding.Sharding] = None,
    rngs: nnx.Rngs,
):
    return LoRAEinsum(rank=rank, base_module=base_module, sharding=sharding, rngs=rngs)


# TODO: add methods for other LoRA layers


@multimethod
def _wrap_compatible_module(
    base_module,
    rank: int,
    *,
    sharding: Optional[jax.sharding.Sharding] = None,
    rngs: nnx.Rngs,
):
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
        lora_module = _wrap_compatible_module(
            base_module, rank, sharding=sharding, rngs=rngs
        )
        setattr(module, attr_name, lora_module)



def _merge_lora_einsum_inplace(lora_einsum: LoRAEinsum) -> None:
    """Merge LoRA weights into the base module in-place and remove adapter.

    This modifies the base_module kernel directly and sets adapter weights to zero.
    """
    # Get the LoRA matrices
    lora_a = lora_einsum.adapter.lora_a
    lora_b = lora_einsum.adapter.lora_b

    # Parse einsum strings to build merge contraction
    adapter_einsum = lora_einsum.adapter.lora_einsum_str
    parts = adapter_einsum.split('->')
    left_parts = parts[0].split(',')

    a_indices = left_parts[1]
    b_indices = left_parts[2]

    original_einsum = lora_einsum.base_module.einsum_str
    weight_indices = original_einsum.split(',')[1].split('->')[0]

    merge_einsum_str = f'{a_indices},{b_indices}->{weight_indices}'

    # Compute delta and merge
    delta_weights = jnp.einsum(merge_einsum_str, lora_a, lora_b)
    merged_kernel = lora_einsum.base_module.kernel + delta_weights

    # Update base module kernel
    lora_einsum.base_module.kernel = merged_kernel

    # Zero out LoRA weights (optional, to indicate they're merged)
    lora_einsum.adapter.lora_a = jnp.zeros_like(lora_a)
    lora_einsum.adapter.lora_b = jnp.zeros_like(lora_b)


def merge(root: nnx.Module):
    for path, module in root.iter_modules():
        for attr_name, child in module.iter_children():
            # if child passes filter and is not LoRA-free yet
            if LORA_MODULE(path, child):
                if isinstance(child.base_module, nnx.Einsum):
                    _merge_lora_einsum_inplace(child)
                    setattr(module, attr_name, child.base_module)
                else:
                    # for linear, should be as easy as:
                    # base_module.kernel += adapter.lora_a @ adapter.lora_b
                    # but I don't have a good test case at the moment
                    raise NotImplementedError(f"merge() is not implemented for module of type {type(child)}")


# TODO: test merge, move both tests to `tests/`


# =======================
# Save/Load
# =======================


def save(model, ckpt_path: str, filter=ALL_LORA_PARAMS):
    ckpt_path = os.path.abspath(ckpt_path)
    checkpointer = ocp.StandardCheckpointer()
    _graphdef, lora_state, _other_state = nnx.split(model, filter, ...)
    checkpointer.save(ckpt_path, lora_state)


def load(model, ckpt_path: str, filter=ALL_LORA_PARAMS):
    ckpt_path = os.path.abspath(ckpt_path)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, lora_state, other_state = nnx.split(model, filter, ...)
    loaded_state = checkpointer.restore(ckpt_path, lora_state)
    del lora_state  # free memory; note that old model still references it
    model = nnx.merge(graphdef, loaded_state, other_state)
    return model


def latest_checkpoint_path(ckpt_base_path: str):
    if not os.path.exists(ckpt_base_path):
        return None
    filenames = os.listdir(ckpt_base_path)
    if len(filenames) == 0:
        return None
    latest = sorted(filenames)[-1]
    return os.path.join(ckpt_base_path, latest)
