# adapted from:
# https://github.com/google-deepmind/gemma/blob/22130bffc1e0fb4255de9758426865cf7e9430a8/gemma/peft/_lora.py
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from gemma.peft import _einsum_utils


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
        self.lora_a = nnx.Param(a_init(key=rngs.params(), shape=a_shape, dtype=dtype))
        b_init = nnx.initializers.zeros_init()
        self.lora_b = nnx.Param(b_init(key=rngs.params(), shape=b_shape, dtype=dtype))

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
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.base_module(inputs) + self.adapter(inputs)
