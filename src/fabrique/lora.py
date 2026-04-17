import os
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.filterlib import Any as AnyOf
from flax.nnx.filterlib import Filter, OfType
from gemma.peft import _einsum_utils
from multimethod import multimethod

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
    parts = adapter_einsum.split("->")
    left_parts = parts[0].split(",")

    a_indices = left_parts[1]
    b_indices = left_parts[2]

    original_einsum = lora_einsum.base_module.einsum_str
    weight_indices = original_einsum.split(",")[1].split("->")[0]

    merge_einsum_str = f"{a_indices},{b_indices}->{weight_indices}"

    # Compute delta and merge
    # delta_weights = jnp.einsum(merge_einsum_str, lora_a, lora_b)
    # merged_kernel = lora_einsum.base_module.kernel + delta_weights
    # merged_kernel = merged_kernel

    delta_weights = jnp.einsum(
        merge_einsum_str, lora_a.astype(jnp.float32), lora_b.astype(jnp.float32)
    )
    merged_kernel = lora_einsum.base_module.kernel.astype(jnp.float32) + delta_weights
    merged_kernel = merged_kernel.astype(lora_einsum.base_module.kernel)

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
                    raise NotImplementedError(
                        f"merge() is not implemented for module of type {type(child)}"
                    )


def test_divergence():
    from fabrique.loading import load_model

    rngs = nnx.Rngs(89)
    x = jnp.arange(10)[None, :]
    _, model = load_model("gemma-3-1b-it")
    q_einsum_orig = model.blocks[4].attn.q_einsum
    out_orig = model(x)
    assert isinstance(q_einsum_orig, nnx.Einsum)

    apply(model, 16, filter=LORA_COMPATIBLE_MODULE, rngs=rngs)
    q_einsum_lora = model.blocks[4].attn.q_einsum
    out_lora = model(x)
    assert isinstance(q_einsum_lora, LoRAEinsum)
    assert q_einsum_lora.base_module == q_einsum_orig
    assert jnp.all(out_orig.logits == out_lora.logits)

    # make non-zero
    q_einsum_lora.adapter.lora_b = jax.random.normal(
        rngs(), q_einsum_lora.adapter.lora_b.shape
    )
    out_lora_new = model(x)

    merge(model)
    assert isinstance(model.blocks[4].attn.q_einsum, nnx.Einsum)
    out_new = model(x)
    assert jnp.all(out_lora_new.logits.argmax(-1) == out_new.logits.argmax(-1))

    # Check difference
    max_diff = jnp.max(jnp.abs(out_lora_new.logits - out_new.logits))
    relative_diff = max_diff / jnp.max(jnp.abs(out_lora_new.logits))

    print(f"Max absolute difference: {max_diff}")
    print(f"Relative difference: {relative_diff}")
    print(f"Output before mean: {jnp.mean(out_lora_new.logits)}")
    print(f"Output after mean: {jnp.mean(out_new.logits)}")


def merge_lora_einsum(lora_einsum: LoRAEinsum, use_float64: bool = True) -> nnx.Einsum:
    """Merge LoRA weights with optional high-precision computation.

    Args:
        lora_einsum: The LoRAEinsum module to merge.
        use_float64: If True, perform merge in float64 for better numerical stability.

    Returns:
        A new nnx.Einsum module with merged weights.
    """
    lora_a = lora_einsum.adapter.lora_a.value
    lora_b = lora_einsum.adapter.lora_b.value
    base_kernel = lora_einsum.base_module.kernel.value

    original_dtype = base_kernel.dtype

    # Optionally cast to float64 for merge
    if use_float64:
        lora_a = lora_a.astype(jnp.float64)
        lora_b = lora_b.astype(jnp.float64)
        base_kernel = base_kernel.astype(jnp.float64)

    # Parse einsum strings
    adapter_einsum = lora_einsum.adapter.lora_einsum_str
    original_einsum = lora_einsum.base_module.einsum_str

    parts = adapter_einsum.split("->")
    left_parts = parts[0].split(",")
    a_indices = left_parts[1]
    b_indices = left_parts[2]
    weight_indices = original_einsum.split(",")[1].split("->")[0]
    merge_einsum_str = f"{a_indices},{b_indices}->{weight_indices}"

    # Compute delta with optimal contraction
    # delta_weights = jnp.einsum(
    #     merge_einsum_str,
    #     lora_a,
    #     lora_b,
    #     optimize='optimal'
    # )
    delta_weights = lora_a @ jnp.moveaxis(lora_b, 0, 1)

    # Merge
    merged_kernel = base_kernel + delta_weights

    # Cast back to original dtype
    if use_float64:
        merged_kernel = merged_kernel.astype(original_dtype)

    # Create new Einsum module
    merged_einsum = nnx.Einsum(
        einsum_str=original_einsum,
        kernel_shape=lora_einsum.base_module.kernel_shape,
        dtype=original_dtype,
        rngs=nnx.Rngs(0),
    )
    merged_einsum.kernel = nnx.Param(merged_kernel)

    return merged_einsum


def test_divergence_small():
    from fabrique.loading import load_model

    rngs = nnx.Rngs(89)
    x = jnp.arange(10)[None, :]
    _, model = load_model("gemma-3-1b-it")
    original_einsum = model.blocks[4].attn.q_einsum
    apply(model, 16, filter=LORA_COMPATIBLE_MODULE, rngs=rngs)
    lora_einsum = model.blocks[4].attn.q_einsum

    # Create test input
    key = jax.random.PRNGKey(42)
    # Determine input shape from einsum string
    input_indices = lora_einsum.base_module.einsum_str.split(",")[0]
    test_input = jax.random.normal(key, (2, 8, 1152))  # Adjust to your actual dims

    # Method 1: Forward through LoRAEinsum
    output_lora = lora_einsum(test_input)

    # Method 2: Manually compute base + adapter
    output_base = lora_einsum.base_module(test_input)
    output_adapter = lora_einsum.adapter(test_input)
    output_manual = output_base + output_adapter

    # Method 3: Forward through merged
    merged = merge_lora_einsum(lora_einsum)
    output_merged = merged(test_input)

    print(f"LoRA output: {output_lora.shape}, mean={jnp.mean(output_lora)}")
    print(
        f"Manual (base+adapter): {output_manual.shape}, mean={jnp.mean(output_manual)}"
    )
    print(f"Merged output: {output_merged.shape}, mean={jnp.mean(output_merged)}")

    print(f"\nDiff (lora vs manual): {jnp.max(jnp.abs(output_lora - output_manual))}")
    print(f"Diff (lora vs merged): {jnp.max(jnp.abs(output_lora - output_merged))}")
    print(f"Diff (manual vs merged): {jnp.max(jnp.abs(output_manual - output_merged))}")


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
