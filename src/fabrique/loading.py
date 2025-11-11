import importlib
import logging
import pkgutil
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
from flax import nnx
from jax.sharding import NamedSharding

from fabrique import models
from fabrique.utils import ensure_path, get_by_path, keys_to_path, set_by_path

logger = logging.getLogger("fabrique")


###############################################################################
#                                  RULES                                      #
###############################################################################


@dataclass
class RuleIgnore:

    def __bool__(self):
        return False


IGNORE = RuleIgnore()


@dataclass
class LoadRule:
    in_pattern: str
    out_pattern: str | RuleIgnore
    converter: Callable | None = None

    def __iter__(self):
        yield self.in_pattern
        yield self.out_pattern
        yield self.converter


R = LoadRule


def _pattern_to_regexp(pat: str) -> str:
    pat = pat.replace(".", "\\.")
    pat = pat.replace("{i}", "(?P<i>\\d+)")
    pat = pat.replace("{j}", "(?P<j>\\d+)")
    pat = pat.replace("{k}", "(?P<k>\\d+)")
    pat = pat.replace("{n}", "(?P<n>\\d+)")
    pat = "^" + pat + "$"
    return pat


def convert_path(path: str, in_pattern: str, out_pattern: str | RuleIgnore) -> Optional[str]:
    """
    Convert path according to input and output patterns. Example:

    ```
    path = "layer_3.attn.attn_vec_einsum.w"
    in_pat = "layer_{n}.attn.attn_vec_einsum.w"
    out_pat = "blocks.{n}.attn.attn_vec_einsum.kernel"

    convert_path(path, in_pat, out_pat)
    # ==> 'blocks.{n}.attn.attn_vec_einsum.kernel'
    ```

    """
    pat_re = _pattern_to_regexp(in_pattern)
    if m := re.match(pat_re, path):
        if isinstance(out_pattern, RuleIgnore):
            return IGNORE
        return out_pattern.format(**m.groupdict())


def update_module_from_params(
    module: nnx.Module,
    rules: tuple[str, str, Any],
    params: dict,
    *,
    mesh: jax.sharding.Mesh | None = None,
) -> None:
    """
    Update Flax NNX module from a Flax Linen param tree
    """
    state = nnx.state(module)  # the model's state, a pure pytree
    pspecs = nnx.get_partition_spec(state)  # strip out the annotations from state
    for param_keys, val in jax.tree.flatten_with_path(params)[0]:
        param_path = keys_to_path(param_keys)
        for in_pattern, out_pattern, converter in rules:
            module_path = convert_path(param_path, in_pattern, out_pattern)
            if not module_path:
                continue
            # path is rules points to Param, but here we work with Array values
            module_path += ".value"
            if converter:
                val = converter(val)
            if mesh:
                pspec = get_by_path(pspecs.raw_mapping, module_path)
                val = jax.lax.with_sharding_constraint(val, NamedSharding(mesh, pspec))
            set_by_path(module, module_path, val)


def transform_path(
    path: str, rules: list[LoadRule], invert: bool = False
) -> Optional[str]:
    for rule in rules:
        in_pattern, out_pattern = rule.in_pattern, rule.out_pattern
        if invert:
            in_pattern, out_pattern = out_pattern, in_pattern
        # print(f"testing \033[94m{in_pattern}\033[0m -> \033[91m{out_pattern}\033[0m")
        if m := convert_path(path, in_pattern, out_pattern):
            return m
    else:
        return None


def transform_tree(
    tree: dict[str, Any], rules: list[LoadRule], invert=False
) -> dict[str, Any]:
    out = {}
    keys_and_vals = jax.tree.flatten_with_path(
        tree, is_leaf=lambda x: isinstance(x, nnx.Param)
    )[0]
    for keys, val in keys_and_vals:
        path = keys_to_path(keys)
        out_path = transform_path(path, rules, invert=invert)
        if out_path is None:
            logger.warning(
                f"Input path {path} isn't covered by any rule; ignoring the path"
            )
            continue
        ensure_path(out, out_path)
        set_by_path(out, out_path, val, ignore_leave_type=True)
    return out


###############################################################################
#                              Model Loading                                  #
###############################################################################


@dataclass
class LoadConfig:
    model_re: str
    loader: Callable


def find_load_configs() -> list[LoadConfig]:
    load_configs = []
    for module_info in pkgutil.iter_modules(
        models.__path__, prefix=models.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        if hasattr(module, "LOAD_CONFIG"):
            cfg = getattr(module, "LOAD_CONFIG")
            load_configs.append(cfg)
    return load_configs


def load_model(variant: str, *, mesh: jax.sharding.Mesh | None = None):
    for cfg in find_load_configs():
        if re.match(cfg.model_re, variant):
            return cfg.loader(variant, mesh=mesh)
    raise ValueError(f"Model {variant} is unknown")



# ================================
# Parameter converters
# ================================


import jax.numpy as jnp
import re
from typing import Tuple, List


def einsum_to_linear(einsum_str: str, w: jnp.ndarray) -> jnp.ndarray:
    """
    Convert einsum weight to linear (matrix multiplication) weight.

    This function takes an einsum operation string and converts the corresponding
    weight tensor to a format suitable for standard linear layers (matrix multiplication).

    Args:
        einsum_str: Einsum notation string (e.g., "BTD,NDH->BTNH")
        w: Weight array with shape matching the second operand in einsum_str

    Returns:
        Linear weight array suitable for matrix multiplication

    Examples:
        >>> # Query projection: BTD,NDH->BTNH
        >>> w_einsum = jnp.ones((8, 512, 64))  # [num_heads, hidden_size, head_dim]
        >>> w_linear = einsum_to_linear("BTD,NDH->BTNH", w_einsum)
        >>> w_linear.shape
        (512, 512)  # [num_heads * head_dim, hidden_size]

        >>> # Different pattern: BHD,DHO->BHO
        >>> w_einsum = jnp.ones((512, 64, 2048))  # [hidden, head_dim, output]
        >>> w_linear = einsum_to_linear("BHD,DHO->BHO", w_einsum)
        >>> w_linear.shape
        (2048, 4096)  # Depends on the contraction pattern
    """
    # Parse the einsum string
    parts = einsum_str.replace(" ", "").split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid einsum string: {einsum_str}. Expected format: 'ABC,DEF->GHI'")

    inputs_str, output_str = parts
    input_parts = inputs_str.split(",")

    if len(input_parts) != 2:
        raise ValueError(f"Expected exactly 2 input operands, got {len(input_parts)}")

    input_data_indices = input_parts[0]  # e.g., "BTD"
    input_weight_indices = input_parts[1]  # e.g., "NDH"
    output_indices = output_str  # e.g., "BTNH"

    # Find batch/sequence dimensions (appear in input_data and output, not in weight)
    batch_dims = [idx for idx in input_data_indices if idx in output_indices and idx not in input_weight_indices]

    # Find the contraction dimension (appears in both inputs but not in output)
    contraction_dims = [idx for idx in input_data_indices if idx in input_weight_indices and idx not in output_indices]

    if len(contraction_dims) != 1:
        raise ValueError(f"Expected exactly 1 contraction dimension, found {len(contraction_dims)}: {contraction_dims}")

    contraction_dim = contraction_dims[0]

    # Find output dimensions from weight (in output but not in batch_dims or contraction)
    weight_output_dims = [idx for idx in output_indices if idx not in batch_dims and idx != contraction_dim]

    # Find input dimensions from weight (in weight but not contraction)
    weight_only_dims = [idx for idx in input_weight_indices if idx != contraction_dim]

    # Determine the permutation needed for the weight tensor
    # We need to rearrange weight dimensions to match: [output_dims..., contraction_dim]

    # Create mapping from index character to position in weight tensor
    weight_idx_to_pos = {idx: pos for pos, idx in enumerate(input_weight_indices)}

    # Build the target order: weight_output_dims (in order they appear in output) + [contraction_dim]
    target_order_indices = []
    for idx in output_indices:
        if idx in weight_idx_to_pos and idx in weight_output_dims:
            target_order_indices.append(idx)

    # Add contraction dimension at the end
    target_order_indices.append(contraction_dim)

    # Convert to positions in the original weight tensor
    perm = [weight_idx_to_pos[idx] for idx in target_order_indices]

    # Transpose the weight
    w_transposed = jnp.transpose(w, perm)

    # Determine the final shape for linear weight
    # Format: [product of output dims, contraction dim]
    output_dim_positions = [i for i, idx in enumerate(target_order_indices) if idx != contraction_dim]
    contraction_dim_position = len(target_order_indices) - 1

    # Calculate the sizes
    output_size = 1
    for pos in output_dim_positions:
        output_size *= w_transposed.shape[pos]

    contraction_size = w_transposed.shape[contraction_dim_position]

    # Reshape to [output_size, contraction_size]
    w_linear = w_transposed.reshape(output_size, contraction_size)

    return w_linear
