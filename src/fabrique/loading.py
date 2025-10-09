import importlib
import logging
import pkgutil
import re
from dataclasses import dataclass
from typing import Callable, Any, Optional

import jax
from flax import nnx
from jax.sharding import NamedSharding

from fabrique import models
from fabrique.utils import get_by_path, set_by_path, keys_to_path, ensure_path

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


def convert_path(path: str, in_pattern: str, out_pattern: str | RuleIgnore):
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
    rules: tuple[str, str],
    params: dict,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
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
