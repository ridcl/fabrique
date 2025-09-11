import re
import logging
import jax
from flax import nnx
from multimethod import multimethod
from fabrique.loading import RuleIgnore, IGNORE


logger = logging.getLogger("fabrique")


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



def log_or_raise(msg: str, raising: bool):
    if raising:
        raise ValueError(msg)
    else:
        logger.warning(msg)


ArrayOrShapeStruct = jax.Array | jax.ShapeDtypeStruct


@multimethod
def check_compatible_values(old_value: ArrayOrShapeStruct, new_value: ArrayOrShapeStruct, raising: bool = False):
    if old_value.dtype != new_value.dtype:
        log_or_raise(f"Mismatch on dtype: {old_value.dtype} -> {new_value.dtype}", raising)
    if old_value.shape != new_value.shape:
        log_or_raise(f"Mismatch on shape: {old_value.shape} -> {new_value.shape}", raising)


@multimethod
def check_compatible_values(old_value, new_value, raising: bool = False):
    if type(old_value) != type(new_value):
        log_or_raise(f"Mismatch on object type: {type(old_value)} -> {type(new_value)}", raising=raising)



def get_by_path(obj, path: str | list[str]):
    keys = path if isinstance(path, list) else path.split(".")
    this = obj
    for key in keys:
        match this:
            case list():
                this = this[int(key)]
            case dict():
                this_or_none = this.get(key)
                this = this_or_none or this[int(key)]  # fallback to int keys
            case _:
                this = getattr(this, key)
    return this


def set_by_path(obj, path: str | list[str], value, raising: bool = False):
    keys = path if isinstance(path, list) else path.split(".")
    if len(keys) == 0:
        raise ValueError("Cannot set element at empty path")
    parent = get_by_path(obj, keys[:-1])
    last_key = keys[-1]
    match parent:
        case list():
            idx = int(last_key)
            if idx >= len(parent):
                log_or_raise(f"Setting value at index {idx}, " +
                            f"but the receiver list only has length {len(parent)}", raising)
            check_compatible_values(parent[idx], value, raising=raising)
            parent[idx] = value
        case dict():
            if last_key not in parent:
                log_or_raise(f"Setting value at key {last_key}, " +
                             "but the receiver dict doesn't contain it", raising)
            check_compatible_values(parent[last_key], value, raising=raising)
            parent[last_key] = value
        case _:
            if not hasattr(parent, last_key):
                log_or_raise(f"Setting attribute {last_key}, but object of type {type(parent)} " +
                             "doesn't have such attribute", raising=raising)
            check_compatible_values(getattr(parent, last_key), value, raising=raising)
            setattr(parent, last_key, value)


def test_get_set_by_path():
    import pytest
    import jax.numpy as jnp
    from dataclasses import dataclass

    @dataclass
    class C:
        val: jax.Array

    @dataclass
    class B:
        cs: list[C]

    @dataclass
    class A:
        bs: dict[str, B]

    zeros = jnp.zeros((3, 4))
    ones = jnp.ones((3, 4))
    twos = 2 * ones
    another = jnp.ones((4,5))
    a = A({"key": B([C(ones)])})

    assert get_by_path(a, "bs.key") == a.bs["key"]
    assert get_by_path(a, "bs.key.cs.0") == a.bs["key"].cs[0]
    assert (get_by_path(a, "bs.key.cs.0.val") == a.bs["key"].cs[0].val).all()

    set_by_path(a, "bs.key.cs.0.val", twos)
    assert (a.bs["key"].cs[0].val == twos).all()
    with pytest.raises(ValueError):
        set_by_path(a, "bs.key.cs.0.val", another, raising=True)
    with pytest.raises(ValueError):
        set_by_path(a, "bs.key.cs.0.val", "wrong type", raising=True)

    set_by_path(a, "bs.key.cs.0", C(zeros))
    assert a.bs["key"].cs[0] == C(zeros)
    with pytest.raises(ValueError):
        set_by_path(a, "bs.key.cs.1", C(zeros), raising=True)

    set_by_path(a, "bs.key", B([]))
    assert a.bs["key"] == B([])
    with pytest.raises(ValueError):
        set_by_path(a, "bs.another_key", B([]), raising=True)



def update_module_from_params(module: nnx.Module, rules: tuple[str, str], params: dict, *, mesh: jax.sharding.Mesh | None = None):
    """
    Update Flax NNX module from a Flax Linen param tree
    """

    def keys_to_path(keys):
        return ".".join(key.key for key in keys)

    state = nnx.state(module)                   # the model's state, a pure pytree
    pspecs = nnx.get_partition_spec(state)      # strip out the annotations from state
    for param_keys, val in jax.tree.flatten_with_path(params)[0]:
        param_path = keys_to_path(param_keys)
        for in_pattern, out_pattern, tform in rules:
            module_path = convert_path(param_path, in_pattern, out_pattern)
            # TODO: apply transform
            # with mesh:
            sharded_val = jax.lax.with_sharding_constraint(
                val,
                get_by_path(pspecs.raw_mapping, module_path)
            )
            set_by_path(module, module_path, sharded_val)



def main():
    import numpy as np
    import jax.numpy as jnp
    from gemma import gm
    from fabrique.models.gemma.modeling import Transformer
    # src = "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.LayerNorm_{j}.bias"
    # dst = "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.LayerNorm_{j}.bias"
    in_path = "layer_3.attn.q_einsum.w"
    in_pattern = "layer_{n}.attn.q_einsum.w"
    out_pattern = "blocks.{n}.attn.q_einsum.kernel.value"

    out_path = convert_path(in_path, in_pattern, out_pattern)


    config = gm.nn.Gemma3_4B.config
    ckpt = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    params = gm.ckpts.load_params(ckpt)
    module = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    )

    mesh = jax.sharding.Mesh(devices=np.array(jax.devices())[None, None, :], axis_names=("batch", "seq", "model"))
    with mesh:
        sharded_val = jax.lax.with_sharding_constraint(
                val,
                get_by_path(pspecs.raw_mapping, module_path)
            )
    # TODO: what sharding is correct for Einsum modules with shape e.g. (8, 2560, 256)?

    val = get_by_path(params, in_path)
    set_by_path(module, out_path, val)


    param_keys, val = jax.tree.flatten_with_path(params)[0][8]