import logging
import re
from typing import Dict, List

import jax
from flax.core import FrozenDict
from jax import tree_util
from multimethod import multimethod


logger = logging.getLogger(__name__)


AnyDict = Dict | FrozenDict
DictOrList = List | AnyDict


def size_gb(variables: dict):
    from math import prod

    bytes = sum([prod(x.shape) * 4 for x in tree_util.tree_leaves(variables)])
    return bytes / (1024**3)


def update_tree(a: Dict, b: Dict):
    """
    Update tree a with keys from tree b.

    Semantics of this operation is the same as dict.update(), but update_tree()
    also works recursively.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            update_tree(a[key], b[key])
        else:
            a[key] = b[key]


# def eachindex(x: DictOrList):
#     if isinstance(x, List):
#         return list(range(len(x)))
#     elif isinstance(x, AnyDict):
#         return list(x.keys())


# def hasindex(x: DictOrList, idx):
#     if isinstance(x, List):
#         return 0 <= idx < len(x)
#     elif isinstance(x, AnyDict):
#         return idx in x


# def update_tree(a: DictOrList, b: DictOrList):
#     """
#     Update tree a with keys from tree b.

#     Semantics of this operation is the same as dict.update(), but update_tree()
#     also works recursively.
#     """
#     for key in eachindex(b):
#         if hasindex(a, key) and isinstance(a[key], DictOrList) and isinstance(b[key], DictOrList):
#             update_tree(a[key], b[key])
#         else:
#             a[key] = b[key]


def print_var(name: str, x):
    """
    Print some of the array properties. Useful for print-based debuging
    (which you normally shouldn't do)
    """
    if x is not None:
        jax.debug.print(
            "{}: mean={}, var={}, shape={}, dtype={}",
            name, x.mean(), x.var(), x.shape, x.dtype
        )
    else:
        jax.debug.print("{} = {}", name, x)


LIST_KEY_REGEXP = r"^([a-zA-Z0-9]+)\[(\d+)\]"
# DICT_KEY_REGEXP = r"^([a-zA-Z0-9]+)\[(\d+)\]"


def set_nested(nested: Dict, keys: List[str], val):
    """
    Set value into the nested dict according to the path of keys.

    Example:

        set_nested({}, ["a", "b", "c"], 42)
        # ==> {'a': {'b': {'c': 42}}}

        set_nested({"a": {"d": 54}}, ["a", "b[3]", "c"], 42)
        # ==> {'a': {'d': 54, 'b': [None, None, None, {'c': 42}]}}
    """
    dct = nested
    for key in keys[:-1]:
        index = None
        is_list_match = None
        if isinstance(key, str):
            is_list_match = re.match(LIST_KEY_REGEXP, key)
        if is_list_match:
            # sub-object is a list
            key, index = is_list_match.groups()
            index = int(index)
            if key not in dct:
                dct[key] = []
            lst = dct[key]
            # if list is too short, extend it
            lst.extend([None for _ in range(index + 1 - len(lst))])
            if lst[index] is None:
                lst[index] = {}
            dct = lst[index]
        else:
            # sub-object is a dict
            if key not in dct:
                dct[key] = {}
            dct = dct[key]
    dct[keys[-1]] = val
    return nested


def set_nested_attr(nested_obj, fields: List[str], val):
    """
    Set attribute value according to the path of fields.

    Like set_nested(), but for object attributes.
    """

    def ensure_field(obj, field):
        if not hasattr(obj, field):
            raise AttributeError(
                f"[set_nested_attr] '{type(obj).__name__}' object has no attribute {field}"
            )

    obj = nested_obj
    for field in fields[:-1]:
        index = None
        if isinstance(field, str) and (m := re.match(LIST_KEY_REGEXP, field)):
            field, index = m.groups()
            index = int(index)
            ensure_field(obj, field)
            lst = getattr(obj, field)
            assert index < len(
                lst
            ), f"Trying to set {type(obj)}.{field}[{index}], but the list only has length of {len(lst)}"
            obj = lst[index]
        elif isinstance(obj, dict):
            # set key instead of field
            assert field in obj, f"Nested dict doesn't have key {field}"
            obj = obj[field]
        else:
            ensure_field(obj, field)
            obj = getattr(obj, field)
    last_field = fields[-1]
    ensure_field(obj, last_field)
    old_val = getattr(obj, last_field)
    if type(old_val) is not type(val) and not isinstance(old_val, jax.ShapeDtypeStruct):
        logger.warning(
            f"Field {'.'.join(fields)} has type {type(old_val)}, "
            + f"but setting it to value of type {type(val)}"
        )
    if (
        isinstance(old_val, jax.Array)
        and isinstance(val, jax.Array)
        and (old_val.shape != val.shape or old_val.dtype != val.dtype)
    ):
        logger.warning(
            f"Field {'.'.join(fields)} has shape={old_val.shape} and dtype={old_val.dtype}, "
            + f"but setting it to array of shape={val.shape} and dtype={val.dtype}"
        )
    setattr(obj, last_field, val)
    return nested_obj


def cache_layout(model, layer_id=0):
    x = model.layers[layer_id].attention.cache_k.value
    flags = x.sum(axis=2)[0, :, 0] != 0
    return flags.astype(int)


def check_and_update_fields(args, **kwargs):
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            klass = args.__class__
            full_class_name = f"{klass.__module__}.{klass}"
            raise ValueError(f"{full_class_name} doesn't have attribute {k}")
    return args



# new style paths

def log_or_raise(msg: str, raising: bool):
    if raising:
        raise ValueError(msg)
    else:
        logger.warning(msg)


ArrayOrShapeStruct = jax.Array | jax.ShapeDtypeStruct


@multimethod
def check_compatible_values(old_value: ArrayOrShapeStruct, new_value: ArrayOrShapeStruct, location: str = "", raising: bool = False):
    loc_str = f" @ {location}" if location else ""
    if old_value.dtype != new_value.dtype:
        log_or_raise(f"Mismatch on dtype: {old_value.dtype} -> {new_value.dtype}{loc_str}", raising)
    if old_value.shape != new_value.shape:
        log_or_raise(f"Mismatch on shape: {old_value.shape} -> {new_value.shape}{loc_str}", raising)


@multimethod
def check_compatible_values(old_value, new_value, location: str = "", raising: bool = False):
    loc_str = f" @ {location}" if location else ""
    if type(old_value) != type(new_value):
        log_or_raise(f"Mismatch on object type: {type(old_value)} -> {type(new_value)}{loc_str}", raising=raising)


def get_by_path(obj, path: str | list[str]):
    keys = path if isinstance(path, list) else path.split(".")
    this = obj
    for key in keys:
        match this:
            case list():
                this = this[int(key)]
            case dict():
                if key.isdigit():
                    # int-like key can be present as text or as actual int
                    this = this.get(key) or this[int(key)]
                else:
                    this = this[key]
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
                            f"but the receiver list only has length {len(parent)} (path = {path})", raising)
            check_compatible_values(parent[idx], value, location=path, raising=raising)
            parent[idx] = value
        case dict():
            if last_key not in parent:
                log_or_raise(f"Setting value at key {last_key}, " +
                             "but the receiver dict doesn't contain it (path = {path})", raising)
            check_compatible_values(parent[last_key], value, location=path, raising=raising)
            parent[last_key] = value
        case _:
            if not hasattr(parent, last_key):
                log_or_raise(f"Setting attribute {last_key}, but object of type {type(parent)} " +
                             "doesn't have such attribute (path = {path})", raising=raising)
            check_compatible_values(getattr(parent, last_key), value, location=path, raising=raising)
            setattr(parent, last_key, value)
