import logging
from collections.abc import Mapping, Sequence
from typing import Any

import jax
from multimethod import multimethod

# from flax import nnx

logger = logging.getLogger("fabrique")


# new style paths


def log_or_raise(msg: str, raising: bool):
    if raising:
        raise ValueError(msg)
    else:
        logger.warning(msg)


ArrayOrShapeStruct = jax.Array | jax.ShapeDtypeStruct


@multimethod
def check_compatible_values(
    old_value: ArrayOrShapeStruct,
    new_value: ArrayOrShapeStruct,
    location: str = "",
    raising: bool = False,
):
    loc_str = f" @ {location}" if location else ""
    if old_value.dtype != new_value.dtype:
        log_or_raise(
            f"Mismatch on dtype: {old_value.dtype} -> {new_value.dtype}{loc_str}",
            raising,
        )
    if old_value.shape != new_value.shape:
        log_or_raise(
            f"Mismatch on shape: {old_value.shape} -> {new_value.shape}{loc_str}",
            raising,
        )


@multimethod
def check_compatible_values(  # noqa: F811
    old_value, new_value, location: str = "", raising: bool = False
):
    loc_str = f" @ {location}" if location else ""
    if type(old_value) is not type(new_value):
        log_or_raise(
            f"Mismatch on object type: {type(old_value)} -> {type(new_value)}{loc_str}",
            raising=raising,
        )


def get_by_path(obj, path: str | list[str]) -> Any:
    """
    Given a nested object, get value at the given path.

    Example
    -------

        @dataclass
        class Point:
            x: int
            y: int

        obj = {"a": [{"b": Point(1, 2)}]}
        val = get_by_path(obj, "a.0.b.x")
        assert val == 1

    Parameters
    ----------
    obj: Mapping or Sequence or Python object
        Object to get value from
    path: str
        Dot-delimited path of the value
    raising: bool, default=False
        If True, will raise ValueError() when a non-leave element of the path
        doesn't exist. If False, will only print a warning.

    Returns
    -------
    Value at the given path
    """
    keys = path if isinstance(path, list) else path.split(".")
    this = obj
    for key in keys:
        match this:
            case Sequence():
                this = this[int(key)]
            case Mapping():
                if key.isdigit():
                    # int-like key can be present as text or as actual int
                    this = this.get(key) or this[int(key)]
                else:
                    this = this[key]
            case _:
                this = getattr(this, key)
    return this


def set_by_path(
    obj,
    path: str | list[str],
    value,
    raising: bool = False,
    ignore_leave_type: bool = False,
):
    """
    Given a nested object, set value at the given path.

    Example
    -------

        @dataclass
        class Point:
            x: int
            y: int

        obj = {"a": [{"b": Point(1, 2)}]}
        set_by_path(obj, "a.0.b.x", 4)
        assert obj == {"a": [{"b": Point(4, 2)}]}

    Parameters
    ----------
    obj: dict or list or Python object
        Object to set value at
    path: str
        Dot-delimited path at which the value must be set
    value: Any
        Value to be set
    raising: bool, default=False
        If True, will raise ValueError() when a non-leave element of the path
        doesn't exist. If False, will only print a warning.
    ignore_leave_type: bool, default=False
        If True, will ignore type mismatch on the leave value. Useful when
        working with dummy leave types (e.g. create by `ensure_path`)
    """
    keys = path if isinstance(path, list) else path.split(".")
    if len(keys) == 0:
        raise ValueError("Cannot set element at empty path")
    parent = get_by_path(obj, keys[:-1])
    last_key = keys[-1]
    match parent:
        case Sequence():
            idx = int(last_key)
            if idx >= len(parent):
                log_or_raise(
                    f"Setting value at index {idx}, "
                    + f"but the receiver list only has length {len(parent)} (path = {path})",
                    raising,
                )
            ignore_leave_type or check_compatible_values(
                parent[idx], value, location=path, raising=raising
            )
            parent[idx] = value
        case Mapping():
            if last_key not in parent:
                log_or_raise(
                    f"Setting value at key {last_key}, "
                    + "but the receiver dict doesn't contain it (path = {path})",
                    raising,
                )
            ignore_leave_type or check_compatible_values(
                parent[last_key], value, location=path, raising=raising
            )
            parent[last_key] = value
        case _:
            if not hasattr(parent, last_key):
                log_or_raise(
                    f"Setting attribute {last_key}, but object of type {type(parent)} "
                    + "doesn't have such attribute (path = {path})",
                    raising=raising,
                )
            ignore_leave_type or check_compatible_values(
                getattr(parent, last_key), value, location=path, raising=raising
            )
            setattr(parent, last_key, value)


def ensure_path(obj: dict, path: str | list[str]):
    """
    Ensure that the given object (dict) contains all parents along
    the given path. If some elements are missing, nested dicts are
    created for them.

    Note: this method works well with `get_by_path` and `set_by_path`,
    but is limited to dicts only
    """
    keys = path if isinstance(path, list) else path.split(".")
    this = obj
    for key in keys:
        match this:
            case Sequence():
                ikey = int(key)
                if len(this) > ikey:
                    this = this[ikey]
                else:
                    for _ in range(ikey - len(this) + 1):
                        this.append({})  # note: may not be optimal for leaves
                    this = this[ikey]
            case Mapping():
                if key in this or (key.isdigit() and int(key) in this):
                    this = get_by_path(this, key)
                else:
                    this[key] = {}
                    this = this[key]
            case _:
                if hasattr(this, key):
                    this = getattr(this, key)
                else:
                    setattr(this, key, {})
                    this = getattr(this, key)


def keys_to_path(keys):
    key_names = []
    for key in keys:
        if hasattr(key, "key"):  # DictKey
            key_names.append(str(key.key))
        elif hasattr(key, "name"):  # GetAttrKey
            key_names.append(key.name)
    return ".".join(key_names)
