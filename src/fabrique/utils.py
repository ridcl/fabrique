import logging

import jax
from multimethod import multimethod

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
                log_or_raise(
                    f"Setting value at index {idx}, "
                    + f"but the receiver list only has length {len(parent)} (path = {path})",
                    raising,
                )
            check_compatible_values(parent[idx], value, location=path, raising=raising)
            parent[idx] = value
        case dict():
            if last_key not in parent:
                log_or_raise(
                    f"Setting value at key {last_key}, "
                    + "but the receiver dict doesn't contain it (path = {path})",
                    raising,
                )
            check_compatible_values(
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
            check_compatible_values(
                getattr(parent, last_key), value, location=path, raising=raising
            )
            setattr(parent, last_key, value)
