"""Convert HuggingFace safetensors checkpoints to and from nnx models.

Replaces ``tunix.models.safetensors_loader`` / ``safetensors_saver``.  Two
behaviours of the tunix loader made it unusable here and are fixed below:

* it read the dtype of the *first* tensor in a file's header and applied that
  itemsize to every tensor in the file, which silently truncates a mixed-dtype
  checkpoint (see ``keep_checkpoint_float32``);
* it assumed a single ``model.safetensors``, so sharded checkpoints failed.

A "key mapping" maps checkpoint tensor names to model parameter paths:

    {regex: (replacement, (permute, reshape) | None)}

The regex must ``fullmatch`` the checkpoint key, and ``replacement`` may use
backreferences (e.g. ``r"layers.\\1.attn.q_proj.w"``).  Load applies ``permute``
then ``reshape``; save undoes them in reverse.  Checkpoint keys matching nothing
are ignored on load and passed through untouched on save.
"""

import glob
import logging
import os
import re
import shutil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors import safe_open

# {checkpoint-key regex: (model-path replacement, (permute, reshape) | None)}
KeyMapping = dict[str, tuple[str, tuple[Any, ...] | None]]

# Files a server needs next to the weights.  The source index is deliberately
# excluded: it names shards that a single-file export no longer has.
SERVING_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _compile(key_mapping: KeyMapping):
    return [(re.compile(pat), repl, tf) for pat, (repl, tf) in key_mapping.items()]


def _match(compiled, name: str):
    """Checkpoint key -> (model path, transform), or None if unmapped."""
    for pattern, repl, transform in compiled:
        if pattern.fullmatch(name):
            return pattern.sub(repl, name), transform
    return None


def _checkpoint_files(file_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(file_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {file_dir}")
    return files


def flatten_state(tree, prefix: str = "") -> dict[str, jax.Array]:
    """Flatten a pure-dict pytree into dotted keys matching a key mapping."""
    out: dict[str, jax.Array] = {}
    items = tree.items() if isinstance(tree, dict) else enumerate(tree)
    for k, v in items:
        key = f"{prefix}{k}"
        if isinstance(v, (dict, list, tuple)):
            out.update(flatten_state(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def load_and_create_model(
    file_dir: str,
    model_class: type,
    config: Any,
    key_mapping: KeyMapping,
    *,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    keep_checkpoint_float32: bool = True,
    log_name: str = "fabrique",
):
    """Load a safetensors checkpoint into a freshly constructed nnx model.

    Args:
      file_dir: Directory holding one or more ``*.safetensors`` files.
      model_class: Model type, constructed as ``model_class(config, rngs=...)``.
      config: Model config passed straight through.
      key_mapping: See the module docstring.
      mesh: If given, parameters are placed according to the model's own
        sharding annotations; otherwise everything lands on device 0.
      dtype: Cast parameters to this dtype.  ``None`` keeps checkpoint dtypes.
      keep_checkpoint_float32: Exempt tensors the checkpoint stores as float32
        from the ``dtype`` cast.  Such tensors are float32 deliberately -- they
        get exponentiated or passed through softplus (Qwen3.5's DeltaNet
        ``A_log``, ``dt_bias`` and gated-norm weights) -- so downcasting them
        loses precision for no benefit.  Harmless for uniform-dtype checkpoints.
      log_name: Prefix for the "skipped unmapped keys" log line.

    Returns:
      An instance of ``model_class`` with parameters loaded.
    """
    compiled = _compile(key_mapping)

    abstract = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(abstract)

    tensors: dict[str, np.ndarray] = {}
    keep_float32: set[str] = set()
    skipped: list[str] = []
    for path in _checkpoint_files(file_dir):
        with safe_open(path, framework="numpy") as sf:
            # safe_open handles are not iterable, so .keys() is the API here
            # rather than dict sugar.
            for name in sorted(sf.keys()):  # noqa: SIM118
                mapped = _match(compiled, name)
                if mapped is None:
                    skipped.append(name)
                    continue
                target, transform = mapped
                arr = sf.get_tensor(name)
                permute, reshape = transform if transform else (None, None)
                if permute:
                    arr = arr.transpose(permute)
                if reshape:
                    arr = arr.reshape(reshape)
                tensors[target] = arr
                if keep_checkpoint_float32 and arr.dtype == np.float32:
                    keep_float32.add(target)

    if skipped:
        logging.info(
            "%s: skipped %d unmapped checkpoint keys (expected for parts the "
            "port does not model, e.g. a vision tower or MTP head), e.g. %s",
            log_name,
            len(skipped),
            ", ".join(sorted(skipped)[:3]),
        )

    if mesh is not None:
        sharding_tree = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    else:
        device = jax.devices()[0]
        sharding_tree = jax.tree.map(lambda _: device, abs_state.to_pure_dict())

    def place(path, sharding):
        key = ".".join(str(p.key if hasattr(p, "key") else p.idx) for p in path)
        if key not in tensors:
            raise KeyError(
                f"checkpoint has no tensor for model parameter {key!r}; "
                f"check the key mapping for {log_name}"
            )
        arr = tensors[key]
        if dtype is not None and key not in keep_float32:
            arr = arr.astype(jnp.dtype(dtype))
        return jax.device_put(arr, sharding)

    state = jax.tree.map_with_path(place, sharding_tree)
    return nnx.merge(graph_def, state)


def save_model_as_safetensors(
    model: nnx.Module,
    key_mapping: KeyMapping,
    source_dir: str,
    output_dir: str,
    *,
    log_name: str = "fabrique",
) -> str:
    """Write a trained model back out in HuggingFace layout.

    Iterates the *source* checkpoint's keys rather than inverting the load
    regexes, which guarantees the exported file has exactly the keys, shapes and
    dtypes the original had.  Tensors the port does not model -- a vision tower,
    an MTP head -- are copied through untouched, so the result loads as the same
    architecture the checkpoint declares (vLLM can serve it directly); only the
    mapped weights differ.

    Returns the output directory.
    """
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)
    compiled = _compile(key_mapping)
    state = flatten_state(nnx.to_pure_dict(nnx.state(model, nnx.Param)))

    tensors: dict[str, np.ndarray] = {}
    n_trained, n_passthrough = 0, 0
    for path in _checkpoint_files(source_dir):
        with safe_open(path, framework="numpy") as sf:
            for name in sorted(sf.keys()):  # noqa: SIM118
                mapped = _match(compiled, name)
                original = sf.get_tensor(name)
                target, transform = mapped if mapped else (None, None)
                if target is None or target not in state:
                    tensors[name] = original  # vision tower, MTP, ...
                    n_passthrough += 1
                    continue
                arr = np.asarray(state[target])
                permute, reshape = transform if transform else (None, None)
                # Undo load-time (permute -> reshape) in reverse order.
                if reshape is not None:
                    shape = (
                        tuple(np.asarray(original.shape)[list(permute)])
                        if permute
                        else original.shape
                    )
                    arr = arr.reshape(shape)
                if permute:
                    arr = arr.transpose(np.argsort(permute))
                if arr.shape != original.shape:
                    raise ValueError(
                        f"{name}: exported shape {arr.shape} != checkpoint "
                        f"{original.shape}"
                    )
                # ascontiguousarray is essential: transpose returns a strided
                # view, and save_file writes the underlying buffer without
                # honouring strides -- producing a file with correct shapes and
                # silently transposed contents.
                tensors[name] = np.ascontiguousarray(arr.astype(original.dtype))
                n_trained += 1

    save_file(tensors, os.path.join(output_dir, "model.safetensors"))

    for fname in SERVING_FILES:
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    logging.info(
        "%s: wrote %d trained + %d passed-through tensors to %s",
        log_name,
        n_trained,
        n_passthrough,
        output_dir,
    )
    return output_dir
