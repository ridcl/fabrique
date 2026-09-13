"""Loading and converting Qwen3.5 PyTorch weights into the JAX model."""

import glob
import logging
import os
import re
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors import safe_open

from fabrique.models.qwen3_5 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    """Map checkpoint keys to JAX paths, with (permute, reshape) per tensor.

    Text weights live under ``model.language_model.*``.  Vision weights
    (``model.visual.*``) and the MTP head are intentionally unmapped -- this is
    a text-only port, and the loader ignores keys with no match.
    """
    n_kv, n_q, hd, d = cfg.num_kv_heads, cfg.num_heads, cfg.head_dim, cfg.embed_dim
    q_out = hd * (2 if cfg.attn_output_gate else 1)
    lm = r"model\.language_model\."
    return {
        lm + r"embed_tokens\.weight": ("embedder.input_embedding", None),
        lm + r"norm\.weight": ("final_norm.w", None),
        # --- gated attention ---------------------------------------------
        # torch Linear is [out, in]; transpose then split heads.
        lm + r"layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.attn.q_proj.w",
            ((1, 0), (d, n_q, q_out)),
        ),
        lm + r"layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.attn.k_proj.w",
            ((1, 0), (d, n_kv, hd)),
        ),
        lm + r"layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.attn.v_proj.w",
            ((1, 0), (d, n_kv, hd)),
        ),
        lm + r"layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.attn.o_proj.w",
            ((1, 0), (n_q, hd, d)),
        ),
        lm + r"layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
            r"layers.\1.attn.q_norm.w",
            None,
        ),
        lm + r"layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
            r"layers.\1.attn.k_norm.w",
            None,
        ),
        # --- gated deltanet ----------------------------------------------
        lm + r"layers\.([0-9]+)\.linear_attn\.in_proj_qkv\.weight": (
            r"layers.\1.linear_attn.in_proj_qkv.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.in_proj_z\.weight": (
            r"layers.\1.linear_attn.in_proj_z.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.in_proj_b\.weight": (
            r"layers.\1.linear_attn.in_proj_b.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.in_proj_a\.weight": (
            r"layers.\1.linear_attn.in_proj_a.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.out_proj\.weight": (
            r"layers.\1.linear_attn.out_proj.kernel",
            ((1, 0), None),
        ),
        # torch keeps a singleton in-channel axis on the depthwise conv:
        # [conv_dim, 1, K] -> [conv_dim, K]
        lm + r"layers\.([0-9]+)\.linear_attn\.conv1d\.weight": (
            r"layers.\1.linear_attn.conv1d",
            (None, (-1, cfg.linear_conv_kernel_dim)),
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.A_log": (
            r"layers.\1.linear_attn.A_log",
            None,
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.dt_bias": (
            r"layers.\1.linear_attn.dt_bias",
            None,
        ),
        lm + r"layers\.([0-9]+)\.linear_attn\.norm\.weight": (
            r"layers.\1.linear_attn.norm.w",
            None,
        ),
        # --- mlp / norms -------------------------------------------------
        lm + r"layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.kernel",
            ((1, 0), None),
        ),
        lm + r"layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.w",
            None,
        ),
        lm + r"layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.w",
            None,
        ),
        r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
    }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen3_5:
    """Load a Qwen3.5 checkpoint into the JAX model.

    Does not use ``tunix.models.safetensors_loader``: that loader reads the
    dtype of the *first* tensor in the header and applies its itemsize to every
    tensor in the file.  Qwen3.5 checkpoints are mixed-dtype -- the DeltaNet
    ``A_log`` and ``dt_bias`` are float32 while everything else is bfloat16 --
    and the first header entry happens to be an ``A_log``, so every bfloat16
    tensor comes back with half its elements.  ``safe_open`` resolves dtypes per
    tensor, so we go through it directly.
    """
    files = sorted(glob.glob(os.path.join(file_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {file_dir}")

    key_map = _get_key_and_transform_mapping(config)
    compiled = [(re.compile(pat), repl, tf) for pat, (repl, tf) in key_map.items()]

    def map_key(name: str):
        """Checkpoint key -> (jax path, transform), or None if unmapped."""
        for pattern, repl, transform in compiled:
            if pattern.fullmatch(name):
                return pattern.sub(repl, name), transform
        return None

    abstract = nnx.eval_shape(
        lambda: model_lib.Qwen3_5(config, rngs=nnx.Rngs(params=0))
    )
    graph_def, abs_state = nnx.split(abstract)

    tensors: dict[str, jax.Array] = {}
    keep_float32: set[str] = set()
    skipped: list[str] = []
    for path in files:
        with safe_open(path, framework="numpy") as sf:
            # safe_open handles are not iterable, so .keys() is the API
            # here rather than dict sugar.
            for name in sorted(sf.keys()):  # noqa: SIM118
                mapped = map_key(name)
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
                if arr.dtype == np.float32:
                    # Qwen3.5 checkpoints are mixed-dtype on purpose: the
                    # DeltaNet A_log, dt_bias and gated-norm weights are stored
                    # in float32 because they are exponentiated / passed through
                    # softplus.  Downcasting them to the compute dtype loses
                    # precision for no benefit, so keep whatever the checkpoint
                    # chose.
                    keep_float32.add(target)

    if skipped:
        logging.info(
            "qwen3_5: skipped %d unmapped checkpoint keys (expected for the "
            "vision tower and MTP head in a text-only port), e.g. %s",
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
                f"check the key mapping in {__name__}"
            )
        arr = tensors[key]
        if dtype is not None and key not in keep_float32:
            arr = arr.astype(jnp.dtype(dtype))
        return jax.device_put(arr, sharding)

    state = jax.tree.map_with_path(place, sharding_tree)
    return nnx.merge(graph_def, state)


def _flatten_state(tree, prefix: str = "") -> dict[str, jax.Array]:
    """Flatten a pure-dict pytree into dotted keys matching the load mapping."""
    out: dict[str, jax.Array] = {}
    items = tree.items() if isinstance(tree, dict) else enumerate(tree)
    for k, v in items:
        key = f"{prefix}{k}"
        if isinstance(v, (dict, list, tuple)):
            out.update(_flatten_state(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def save_model_as_safetensors(
    model: model_lib.Qwen3_5,
    config: model_lib.ModelConfig,
    source_dir: str,
    output_dir: str,
) -> str:
    """Write a trained model back out in HuggingFace layout.

    Iterates the *source* checkpoint's keys rather than inverting the load
    regexes, which guarantees the exported file has exactly the keys, shapes and
    dtypes the original had.  Tensors this port does not model -- the vision
    tower and the MTP head -- are copied through untouched, so the result loads
    as the same architecture the checkpoint declares (vLLM can serve it
    directly); only the text weights differ.

    Returns the output directory.
    """
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)
    key_map = _get_key_and_transform_mapping(config)
    compiled = [(re.compile(pat), repl, tf) for pat, (repl, tf) in key_map.items()]
    state = _flatten_state(nnx.to_pure_dict(nnx.state(model, nnx.Param)))

    files = sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {source_dir}")

    tensors: dict[str, np.ndarray] = {}
    n_trained, n_passthrough = 0, 0
    for path in files:
        with safe_open(path, framework="numpy") as sf:
            for name in sorted(sf.keys()):  # noqa: SIM118
                target, transform = None, None
                for pattern, repl, tf in compiled:
                    if pattern.fullmatch(name):
                        target, transform = pattern.sub(repl, name), tf
                        break
                original = sf.get_tensor(name)
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

    # Copy the files a server needs alongside the weights.  The source index
    # names sharded files that no longer exist, so it is deliberately skipped.
    for fname in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    logging.info(
        "qwen3_5: wrote %d trained + %d passed-through tensors to %s",
        n_trained,
        n_passthrough,
        output_dir,
    )
    return output_dir
