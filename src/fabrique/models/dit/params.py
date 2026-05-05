"""Weight loading for DiT from Qwen3-VL checkpoints.

Loads the 24 vision transformer blocks from a Qwen3-VL-2B or -4B checkpoint
(norms, attention, MLP) into the corresponding DiT blocks. The new DiT-only
parameters (time_embed, cond_proj, noise_head) retain their initialised values:
zero-init for cond_proj and noise_head, random for time_embed.

patch_embed is loaded only when config.patch_size == 16 (Qwen3-VL native patch
size). The Qwen3-VL checkpoint stores a Conv3d weight over temporal_patch_size=2
frames; we average the temporal dimension before loading.

Weight shape reference for Qwen3-VL-2B/4B (hidden=1024):

  HF key                                    HF shape          DiT NNX path
  ─────────────────────────────────────────────────────────────────────────────
  model.visual.patch_embed.proj.weight      [1024,3,2,16,16]  patch_embed.proj.kernel
  model.visual.blocks.N.attn.qkv.weight     [3072, 1024]      blocks.N.attn.qkv_proj.kernel
  model.visual.blocks.N.attn.qkv.bias       [3072]            blocks.N.attn.qkv_proj.bias
  model.visual.blocks.N.attn.proj.weight    [1024, 1024]      blocks.N.attn.out_proj.kernel
  model.visual.blocks.N.attn.proj.bias      [1024]            blocks.N.attn.out_proj.bias
  model.visual.blocks.N.mlp.linear_fc1.*    [4096, 1024]      blocks.N.mlp.linear1.*
  model.visual.blocks.N.mlp.linear_fc2.*    [1024, 4096]      blocks.N.mlp.linear2.*
  model.visual.blocks.N.norm{1,2}.weight    [1024]            blocks.N.norm{1,2}.scale
  model.visual.blocks.N.norm{1,2}.bias      [1024]            blocks.N.norm{1,2}.bias
"""

from __future__ import annotations

import logging
import pathlib
import re

import jax.numpy as jnp
import safetensors
from flax import nnx

from fabrique.models.dit.model import DiT, DiTConfig

log = logging.getLogger(__name__)

# Native patch size of the Qwen3-VL vision encoder.
# patch_embed weights are only loaded when config.patch_size equals this.
_QWEN_PATCH_SIZE = 16


def _key_mapping(config: DiTConfig) -> dict:
    """Return HF-safetensors → DiT-NNX key mapping for the given config."""
    mapping = {
        # ── attention ─────────────────────────────────────────────────────────
        # qkv weight: HF [3*H, H] → NNX [H, 3*H] (transpose)
        r"model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.weight": (
            r"blocks.\1.attn.qkv_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.bias": (
            r"blocks.\1.attn.qkv_proj.bias",
            None,
        ),
        # out proj: HF [H, H] → NNX [H, H] (transpose)
        r"model\.visual\.blocks\.([0-9]+)\.attn\.proj\.weight": (
            r"blocks.\1.attn.out_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.visual\.blocks\.([0-9]+)\.attn\.proj\.bias": (
            r"blocks.\1.attn.out_proj.bias",
            None,
        ),
        # ── MLP ───────────────────────────────────────────────────────────────
        r"model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc1\.weight": (
            r"blocks.\1.mlp.linear1.kernel",
            ((1, 0), None),
        ),
        r"model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc1\.bias": (
            r"blocks.\1.mlp.linear1.bias",
            None,
        ),
        r"model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc2\.weight": (
            r"blocks.\1.mlp.linear2.kernel",
            ((1, 0), None),
        ),
        r"model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc2\.bias": (
            r"blocks.\1.mlp.linear2.bias",
            None,
        ),
        # ── norms (HF 'weight' → NNX 'scale') ────────────────────────────────
        r"model\.visual\.blocks\.([0-9]+)\.norm1\.weight": (
            r"blocks.\1.norm1.scale",
            None,
        ),
        r"model\.visual\.blocks\.([0-9]+)\.norm1\.bias": (
            r"blocks.\1.norm1.bias",
            None,
        ),
        r"model\.visual\.blocks\.([0-9]+)\.norm2\.weight": (
            r"blocks.\1.norm2.scale",
            None,
        ),
        r"model\.visual\.blocks\.([0-9]+)\.norm2\.bias": (
            r"blocks.\1.norm2.bias",
            None,
        ),
    }

    if config.patch_size == _QWEN_PATCH_SIZE:
        # Qwen3-VL Conv3d weight: [hidden, in_ch, temporal, patch, patch]
        # Permute to [in_ch, temporal, patch, patch, hidden]; temporal
        # averaging is done in _preprocess.
        mapping[r"model\.visual\.patch_embed\.proj\.weight"] = (
            "patch_embed.proj.kernel",
            ((1, 2, 3, 4, 0), None),  # reshape handled in _preprocess
        )
        mapping[r"model\.visual\.patch_embed\.proj\.bias"] = (
            "patch_embed.proj.bias",
            None,
        )

    return mapping


def _preprocess(params: dict) -> dict:
    """Average the temporal dimension of patch_embed if it was loaded.

    After the key-mapping permute (1,2,3,4,0) the weight has shape
    [in_ch, temporal, patch_h, patch_w, hidden].  We average over the
    temporal axis and reshape to [in_ch*patch_h*patch_w, hidden] to match
    the DiT PatchEmbed Linear kernel.
    """
    key = "patch_embed.proj.kernel"
    if key in params:
        w = params[key]
        if w.ndim == 5:
            # [in_ch, temporal, ph, pw, hidden] → mean over axis 1
            w = w.mean(axis=1)                  # [in_ch, ph, pw, hidden]
            in_ch, ph, pw, hidden = w.shape
            params[key] = w.reshape(in_ch * ph * pw, hidden)
    return params


def _set_param(model: DiT, dotpath: str, value: jnp.ndarray) -> None:
    """Set a parameter by dotted path, e.g. 'blocks.0.attn.qkv_proj.kernel'."""
    parts = dotpath.split(".")
    obj = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    getattr(obj, parts[-1]).value = value


def load_from_qwen_vl(
    file_dir: str,
    config: DiTConfig,
    mesh: "jax.sharding.Mesh | None" = None,
    dtype: jnp.dtype | None = None,
) -> DiT:
    """Load DiT weights from a Qwen3-VL checkpoint.

    Copies the vision transformer blocks (norms, attention projections, MLP)
    from the Qwen3-VL-2B or -4B vision encoder into the DiT model.  Parameters
    that have no pretrained counterpart (time_embed, cond_proj, noise_head,
    final_norm) keep their default initialised values.

    patch_embed is loaded only when config.patch_size == 16 (the Qwen3-VL
    native patch size); for other patch sizes it is randomly initialised.

    Args:
        file_dir: Path to the Qwen3-VL checkpoint directory.  Must contain
            ``model.safetensors`` or ``model.safetensors.index.json``.
        config: DiTConfig.  For full block weight compatibility use
            ``dit_qwen()`` (depth=24, hidden_size=1024, num_heads=16,
            intermediate_size=4096).
        mesh: Optional JAX device mesh for sharding.
        dtype: dtype to cast all loaded weights to (default: bfloat16).

    Returns:
        DiT instance with vision encoder weights loaded and conditioning
        parameters (cond_proj, noise_head) zero-initialised.
    """
    # Initialise the model for real so that unmapped params (time_embed,
    # cond_proj, noise_head) get their proper zero/random init values.
    # tunix's safetensors_loader starts from nnx.eval_shape (abstract shapes),
    # so it can't be used for partial loading — unmapped keys stay as
    # ShapeDtypeStruct and cause InvalidInputException when JIT is called.
    model = DiT(config, rngs=nnx.Rngs(0))

    files = sorted(pathlib.Path(file_dir).glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found in {file_dir}")

    key_map = _key_mapping(config)
    loaded: dict[str, object] = {}

    for fpath in files:
        with safetensors.safe_open(str(fpath), framework="numpy") as sf:
            for hf_key in sf.keys():
                for pattern, (nnx_path_tmpl, transform) in key_map.items():
                    if re.fullmatch(pattern, hf_key):
                        arr = sf.get_tensor(hf_key)
                        nnx_key = re.sub(pattern, nnx_path_tmpl, hf_key)
                        if transform is not None:
                            permute, reshape = transform
                            if permute:
                                arr = arr.transpose(permute)
                            if reshape:
                                arr = arr.reshape(reshape)
                        loaded[nnx_key] = arr
                        break

    loaded = _preprocess(loaded)

    for nnx_key, arr in loaded.items():
        jax_arr = jnp.array(arr)
        if dtype is not None:
            jax_arr = jax_arr.astype(dtype)
        _set_param(model, nnx_key, jax_arr)

    log.info("Loaded %d tensors from %s", len(loaded), file_dir)
    return model
