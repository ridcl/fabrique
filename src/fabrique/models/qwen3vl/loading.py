import json
import os

import huggingface_hub
import jax
import jax.numpy as jnp
from transformers import AutoProcessor, Qwen2VLProcessor

from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl import params as params_lib

# ---------------------------------------------------------------------------
# Model ID / directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(model_id_or_dir: str) -> str:
    """Return a local directory path for the given model ID or local path.

    If ``model_id_or_dir`` is an existing directory it is returned as-is.
    Otherwise it is treated as a HuggingFace Hub repo ID and the snapshot is
    downloaded (or retrieved from the local cache) via ``huggingface_hub``.
    """
    if os.path.isdir(model_id_or_dir):
        return model_id_or_dir
    print(f'Downloading snapshot for "{model_id_or_dir}" from HuggingFace Hub…')
    return huggingface_hub.snapshot_download(model_id_or_dir)


# ---------------------------------------------------------------------------
# Processor and model loading
# ---------------------------------------------------------------------------


def _load_processor(model_dir: str) -> AutoProcessor:
    """Load a Qwen3-VL processor without requiring PyTorch / torchvision.

    ``AutoProcessor.from_pretrained`` for Qwen3-VL normally also instantiates
    an ``AutoVideoProcessor``, which has a hard PyTorch dependency.  Tunix
    only processes images (not video), so the video processor is not needed.

    This function assembles the processor from its two torch-free components:

    * ``Qwen2VLImageProcessor`` — the *slow* image processor (PIL + NumPy only).
    * ``AutoTokenizer`` — the standard HuggingFace tokenizer.

    The two are wrapped in ``Qwen2VLProcessor`` with ``video_processor=None``.
    To satisfy the base-class type check for optional processors, we apply a
    one-time patch that allows ``None`` values for optional processor slots.

    Args:
      model_dir: Local directory of a Qwen3-VL checkpoint.

    Returns:
      A ``Qwen2VLProcessor`` instance that is API-compatible with the full
      ``AutoProcessor`` for all image + text use cases.
    """
    # One-time patch: allow None for optional processor arguments in the base
    # class validator.  This is safe — None simply means "not used".
    import transformers.processing_utils as _pu  # local to keep top-level imports clean

    if not getattr(
        _pu.ProcessorMixin.check_argument_for_proper_class, "_none_patched", False
    ):
        _orig = _pu.ProcessorMixin.check_argument_for_proper_class

        def _patched(self, argument_name, argument):
            if argument is None:
                return None
            return _orig(self, argument_name, argument)

        _patched._none_patched = True
        _pu.ProcessorMixin.check_argument_for_proper_class = _patched

    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

    tok = AutoTokenizer.from_pretrained(model_dir)

    # Image-processor settings live in preprocessor_config.json on older
    # checkpoints, but recent transformers writes them nested under the
    # "image_processor" key of processor_config.json.  ``from_pretrained`` only
    # looks for the former, so for such checkpoints it silently falls back to
    # library defaults -- notably max_pixels=1003520 instead of the value the
    # model was trained/served with, which downscales large pages.
    img_proc = None
    proc_cfg_path = os.path.join(model_dir, "processor_config.json")
    if not os.path.exists(
        os.path.join(model_dir, "preprocessor_config.json")
    ) and os.path.exists(proc_cfg_path):
        with open(proc_cfg_path) as f:
            nested = json.load(f).get("image_processor")
        if nested:
            # Qwen2VLImageProcessor.from_dict ignores size.{shortest,longest}_edge
            # and falls back to the default min_pixels/max_pixels unless they are
            # given explicitly, so map them across by hand.
            size = nested.get("size") or {}
            if "shortest_edge" in size and "min_pixels" not in nested:
                nested["min_pixels"] = size["shortest_edge"]
            if "longest_edge" in size and "max_pixels" not in nested:
                nested["max_pixels"] = size["longest_edge"]
            img_proc = Qwen2VLImageProcessor.from_dict(nested)
    if img_proc is None:
        img_proc = Qwen2VLImageProcessor.from_pretrained(model_dir)

    # The chat template lives either in tokenizer_config.json (older
    # checkpoints) or in a standalone chat_template.jinja (what recent
    # transformers versions write, e.g. Unsloth-exported models).  Check both
    # so that processor.apply_chat_template works either way.
    chat_template = None
    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path) as f:
            chat_template = json.load(f).get("chat_template")
    if chat_template is None:
        for name in ("chat_template.jinja", "chat_template.json"):
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                raw = f.read()
            chat_template = (
                json.loads(raw).get("chat_template") if name.endswith(".json") else raw
            )
            break

    return Qwen2VLProcessor(
        image_processor=img_proc,
        tokenizer=tok,
        video_processor=None,
        chat_template=chat_template,
    )


def _resolve_model_config(model_id_or_dir: str):
    model_id_or_dir_lower = model_id_or_dir.lower()
    if "-2b" in model_id_or_dir_lower:
        return model_lib.ModelConfig.qwen3vl_2b()
    elif "-4b" in model_id_or_dir_lower:
        return model_lib.ModelConfig.qwen3vl_4b()
    elif "-8b" in model_id_or_dir_lower:
        return model_lib.ModelConfig.qwen3vl_8b()
    elif "-32b" in model_id_or_dir_lower:
        return model_lib.ModelConfig.qwen3vl_32b()
    else:
        raise ValueError(
            f"Cannot resolve model config from string {model_id_or_dir}. "
            + "If it is a model id on HuggingfaceHub, make sure the spelling is correct. "
            + "If it is a local path, specify `model_config` as a separate argument"
        )


def load_model(
    model_id_or_dir: str,
    dtype: jnp.dtype = jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
    config: model_lib.ModelConfig | None = None,
) -> tuple[Qwen2VLProcessor, model_lib.Qwen3VL]:
    """Load a Qwen3-VL processor and model.

    Args:
      model_id_or_dir: HuggingFace repo ID or local checkpoint directory.
      dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
      config: Instance of Qwen3-VL ModelConfig to use (optional).

    Returns:
      Tuple of Qwen3-VL processor and model.
    """
    model_dir = resolve_model_dir(model_id_or_dir)
    config = config or _resolve_model_config(model_id_or_dir)

    # with jax.default_device(jax.devices()[0]):
    model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=mesh, dtype=dtype
    )

    processor = _load_processor(model_dir)
    return processor, model
