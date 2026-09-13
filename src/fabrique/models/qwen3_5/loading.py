"""Model and tokenizer loading for Qwen3.5 (dense, text-only)."""

import os

import huggingface_hub
import jax
import jax.numpy as jnp

from fabrique.models.qwen3_5 import model as model_lib
from fabrique.models.qwen3_5 import params as params_lib


def resolve_model_dir(model_id_or_dir: str) -> str:
    """Local directory for a repo ID or path, downloading if necessary."""
    if os.path.isdir(model_id_or_dir):
        return model_id_or_dir
    print(f'Downloading snapshot for "{model_id_or_dir}" from HuggingFace Hub…')
    return huggingface_hub.snapshot_download(model_id_or_dir)


def load_tokenizer(model_dir: str):
    """Tokenizer only -- no processor, since this is the text-only model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def load_model(
    model_id_or_dir: str,
    dtype: jnp.dtype = jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
    config: model_lib.ModelConfig | None = None,
) -> tuple[model_lib.ModelConfig, model_lib.Qwen3_5]:
    """Load a dense Qwen3.5 checkpoint.

    The config is read from the checkpoint's own ``config.json`` unless one is
    passed explicitly, so the weights are always the source of truth.
    """
    model_dir = resolve_model_dir(model_id_or_dir)
    if config is None:
        config = model_lib.ModelConfig.from_hf_config(model_dir)
    # Compute dtype comes from config.param_dtype; keep the two in step or the
    # model silently runs at a different precision than requested.
    config.param_dtype = dtype
    model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=mesh, dtype=dtype
    )
    return config, model
