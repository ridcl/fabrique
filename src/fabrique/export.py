import os
import logging
from glob import glob
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.flax import load_file, save_file
from fabrique.utils import get_by_path
from fabrique.loading import convert_path, LoadRule
from fabrique.models.gemma.load_rules import HF_EXPORT_RULES


logger = logging.getLogger("fabrique")


HF_IMPORTANT_FILES = ["config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "model.safetensors.index.json"]




def _update_safetensor_dict(model, st: dict, rules: list[LoadRule]) -> None:
    matched_paths = []
    for st_path, old_value in st.items():
        for r in rules:
            # note: rules are inverted
            model_path = convert_path(st_path, r.out_pattern, r.in_pattern)
            if not model_path:
                continue
            new_value = get_by_path(model, model_path).value
            if r.converter:
                new_value = r.converter(new_value)
            if new_value.shape != old_value.shape:
                logger.warning(
                    "Incompatible shape of model and safetensor data " +
                    f"at {st_path}: {new_value.shape} != {old_value.shape}"
                )
            if new_value.dtype != old_value.dtype:
                logger.warning(
                    "Incompatible dtype of model and safetensor data " +
                    f"at {st_path}: {new_value.dtype} != {old_value.dtype}"
                )
            st[st_path] = new_value
            matched_paths.append(st_path)
            break
    logging.debug(f"Updating safetensor dict: matched {len(matched_paths)}/{len(st)} paths")


def to_huggingface(model, hf_repo_id, output_dir: str, strict=True) -> None:
    if strict and os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise ValueError(f"Output directory {output_dir} already exists and is not empty")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(hf_repo_id, repo_type="model", local_dir=output_dir)
    for filename in glob(f"{output_dir}/*.safetensors"):
        st = load_file(filename)
        _update_safetensor_dict(model, st, HF_EXPORT_RULES)
        save_file(st, filename)  # overrides the file



def main():
    import jax.numpy as jnp
    from fabrique.loading import load_model, einsum_to_linear
    tokenizer, model = load_model("gemma-3-4b-it")

    hf_repo_id = "google/gemma-3-4b-it"
    output_dir = "output/hf_export"

    path = "/home/devpod/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/model-00001-of-00002.safetensors"
    path = "/home/devpod/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/model-00002-of-00002.safetensors"
    st = load_file(path)

    w = model.blocks[0].mlp.linear.kernel.value
    # wl = einsum_to_linear("...F,NHF->...NH", w)
    # wl = jnp.split(wl, 2)[0]
    t = lambda w: einsum_to_linear("...H,HF->...F", w)
    wl = t(w)
    dl = st["language_model.model.layers.0.input_layernorm.weight"]
    jnp.all(wl == dl)

    rules = HF_EXPORT_RULES
