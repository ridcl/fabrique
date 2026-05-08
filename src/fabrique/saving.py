# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for saving models with merged LoRA weights in safetensors format.

Fixes the upstream tunix saver which assumes a single model.safetensors file.
Larger models (e.g. Qwen3-VL-4B) are sharded across multiple files described
by model.safetensors.index.json; this module handles both cases.
"""

import json
import os
import re
import shutil
from typing import Any, Callable

import jax.numpy as jnp
import safetensors.numpy as safe_np
from flax import nnx

from fabrique.models.qwen3vl.loading import resolve_model_dir


def _join_path(path) -> str:
    return ".".join([str(field) for field in path])


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
    state_key_transform_fn: Callable[[str], str],
    custom_layer_extractor_fn: (
        Callable[[dict[str, list[Any]]], dict[str, list[Any]]] | None
    ) = None,
    transpose_rules: dict[str, tuple[int, ...]] | None = None,
):
    """Save a model with LoRA weights merged into safetensors format.

    Supports both single-file models (model.safetensors) and sharded models
    described by model.safetensors.index.json.  Each shard is loaded, patched,
    and saved independently so peak RAM stays proportional to the largest shard.

    Args:
        local_model_path: Directory of the base model safetensors checkpoint.
        output_dir: Directory where the merged model will be written.
        lora_model: Model instance with LoRA weights.
        rank: LoRA rank used during training.
        alpha: LoRA alpha used during training.
        state_key_transform_fn: Converts internal layer paths to safetensors keys.
        custom_layer_extractor_fn: Optional post-processing hook for the LoRA
            layer dict (same semantics as the upstream tunix version).
        transpose_rules: Optional mapping from key substring to transpose axes,
            applied to the merged delta before adding to the base weight.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Collect all LoRA (A, B) pairs indexed by their layer path string.
    lora_layers: dict[str, list[Any]] = {}
    for path, value in nnx.iter_graph(lora_model):
        if isinstance(value, nnx.LoRAParam):
            path_str = _join_path(path[:-1])
            if path_str in lora_layers:
                assert (
                    "lora_b" in path[-1]
                ), f"Expect second LoRAParam to be lora_b, got {path[-1]}"
                lora_layers[path_str].append(value)
            else:
                assert (
                    "lora_a" in path[-1]
                ), f"Expect first LoRAParam to be lora_a, got {path[-1]}"
                lora_layers[path_str] = [value]

    if custom_layer_extractor_fn:
        lora_layers = custom_layer_extractor_fn(lora_layers)

    # Pre-compute {safetensors_key: (lora_a, lora_b)} for fast per-shard lookup.
    lora_deltas: dict[str, tuple[Any, Any]] = {
        state_key_transform_fn(path): (lora_a, lora_b)
        for path, (lora_a, lora_b) in lora_layers.items()
    }

    # Determine whether the base model is single-file or sharded.
    single_path = os.path.join(local_model_path, "model.safetensors")
    index_path = os.path.join(local_model_path, "model.safetensors.index.json")

    if os.path.exists(single_path):
        shard_files = ["model.safetensors"]
    elif os.path.exists(index_path):
        with open(index_path) as f:
            index_data = json.load(f)
        seen: set[str] = set()
        shard_files = []
        for shard in index_data["weight_map"].values():
            if shard not in seen:
                seen.add(shard)
                shard_files.append(shard)
    else:
        raise FileNotFoundError(f"No safetensors weights found in {local_model_path}")

    # Process each shard: apply any LoRA deltas whose keys live in that shard.
    applied_keys: set[str] = set()
    for shard_filename in shard_files:
        shard_path = os.path.join(local_model_path, shard_filename)
        shard_state = safe_np.load_file(shard_path)

        for state_key, (lora_a, lora_b) in lora_deltas.items():
            if state_key not in shard_state:
                continue

            lora_a_val = jnp.asarray(getattr(lora_a, "value", lora_a))
            lora_b_val = jnp.asarray(getattr(lora_b, "value", lora_b))

            if lora_a_val.ndim == 3:
                d0, d1, d2 = lora_a_val.shape
                lora_a_val = lora_a_val.reshape(d0 * d1, d2)
            if lora_b_val.ndim == 3:
                d0, d1, d2 = lora_b_val.shape
                lora_b_val = lora_b_val.reshape(d0, d1 * d2)

            combined_lora = (lora_a_val @ lora_b_val) * (alpha / rank)
            if transpose_rules:
                for t_key, rule in transpose_rules.items():
                    if t_key in state_key:
                        combined_lora = combined_lora.transpose(rule)
                        break

            shard_state[state_key] += combined_lora.astype(shard_state[state_key].dtype)
            applied_keys.add(state_key)

        safe_np.save_file(shard_state, os.path.join(output_dir, shard_filename))

    missing = set(lora_deltas.keys()) - applied_keys
    assert not missing, f"LoRA layers not found in any base model shard: {missing}"

    # Copy non-safetensors files (config, tokenizer, etc.).
    for filename in os.listdir(local_model_path):
        if not filename.endswith(".safetensors"):
            src = os.path.join(local_model_path, filename)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(output_dir, filename))


# ---------------------------------------------------------------------------
# Qwen3-VL convenience wrapper
# ---------------------------------------------------------------------------


def _qwen3_state_key(lora_name: str) -> str:
    key = f"model.{lora_name}.weight".replace(".attn.", ".self_attn.")
    # Qwen3-VL checkpoints nest language model layers under 'language_model.'
    # (model.language_model.layers.*), while text-only Qwen3 checkpoints use
    # model.layers.* directly.  Only rewrite the layers prefix.
    if key.startswith("model.layers."):
        key = "model.language_model." + key[len("model.") :]
    return key


_QWEN3_TRANSPOSE_RULES: dict[str, tuple[int, ...]] = {
    "q_proj": (1, 0),
    "k_proj": (1, 0),
    "v_proj": (1, 0),
    "o_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "gate_proj": (1, 0),
    "gate": (1, 0),
}


def save_qwen3vl_lora_merged(
    model_id_or_dir: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
) -> None:
    """Save a Qwen3-VL LoRA model with weights merged into the base checkpoint."""
    if re.match(r"Qwen/Qwen3-VL-.*", model_id_or_dir):
        model_id_or_dir = resolve_model_dir(model_id_or_dir)
    save_lora_merged_model_as_safetensors(
        local_model_path=model_id_or_dir,
        output_dir=output_dir,
        lora_model=lora_model,
        rank=rank,
        alpha=alpha,
        state_key_transform_fn=_qwen3_state_key,
        transpose_rules=_QWEN3_TRANSPOSE_RULES,
    )
