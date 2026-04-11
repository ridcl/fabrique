# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer-by-layer consistency test: JAX Gemma 4 vs HuggingFace PyTorch.

Loads the same checkpoint into both frameworks, feeds an identical input, and
compares hidden states after every decoder layer as well as the final logits.

Usage::

    # HuggingFace model ID (downloaded automatically):
    python -m fabrique.models.gemma4.consistency_test --model_id_or_dir google/gemma-4-e4b-it

    # Local checkpoint directory:
    python -m fabrique.models.gemma4.consistency_test --model_id_or_dir /path/to/checkpoint

The script prints a diff table and exits with code 0 if the top-1 prediction
at the last token position matches across both frameworks.
"""

from __future__ import annotations

import argparse
import io
import os

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from fabrique.models.gemma4 import model as model_lib
from fabrique.models.gemma4 import params as params_lib

try:
    import accelerate  # noqa: F401
    import torch

except ImportError:
    print(
        "Consistency test requires PyTorch dependencies. Install them using\n\t"
        + "pip install torch accelerate"
    )
    raise


# ---------------------------------------------------------------------------
# Model ID / directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(model_id_or_dir: str) -> str:
    """Return a local directory path for the given model ID or local path."""
    if os.path.isdir(model_id_or_dir):
        return model_id_or_dir
    print(f'Downloading snapshot for "{model_id_or_dir}" from HuggingFace Hub…')
    return huggingface_hub.snapshot_download(model_id_or_dir)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_jax(t: "torch.Tensor | None") -> "jnp.ndarray | None":
    """Convert a PyTorch tensor to a JAX array, preserving dtype."""
    if t is None:
        return None
    dtype_map = {
        torch.bfloat16: jnp.bfloat16,
        torch.float32: jnp.float32,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.bool: jnp.bool_,
    }
    if t.dtype == torch.bool:
        return jnp.array(t.detach().cpu().numpy())
    return jnp.array(t.detach().cpu().float().numpy()).astype(
        dtype_map.get(t.dtype, jnp.float32)
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    pt_device: str = "cpu",
    pt_dtype: "torch.dtype" = None,
    jax_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple["model_lib.Gemma4", "Gemma4ForConditionalGeneration"]:
    """Load JAX and PyTorch models from the same safetensors checkpoint."""
    if pt_dtype is None:
        pt_dtype = torch.bfloat16
    model_dir = resolve_model_dir(model_id_or_dir)
    pt_model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=pt_dtype,
        device_map=pt_device,
        attn_implementation="eager",
    )
    pt_model.eval()

    import dataclasses as _dc

    gpu_config = _dc.replace(config, dtype=jax_dtype, param_dtype=jax_dtype)
    _on_gpu = jax.devices()[0].platform != "cpu"
    with jax.default_device(jax.devices()[0]):
        jax_model = params_lib.create_model_from_safe_tensors(
            model_dir,
            gpu_config,
            mesh=None,
            dtype=jax_dtype,
            cpu_embed=_on_gpu,
        )
    return jax_model, pt_model


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def compare_outputs(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    pt_device: str = "cpu",
    dtype: str = "bfloat16",
    image_path: str | None = None,
) -> bool:
    """Run a layer-by-layer hidden-state comparison between JAX and HF models.

    Loads PyTorch first, runs the HF forward pass, frees all PT memory, then
    loads the JAX model and runs the JAX forward pass.  This sequential loading
    strategy avoids having both ~8 GB models resident simultaneously.

    Args:
      model_id_or_dir: HuggingFace repo ID or local checkpoint directory.
      config: ``ModelConfig`` for the JAX model (must match the checkpoint).
      prompt: Text prompt to use as input.
      pt_device: PyTorch device string (e.g. ``'cpu'`` or ``'cuda'``).
      dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
      image_path: Optional local path to an image for multimodal testing.

    Returns:
      ``True`` if the top-1 next-token prediction matches across both frameworks.
    """
    import gc

    pt_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    jax_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32

    model_dir = resolve_model_dir(model_id_or_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    # ------------------------------------------------------------------
    # Build inputs (processor only — no model needed yet)
    # ------------------------------------------------------------------
    if image_path is not None:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        image = None
        messages = [{"role": "user", "content": prompt}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        return_tensors="pt",
        padding=True,
    )

    input_ids_pt = inputs["input_ids"].to(pt_device)
    attention_mask_pt = inputs["attention_mask"].to(pt_device)
    pixel_values_pt = inputs.get("pixel_values")
    image_position_ids_pt = inputs.get("image_position_ids")

    if pixel_values_pt is not None:
        pixel_values_pt = pixel_values_pt.to(pt_device, pt_dtype)
    if image_position_ids_pt is not None:
        image_position_ids_pt = image_position_ids_pt.to(pt_device)

    seq_len = input_ids_pt.shape[1]
    last = seq_len - 1

    # Stash numpy copies of PT inputs for later JAX conversion (done after
    # PT model is freed so we avoid peak memory overlap).
    input_ids_np = input_ids_pt.int().cpu().numpy()
    attn_mask_np = attention_mask_pt.bool().cpu().numpy()
    pixel_values_np = (
        inputs["pixel_values"].float().numpy() if pixel_values_pt is not None else None
    )
    pixel_position_ids_np = (
        image_position_ids_pt.int().cpu().numpy()
        if image_position_ids_pt is not None
        else None
    )

    # ------------------------------------------------------------------
    # PHASE 1: PyTorch forward pass
    # ------------------------------------------------------------------
    print("Loading PyTorch model…")
    pt_model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=pt_dtype,
        device_map=pt_device,
        attn_implementation="eager",
    )
    pt_model.eval()

    print("Running HF forward pass…")
    with torch.no_grad():
        hf_out = pt_model(
            input_ids=input_ids_pt,
            attention_mask=attention_mask_pt,
            pixel_values=pixel_values_pt,
            image_position_ids=image_position_ids_pt,
            use_cache=False,
        )
    hf_last_np = hf_out.logits[:, last, :].detach().cpu().float().numpy()
    hf_top1_np = int(np.argmax(hf_last_np[0]))

    # Vision encoder reference (if image provided)
    hf_vis_np = None
    if pixel_values_pt is not None and config.vision_config is not None:
        with torch.no_grad():
            hf_vis_out = pt_model.model.get_image_features(
                pixel_values_pt, image_position_ids_pt, return_dict=True
            )
        hf_vis_np = hf_vis_out.pooler_output.detach().cpu().float().numpy()

    del hf_out, pt_model
    gc.collect()
    print("PT model freed.")

    # ------------------------------------------------------------------
    # PHASE 2: JAX forward pass
    # ------------------------------------------------------------------
    # Build JAX arrays from the numpy copies we saved before.
    input_tokens_jax = jnp.array(input_ids_np, dtype=jnp.int32)
    attention_mask_jax = jnp.array(attn_mask_np)
    B, L = input_tokens_jax.shape

    cum = jnp.cumsum(attention_mask_jax, axis=1) - 1
    positions_jax = jnp.where(attention_mask_jax, cum, 0).astype(jnp.int32)

    pixel_values_jax = (
        jnp.array(pixel_values_np, dtype=jnp.float32)
        if pixel_values_np is not None
        else None
    )
    pixel_position_ids_jax = (
        jnp.array(pixel_position_ids_np, dtype=jnp.int32)
        if pixel_position_ids_np is not None
        else None
    )

    print("Loading JAX model…")
    # Override param_dtype to match the compute dtype so weights are stored in
    # bfloat16 (~8 GB) rather than float32 (~16 GB). This is required to fit on
    # a single 24 GB GPU; correctness is unaffected for inference.
    import dataclasses as _dc

    gpu_config = _dc.replace(config, dtype=jax_dtype, param_dtype=jax_dtype)
    # On GPU, keep the large embedding tables (~6.5 GiB for E4B) on CPU so
    # that the transformer layers fit in the 24 GiB device memory budget.
    _on_gpu = jax.devices()[0].platform != "cpu"
    with jax.default_device(jax.devices()[0]):
        jax_model = params_lib.create_model_from_safe_tensors(
            model_dir,
            gpu_config,
            mesh=None,
            dtype=jax_dtype,
            cpu_embed=_on_gpu,
        )

    # Vision encoder comparison (if image provided)
    if (
        pixel_values_jax is not None
        and config.vision_config is not None
        and hf_vis_np is not None
    ):
        print("\n--- Vision encoder comparison ---")
        jax_pooled, valid_mask = jax_model.vision_tower(
            pixel_values_jax, pixel_position_ids_jax
        )
        jax_proj = jax_model.embed_vision(
            jax_pooled.astype(jnp.bfloat16)
        )  # [1, output_length, D]
        jax_proj_valid = jax_proj[0][valid_mask[0]]  # [n_valid, D]
        hf_pooled = jnp.array(hf_vis_np)
        n_hf = hf_pooled.shape[0]
        n_jax = jax_proj_valid.shape[0]
        if n_hf != n_jax:
            print(f"  WARNING: HF valid tokens={n_hf}, JAX valid tokens={n_jax}")
            n_cmp = min(n_hf, n_jax)
            hf_pooled = hf_pooled[:n_cmp]
            jax_proj_valid = jax_proj_valid[:n_cmp]
        vis_diff = jnp.abs(
            jax_proj_valid.astype(jnp.float32) - hf_pooled.astype(jnp.float32)
        )
        print(
            f"  Projected vision tokens  max={float(vis_diff.max()):.4f}"
            f"  mean={float(vis_diff.mean()):.4f}"
        )

    print("\n--- Full forward pass ---")
    pos = jnp.arange(L)[None, :]
    causal = pos[:, :, None] >= pos[:, None, :]
    padding_mask = attention_mask_jax[:, None, :]
    attn_mask_jax = causal & padding_mask

    print("Running JAX forward pass…")
    # When cpu_embed=True (GPU run), some parameters live on CPU while others
    # are on GPU.  nnx.jit rejects mixed-device parameter lists, so we call
    # the unwrapped forward function directly (eager, no XLA compilation).
    _call = type(jax_model).__call__
    _forward = getattr(_call, "__wrapped__", _call)
    jax_logits, _ = _forward(
        jax_model,
        tokens=input_tokens_jax,
        positions=positions_jax,
        pixel_values=pixel_values_jax,
        pixel_position_ids=pixel_position_ids_jax,
        cache=None,
        attention_mask=attn_mask_jax,
    )

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    jax_last = jax_logits[:, last, :]
    hf_last = jnp.array(hf_last_np)

    logit_diff = jnp.abs(jax_last.astype(jnp.float32) - hf_last.astype(jnp.float32))
    print(
        f"Logits at last pos  max={float(logit_diff.max()):.4f}"
        f"  mean={float(logit_diff.mean()):.4f}"
    )

    jax_top5 = jnp.argsort(jax_last[0])[-5:][::-1].tolist()
    hf_top5 = jnp.argsort(hf_last[0])[-5:][::-1].tolist()
    print(f"JAX top-5 tokens: {jax_top5}")
    print(f" HF top-5 tokens: {hf_top5}")
    top1_match = jax_top5[0] == hf_top5[0]
    print(f"Top-1 match: {top1_match}")
    return top1_match


# ---------------------------------------------------------------------------
# Optional layer-by-layer comparison
# ---------------------------------------------------------------------------


def compare_layerwise(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    prompt: str = "Hello!",
    pt_device: str = "cpu",
    dtype: str = "bfloat16",
    image_path: str | None = None,
) -> bool:
    """Step through decoder layers one-by-one and print per-layer diffs.

    This is slower than ``compare_outputs`` but easier to debug weight-loading
    issues since it isolates which layer first diverges.

    The JAX model is executed step-by-step using the same loop structure as
    ``Gemma4.__call__``.  The HF model is run with ``output_hidden_states=True``
    so that intermediate states are available without re-implementing the loop.

    Args:
      model_id_or_dir: HuggingFace repo ID or local checkpoint directory.
      config: ``ModelConfig`` for the JAX model.
      prompt: Text prompt.
      pt_device: PyTorch device.
      dtype: Compute dtype.
      image_path: Optional image path for multimodal testing.

    Returns:
      ``True`` if the top-1 next-token prediction matches.
    """
    import itertools

    pt_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    jax_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32

    jax_model, pt_model = load_models(
        model_id_or_dir,
        config,
        pt_device=pt_device,
        pt_dtype=pt_dtype,
        jax_dtype=jax_dtype,
    )
    model_dir = resolve_model_dir(model_id_or_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    # Build inputs
    if image_path is not None:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        image = None
        messages = [{"role": "user", "content": prompt}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        return_tensors="pt",
        padding=True,
    )

    input_ids_pt = inputs["input_ids"].to(pt_device)
    attention_mask_pt = inputs["attention_mask"].to(pt_device)
    pixel_values_pt = inputs.get("pixel_values")
    image_position_ids_pt = inputs.get("image_position_ids")
    if pixel_values_pt is not None:
        pixel_values_pt = pixel_values_pt.to(pt_device, pt_dtype)
    if image_position_ids_pt is not None:
        image_position_ids_pt = image_position_ids_pt.to(pt_device)

    seq_len = input_ids_pt.shape[1]
    input_tokens_jax = to_jax(input_ids_pt.int())
    attention_mask_jax = to_jax(attention_mask_pt.bool())

    B, L = 1, seq_len
    cum = jnp.cumsum(attention_mask_jax, axis=1) - 1
    positions_jax = jnp.where(attention_mask_jax, cum, 0).astype(jnp.int32)

    pixel_values_jax = (
        to_jax(inputs["pixel_values"].to(torch.float32))
        if pixel_values_pt is not None
        else None
    )
    pixel_position_ids_jax = (
        to_jax(image_position_ids_pt.int())
        if image_position_ids_pt is not None
        else None
    )

    # Causal mask for JAX
    pos = jnp.arange(L)[None, :]
    causal = pos[:, :, None] >= pos[:, None, :]
    padding = attention_mask_jax[:, None, :]
    attn_mask_jax = causal & padding

    # ------------------------------------------------------------------
    # HF: full forward with output_hidden_states=True
    # ------------------------------------------------------------------
    with torch.no_grad():
        hf_out = pt_model(
            input_ids=input_ids_pt,
            attention_mask=attention_mask_pt,
            pixel_values=pixel_values_pt,
            image_position_ids=image_position_ids_pt,
            use_cache=False,
            output_hidden_states=True,
        )
    # hf_out.hidden_states: tuple of (n_layers+1) tensors, each [1, L, D].
    # Index 0 = embeddings, index i = output after layer i-1.
    hf_hidden = hf_out.hidden_states  # tuple

    # ------------------------------------------------------------------
    # JAX: replicate the forward pass step-by-step
    # ------------------------------------------------------------------
    x = jax_model.embedder.encode(input_tokens_jax)

    per_layer_inputs = None
    if jax_model.config.per_layer_input_dim > 0:
        per_layer_inputs = jax_model.embedder.encode_per_layer_input(
            x, input_tokens_jax
        )

    # Vision injection
    if (
        config.vision_config is not None
        and pixel_values_jax is not None
        and pixel_position_ids_jax is not None
    ):
        image_token_id = config.vision_config.image_token_id
        pooled, _ = jax_model.vision_tower(pixel_values_jax, pixel_position_ids_jax)
        proj = jax_model.embed_vision(pooled.astype(jnp.bfloat16)).astype(
            jax_model.config.dtype
        )
        num_vis = proj.shape[1]

        def _inject(h, tok, vis):
            p = jnp.where(
                tok == jnp.int32(image_token_id), size=num_vis, fill_value=-1
            )[0]
            valid = p >= 0
            p = jnp.where(valid, p, 0)
            updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[p])
            return h.at[p].set(updates)

        x = jax.vmap(_inject)(x, input_tokens_jax, proj)

    # Embedding diff (after vision injection)
    emb_diff = jnp.abs(
        x.astype(jnp.float32) - to_jax(hf_hidden[0]).astype(jnp.float32)
    ).max()
    print(f"Embedding max diff: {float(emb_diff):.6f}")

    print(f'\n{"Layer":>5}  {"max diff":>10}  {"mean diff":>10}')
    print("-" * 35)

    _kv_share_store: dict[str, dict] = {}
    for i, layer in enumerate(jax_model.layers):
        shared_idx = jax_model.kv_cache_sharing_patterns[i]
        if shared_idx != i:
            kv_shared_cache = _kv_share_store.get(f"layer_{shared_idx}")
        else:
            kv_shared_cache = None

        layer_cache, x = layer(
            x,
            positions_jax,
            cache=None,
            attn_mask=attn_mask_jax,
            per_layer_input=(
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            ),
            kv_shared_cache=kv_shared_cache,
        )
        _kv_share_store[f"layer_{i}"] = layer_cache

        # HF hidden_states[i+1] is the output of layer i.
        if i + 1 < len(hf_hidden):
            abs_diff = jnp.abs(
                x.astype(jnp.float32) - to_jax(hf_hidden[i + 1]).astype(jnp.float32)
            )
            print(
                f"{i:>5}  {float(abs_diff.max()):>10.4f}  {float(abs_diff.mean()):>10.6f}"
            )

    x = jax_model.final_norm(x)

    last = seq_len - 1
    jax_logits = jax_model.embedder.decode(x[:, last : last + 1, :])
    hf_logits = to_jax(hf_out.logits[:, last : last + 1, :])

    logit_diff = jnp.abs(jax_logits.astype(jnp.float32) - hf_logits.astype(jnp.float32))
    print(
        f"\nLogits  max={float(logit_diff.max()):.4f}"
        f"  mean={float(logit_diff.mean()):.4f}"
    )

    jax_top5 = jnp.argsort(jax_logits[0, 0])[-5:][::-1].tolist()
    hf_top5 = jnp.argsort(hf_logits[0, 0])[-5:][::-1].tolist()
    print(f"JAX top-5 tokens: {jax_top5}")
    print(f" HF top-5 tokens: {hf_top5}")
    top1_match = jax_top5[0] == hf_top5[0]
    print(f"Top-1 match: {top1_match}")
    return top1_match


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_SIZE_TO_CONFIG = {
    "e4b": model_lib.ModelConfig.gemma4_e4b,
    "e4b-it": model_lib.ModelConfig.gemma4_e4b_it,
    "2b": model_lib.ModelConfig.gemma4_e2b,
}


def _infer_config(model_id_or_dir: str) -> model_lib.ModelConfig:
    """Infer ModelConfig from the model ID or directory name."""
    name = model_id_or_dir.lower()
    # Multimodal instruct variant
    if "e4b" in name and ("it" in name or "instruct" in name):
        return model_lib.ModelConfig.gemma4_e4b_it()
    if "e4b" in name:
        return model_lib.ModelConfig.gemma4_e4b()
    if "2b" in name:
        return model_lib.ModelConfig.gemma4_e2b()
    raise ValueError(
        f"Cannot infer model size from {model_id_or_dir!r}. "
        "Pass --config explicitly."
    )


def main(
    model_id_or_dir: str = "google/gemma-4-e4b-it",
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    device: str = "cpu",
    dtype: str = "bfloat16",
    image_path: str | None = None,
    config: model_lib.ModelConfig | None = None,
    layerwise: bool = False,
) -> bool:
    """Run the consistency check.

    Can be called directly from a Python console::

        from fabrique.models.gemma4.consistency_test import main
        main('google/gemma-4-e4b-it')
        main('/path/to/local/gemma-4-e4b-it')
        main('google/gemma-4-e4b-it', image_path='/path/to/image.jpg')
    """
    if config is None:
        config = _infer_config(model_id_or_dir)
    fn = compare_layerwise if layerwise else compare_outputs
    return fn(
        model_id_or_dir=model_id_or_dir,
        config=config,
        prompt=prompt,
        pt_device=device,
        dtype=dtype,
        image_path=image_path,
    )


if __name__ == "__main__" and "__file__" in globals():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_id_or_dir",
        default="google/gemma-4-e4b-it",
        help="HuggingFace repo ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog.",
        help="Text prompt.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='PyTorch device (e.g. "cuda" or "cpu").',
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--image_path",
        default=None,
        help="Local path to an image for multimodal testing.",
    )
    parser.add_argument(
        "--config",
        choices=list(_SIZE_TO_CONFIG),
        default=None,
        help="Model variant (auto-detected from model_id_or_dir if omitted).",
    )
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Run the slower layer-by-layer comparison instead of full pass.",
    )
    _args = parser.parse_args()
    _config = _SIZE_TO_CONFIG[_args.config]() if _args.config else None
    raise SystemExit(
        0
        if main(
            model_id_or_dir=_args.model_id_or_dir,
            prompt=_args.prompt,
            device=_args.device,
            dtype=_args.dtype,
            image_path=_args.image_path,
            config=_config,
            layerwise=_args.layerwise,
        )
        else 1
    )
