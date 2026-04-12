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

"""Loads Gemma 4 parameters from safetensors files.

Extends the Tunix text-only key mapping with vision tower and projector keys
so that multimodal checkpoints (e.g. gemma-4-e4b-it) can be fully loaded.

Vision weight key conventions in the HuggingFace checkpoint
-----------------------------------------------------------
All linear layers in the vision tower are wrapped in ``Gemma4ClippableLinear``,
which adds a ``linear.`` prefix to the weight key:

  model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight   [768, 768]

The projector (``embed_vision``) uses a plain ``nn.Linear`` without the wrapper:

  model.embed_vision.embedding_projection.weight                         [2560, 768]

Norms without a learnable scale (``with_scale=False``) have no checkpoint tensor
and are therefore absent from the mapping:

  model.embed_vision.embedding_pre_projection_norm   ← no weight
  vision encoder v_norm (per-layer, no weight)       ← no weight

Input / output clipping statistics keys (``input_max``, ``input_min``,
``output_max``, ``output_min``) are skipped automatically because they have no
corresponding parameter in the JAX model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.gemma4 import params_safetensors as tunix_params

from fabrique.models.gemma4 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    """Return torch_key → (nnx_key, transform) mapping for Gemma 4.

    Combines the Tunix text-only mapping with vision tower / projector entries.
    HF linear weights are stored as [out, in]; our nnx.Linear.kernel is [in, out],
    so all weight tensors need a ``((1, 0), None)`` transpose.
    """
    # Base text mapping from Tunix (handles local/global attention, PLE, MoE, …).
    mapping = tunix_params._get_key_and_transform_mapping(cfg)

    if cfg.vision_config is None:
        return mapping

    # -----------------------------------------------------------------------
    # Vision tower: patch embedder
    # -----------------------------------------------------------------------
    mapping[r"model\.vision_tower\.patch_embedder\.input_proj\.weight"] = (
        "vision_tower.patch_embedder.input_proj.kernel",
        ((1, 0), None),
    )
    mapping[r"model\.vision_tower\.patch_embedder\.position_embedding_table"] = (
        "vision_tower.patch_embedder.position_embedding_table",
        None,
    )

    # -----------------------------------------------------------------------
    # Vision tower: per-layer encoder weights
    # -----------------------------------------------------------------------
    _vis_layer = r"model\.vision_tower\.encoder\.layers\.([0-9]+)"

    # Attention projections (ClippableLinear: .linear.weight + 4 clip scalars)
    for proj in ("q", "k", "v", "o"):
        mapping[rf"{_vis_layer}\.self_attn\.{proj}_proj\.linear\.weight"] = (
            rf"vision_tower.encoder.layers.\1.self_attn.{proj}_proj.linear.kernel",
            ((1, 0), None),
        )
        for clip in ("input_min", "input_max", "output_min", "output_max"):
            mapping[rf"{_vis_layer}\.self_attn\.{proj}_proj\.{clip}"] = (
                rf"vision_tower.encoder.layers.\1.self_attn.{proj}_proj.{clip}",
                None,
            )

    # QK norms (learnable scale; v_norm has no scale → omitted)
    for norm in ("q", "k"):
        mapping[rf"{_vis_layer}\.self_attn\.{norm}_norm\.weight"] = (
            rf"vision_tower.encoder.layers.\1.self_attn.{norm}_norm.scale",
            None,
        )

    # MLP projections (ClippableLinear: .linear.weight + 4 clip scalars)
    for proj in ("gate", "up", "down"):
        mapping[rf"{_vis_layer}\.mlp\.{proj}_proj\.linear\.weight"] = (
            rf"vision_tower.encoder.layers.\1.mlp.{proj}_proj.linear.kernel",
            ((1, 0), None),
        )
        for clip in ("input_min", "input_max", "output_min", "output_max"):
            mapping[rf"{_vis_layer}\.mlp\.{proj}_proj\.{clip}"] = (
                rf"vision_tower.encoder.layers.\1.mlp.{proj}_proj.{clip}",
                None,
            )

    # Four RMSNorms per encoder layer
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        mapping[rf"{_vis_layer}\.{norm}\.weight"] = (
            rf"vision_tower.encoder.layers.\1.{norm}.scale",
            None,
        )

    # -----------------------------------------------------------------------
    # Vision projector (embed_vision)
    # -----------------------------------------------------------------------
    # embedding_pre_projection_norm has with_scale=False → no checkpoint tensor.
    mapping[r"model\.embed_vision\.embedding_projection\.weight"] = (
        "embed_vision.embedding_projection.kernel",
        ((1, 0), None),
    )

    return mapping


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    cpu_embed: bool = False,
) -> model_lib.Gemma4:
    """Load safetensors weights and return an initialised Gemma4 model.

    Args:
      file_dir: Directory containing safetensors files.
      config: Model configuration.
      mesh: Optional JAX mesh for sharding.
      dtype: Optional dtype to cast loaded parameters to.
      cpu_embed: If True, keep the large embedding tables
        (``input_embedding`` ~1.25 GiB and ``per_layer_input_embedding``
        ~5.25 GiB for E4B) on CPU while all other parameters live on the
        accelerator.  This saves ~6.5 GiB of device memory and is required
        to run the E4B model on a 24 GiB GPU.  The ``Embedder`` performs
        explicit CPU→GPU transfers during the forward pass so that lookup
        results (tiny) are moved rather than the full tables.
    """
    model = safetensors_loader.load_and_create_model(
        file_dir=file_dir,
        model_class=model_lib.Gemma4,
        config=config,
        key_mapping=_get_key_and_transform_mapping,
        mesh=mesh,
        preprocess_fn=tunix_params._make_preprocess_fn(config),
        dtype=dtype,
    )

    if cpu_embed:
        import gc as _gc
        import numpy as _np

        # Extract the large embedding tables as numpy arrays (CPU RAM) and wrap
        # them in _CPUEmbedding so NNX stores them as Static in the graphdef –
        # not as JAX-array leaves in the traced JIT state.  This allows the
        # jax.jit-compiled decode loop to run without CPU-array arguments.
        model.embedder._input_embed_np = model_lib._CPUEmbedding(
            _np.asarray(jax.device_get(model.embedder.input_embedding.value))
        )
        # Replace the large GPU Variable value with a tiny placeholder so that
        # nnx.variables(model) no longer returns the 1.3 GiB tensor.
        model.embedder.input_embedding.value = jnp.zeros(
            (1,), dtype=model.embedder.input_embedding.value.dtype
        )

        if hasattr(model.embedder, "per_layer_input_embedding"):
            model.embedder._ple_np = model_lib._CPUEmbedding(
                _np.asarray(
                    jax.device_get(model.embedder.per_layer_input_embedding.value)
                )
            )
            model.embedder.per_layer_input_embedding.value = jnp.zeros(
                (1,), dtype=model.embedder.per_layer_input_embedding.value.dtype
            )

        jax.effects_barrier()
        _gc.collect()
        # Tell the embedder to route all lookups through jax.pure_callback.
        model.embedder._cpu_offload = True

    return model
