"""Loading and converting Qwen3.5 PyTorch weights into the JAX model."""

import jax
import jax.numpy as jnp

from fabrique import safetensors_io
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

    Qwen3.5 checkpoints are mixed-dtype -- the DeltaNet ``A_log`` and
    ``dt_bias`` are float32 while everything else is bfloat16 -- which is why
    the loader resolves dtypes per tensor and exempts float32 tensors from the
    ``dtype`` cast.  See ``fabrique.safetensors_io``.
    """
    return safetensors_io.load_and_create_model(
        file_dir,
        model_lib.Qwen3_5,
        config,
        _get_key_and_transform_mapping(config),
        mesh=mesh,
        dtype=dtype,
        log_name="qwen3_5",
    )


def save_model_as_safetensors(
    model: model_lib.Qwen3_5,
    config: model_lib.ModelConfig,
    source_dir: str,
    output_dir: str,
) -> str:
    """Write a trained Qwen3.5 model back out in HuggingFace layout.

    The vision tower and MTP head this port does not model are copied through
    untouched, so the export loads as the architecture the checkpoint declares
    and vLLM can serve it directly.  Returns the output directory.
    """
    return safetensors_io.save_model_as_safetensors(
        model,
        _get_key_and_transform_mapping(config),
        source_dir,
        output_dir,
        log_name="qwen3_5",
    )
