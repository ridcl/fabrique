import jax
import jax.numpy as jnp
from flax import nnx
from gemma import gm
from fabrique.loading import LoadConfig, update_module_from_params
from fabrique.models.gemma.load_rules import RULES
from fabrique.models.gemma.modeling import Transformer


GEMMA_MODEL_MAP = {
    "gemma-3-1b-pt": (
        gm.nn.Gemma3_1B.config,
        gm.ckpts.CheckpointPath.GEMMA3_1B_PT
    ),
    "gemma-3-1b-it": (
        gm.nn.Gemma3_1B.config,
        gm.ckpts.CheckpointPath.GEMMA3_1B_IT
    ),
    "gemma-3-4b-pt": (
        gm.nn.Gemma3_4B.config,
        gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    ),
    "gemma-3-4b-it": (
        gm.nn.Gemma3_4B.config,
        gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    ),
    "gemma-3-12b-pt": (
        gm.nn.Gemma3_12B.config,
        gm.ckpts.CheckpointPath.GEMMA3_12B_IT
    ),
    "gemma-3-12b-it": (
        gm.nn.Gemma3_12B.config,
        gm.ckpts.CheckpointPath.GEMMA3_12B_IT
    ),
    "gemma-3-27b-pt": (
        gm.nn.Gemma3_27B.config,
        gm.ckpts.CheckpointPath.GEMMA3_27B_IT
    ),
    "gemma-3-27b-it": (
        gm.nn.Gemma3_27B.config,
        gm.ckpts.CheckpointPath.GEMMA3_27B_IT
    ),
}


def load_gemma(variant: str, *, mesh: jax.sharding.Mesh | None = None):
    if variant not in GEMMA_MODEL_MAP:
        available_variants_str = "\n".join(f" - {v}" for v in GEMMA_MODEL_MAP.keys())
        raise ValueError(f"Model {variant} is not available. Available Gemma models are:\n{available_variants_str}")
    config, ckpt = GEMMA_MODEL_MAP[variant]
    param_dtype = jnp.bfloat16
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )
    params = gm.ckpts.load_params(ckpt)
    update_module_from_params(model, RULES, params, mesh=mesh)
    model.vision_encoder.rngs = nnx.Rngs(0)  # otherwise rngs will be abstract array
    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model


LOAD_CONFIG = LoadConfig(
    model_re="gemma.*",
    loader=load_gemma,
)
