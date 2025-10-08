import logging

import jax
import jax.numpy as jnp
from flax import nnx
from gemma import gm
from gemma.gm.ckpts._checkpoint import _CheckpointTree
from jax.sharding import NamedSharding

from fabrique.loading import LoadConfig, update_module_from_params, transform_tree
from fabrique.models.gemma.load_rules import RULES
from fabrique.models.gemma.modeling import Transformer
from fabrique.utils import ensure_path, set_by_path

logger = logging.getLogger("fabrique")


GEMMA_MODEL_MAP = {
    "gemma-3-1b-pt": (gm.nn.Gemma3_1B.config, gm.ckpts.CheckpointPath.GEMMA3_1B_PT),
    "gemma-3-1b-it": (gm.nn.Gemma3_1B.config, gm.ckpts.CheckpointPath.GEMMA3_1B_IT),
    "gemma-3-4b-pt": (gm.nn.Gemma3_4B.config, gm.ckpts.CheckpointPath.GEMMA3_4B_IT),
    "gemma-3-4b-it": (gm.nn.Gemma3_4B.config, gm.ckpts.CheckpointPath.GEMMA3_4B_IT),
    "gemma-3-12b-pt": (gm.nn.Gemma3_12B.config, gm.ckpts.CheckpointPath.GEMMA3_12B_IT),
    "gemma-3-12b-it": (gm.nn.Gemma3_12B.config, gm.ckpts.CheckpointPath.GEMMA3_12B_IT),
    "gemma-3-27b-pt": (gm.nn.Gemma3_27B.config, gm.ckpts.CheckpointPath.GEMMA3_27B_IT),
    "gemma-3-27b-it": (gm.nn.Gemma3_27B.config, gm.ckpts.CheckpointPath.GEMMA3_27B_IT),
}


def load_gemma_with_sharding(variant: str, *, mesh: jax.sharding.Mesh | None = None):
    raise AssertionError(
        "This implementation is currently broken, use `load_gemma` instead"
    )
    if variant not in GEMMA_MODEL_MAP:
        available_variants_str = "\n".join(f" - {v}" for v in GEMMA_MODEL_MAP.keys())
        raise ValueError(
            f"Model {variant} is not available. Available Gemma models are:\n{available_variants_str}"
        )
    config, ckpt_path = GEMMA_MODEL_MAP[variant]
    param_dtype = jnp.bfloat16
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )

    graphdef, state = nnx.split(model)
    pspecs = nnx.get_partition_spec(state)
    pspecs_in_ckpt_layout = transform_tree(pspecs.raw_mapping, RULES, invert=True)
    if mesh is not None:
        sharding_in_ckpt_layout = jax.tree.map(
            lambda p: NamedSharding(mesh, p.value),
            pspecs_in_ckpt_layout,
            is_leaf=lambda p: isinstance(p, nnx.Param),
        )
        params_in_ckpt_layout = gm.ckpts.load_params(
            ckpt_path, sharding=_CheckpointTree(tree=sharding_in_ckpt_layout)
        )
    else:
        params_in_ckpt_layout = gm.ckpts.load_params(ckpt_path)
    params = transform_tree(params_in_ckpt_layout, RULES)
    # Above we ignored Rngs in vision encoder (ToNNX). Here we add it back
    ensure_path(params, "vision_encoder.rngs")
    set_by_path(params, "vision_encoder.rngs", nnx.state(nnx.Rngs(0)[1]).raw_mapping)
    # Re-create the model with actual params
    model = nnx.merge(graphdef, nnx.State(params))

    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model


def load_gemma(variant: str, *, mesh: jax.sharding.Mesh | None = None):
    if variant not in GEMMA_MODEL_MAP:
        available_variants_str = "\n".join(f" - {v}" for v in GEMMA_MODEL_MAP.keys())
        raise ValueError(
            f"Model {variant} is not available. Available Gemma models are:\n{available_variants_str}"
        )
    config, ckpt = GEMMA_MODEL_MAP[variant]
    param_dtype = jnp.bfloat16
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )
    params = gm.ckpts.load_params(ckpt)
    update_module_from_params(model, RULES, params, mesh=mesh)
    # model.vision_encoder.rngs = nnx.Rngs(0)  # otherwise rngs will be abstract array
    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model


LOAD_CONFIG = LoadConfig(
    model_re="gemma.*",
    loader=load_gemma,
)


#############################

# def main():
#     import jax
#     import numpy as np
#     from jax.sharding import NamedSharding

#     device_arr = np.array(jax.devices())[None, :]
#     mesh = jax.sharding.Mesh(devices=device_arr, axis_names=("data", "model"))
#     # variant = "gemma-3-12b-it"
#     variant = "gemma-3-4b-it"
#     t, m = load_gemma(variant, mesh=mesh)


#     config, ckpt_path = GEMMA_MODEL_MAP[variant]
#     param_dtype = jnp.bfloat16
#     model = nnx.eval_shape(
#         lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
#     )
#     PARAMS = gm.ckpts.load_params(ckpt_path)

#     state = nnx.state(model)
#     abstract_params = transform_tree(state.raw_mapping, RULES, invert=True)
#     abstract_arrays = transform_tree(state.raw_mapping, RULES, invert=True)

#     pspecs = transform_tree(nnx.get_partition_spec(state).raw_mapping, RULES, invert=True)  # strip out the annotations from state
#     pspecs_sharding = jax.tree.map(lambda p: NamedSharding(mesh, p.value), pspecs,
#                                    is_leaf=lambda p: isinstance(p, nnx.Param))

#     gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=pspecs))
#     gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=abstract_params))
#     gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=abstract_arrays))
#     params = gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=pspecs_sharding))


#     jax.lax.with_sharding_constraint(PARAMS, pspecs_sharding)
