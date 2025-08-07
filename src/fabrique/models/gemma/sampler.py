import jax
import jax.numpy as jnp
from flax import nnx
from gemma import gm

from fabrique.loading import update_module_from_params
from fabrique.models.gemma.load_rules import RULES
from fabrique.models.gemma.modeling import Transformer

# TODO:
# 4. add sampler



def load_gemma(variant: str):
    match variant.lower():
        case "1b":
            config = gm.nn.Gemma3_1B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_1B_IT
        case "4b":
            config = gm.nn.Gemma3_4B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
        case "12b":
            config = gm.nn.Gemma3_12B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_12B_IT
        case "27b":
            config = gm.nn.Gemma3_27B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_27B_IT
        case _:
            raise ValueError(f"Unknown Gemma variant: {variant}")
    param_dtype = jnp.bfloat16
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )
    params = gm.ckpts.load_params(ckpt)
    update_module_from_params(model, RULES, params)
    model.vision_encoder.rngs = nnx.Rngs(0)   # otherwise rngs will be abstract array
    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model




def main():
    rngs = nnx.Rngs(params=0)
    tokenizer, model = load_gemma("4b")
    tokens = jnp.array(tokenizer.encode("Once upon a time", add_bos=True)).reshape(
        1, -1
    )
    # images = jax.random.randint(key, (1, 900, 900, 3), 0, 255, dtype=jnp.uint8)
    images = None
    positions = None
    attention_mask = None

    out = model(tokens=tokens, images=images)

    out_tokens = out.logits.argmax(axis=-1)
    tokenizer.decode(out_tokens)