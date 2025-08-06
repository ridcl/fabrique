import jax
import jax.numpy as jnp
from flax import nnx
from gemma import gm

from fabrique.loading import update_module_from_params
from fabrique.models.gemma.load_rules import RULES
from fabrique.models.gemma.modeling import Transformer

# TODO:
# 3. add loader
# 4. add sampler


def main():
    rngs = nnx.Rngs(params=116)
    # param_dtype = jnp.bfloat16
    param_dtype = jnp.bfloat16
    config = gm.nn.Gemma3_1B.config
    # model = Transformer(config, param_dtype=param_dtype, rngs=rngs)
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )

    tokenizer = gm.text.Gemma3Tokenizer()
    tokens = jnp.array(tokenizer.encode("Once upon a time", add_bos=True)).reshape(
        -1, 1
    )
    # images = jax.random.randint(key, (1, 900, 900, 3), 0, 255, dtype=jnp.uint8)
    images = None
    positions = None
    attention_mask = None

    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)

    update_module_from_params(model, RULES, params)

    out = model(tokens=tokens, images=images)

    out_tokens = out.logits.argmax(axis=-1)
