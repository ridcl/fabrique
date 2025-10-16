import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from gemma.multimodal import vision_utils as vu
from fabrique.models.gemma.vision_utils import MlpBlock
from fabrique.loading import LoadRule as R, update_module_from_params

@pytest.mark.parametrize(
    "input_dim,mlp_dim,deterministic,dtype",
    [
        (4, None, True, jnp.float32),
        (4, 8, True, jnp.float32),
        # (4, None, False, jnp.float32),
        (4, None, True, jnp.bfloat16),
    ]
)
def test_mlp_block(input_dim: int, mlp_dim: int, deterministic: bool, dtype: jax.typing.DTypeLike):
    block_id = 0
    batch_size = 2
    dropout = 0.5
    rngs = nnx.Rngs(params=101, default=18)

    kw = {
        "block_id": block_id,
        "mlp_dim": mlp_dim,
        "dropout": dropout,
        "dtype_mm": dtype,
    }
    x = jax.random.normal(rngs(), (batch_size, input_dim))
    mlp_nn = vu.MlpBlock(**kw)
    variables = mlp_nn.init(rngs.params(), x)
    mlp = MlpBlock(input_dim=input_dim, **kw, rngs=rngs)
    rules = [
        R("Dense_0.kernel", "linear1.kernel"),
        R("Dense_0.bias", "linear1.bias"),
        R("Dense_1.kernel", "linear2.kernel"),
        R("Dense_1.bias", "linear2.bias"),
    ]
    update_module_from_params(mlp, rules, variables["params"])

    out_nn = mlp_nn.apply(variables, x, deterministic=deterministic)
    out = mlp(x)

    assert jnp.all(out_nn == out)