import jax
import jax.numpy as jnp
from flax import nnx


class GemmaRMSNorm(nnx.Module):
    """
    Gemma RMSNorm layer.

    This layer (Linen version) is used in the original Gemma implementation [1]
    and behaves slightly differently than nnx.RMSNorm. We use it here
    for compatibility with Gemma's weights.
    """

    def __init__(
        self,
        num_features: int,
        *,
        param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.scale = nnx.Param(jnp.zeros(shape=num_features, dtype=param_dtype))

    def __call__(self, x):
        scale = self.scale
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs
