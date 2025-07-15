import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input array.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        self.dim = dim
        self.eps = eps
        self.param_dtype = param_dtype
        self.weight = nnx.Param(jnp.ones(self.dim, dtype=self.param_dtype))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input array.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The normalized array.

        """
        return x * jax.lax.rsqrt(
            jnp.power(x, 2).mean(axis=-1, keepdims=True) + self.eps
        )

    def __call__(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The output array after applying RMSNorm.

        """
        output = self._norm(x.astype("float32")).astype(x.dtype)
        return output * self.weight
