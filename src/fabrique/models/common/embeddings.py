import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(xq: jax.Array, xk: jax.Array, sincos, start_pos: int):
    dtype = xq.dtype
    assert len(xq.shape) >= 4  # (*bs, seqlen, n_heads, head_dim)
    seqlen = xq.shape[-3]

    sincos_slice = lax.dynamic_slice(sincos, (start_pos, 0), (seqlen, sincos.shape[-1]))
    sincos_slice = sincos_slice[jnp.newaxis, :, jnp.newaxis, :]
    sin_pos, cos_pos = jnp.split(sincos_slice, 2, axis=-1)
    xq = (xq * cos_pos) + (rotate_half(xq) * sin_pos)
    xk = (xk * cos_pos) + (rotate_half(xk) * sin_pos)
    xq = jnp.asarray(xq, dtype=dtype)
    xk = jnp.asarray(xk, dtype=dtype)
    return xq, xk
