import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen.attention import combine_masks
from jax import lax


class KVCache(nnx.Variable):
    pass


def concatenate_to_cache(
    cache_k: KVCache,
    cache_v: KVCache,
    xk: jax.Array,
    xv: jax.Array,
    xq: jax.Array,
    attn_mask: jax.Array,
    start_pos: int,
):
    """
    Take projected key & value states from a single input token and concatenates the states to cached
    states from previous steps.
    """
    *bs, q_len, _, _ = xq.shape
    *_, max_length, _, _ = cache_k.value.shape

    # update values in cache
    indices = (0,) * len(bs) + (start_pos, 0, 0)
    # note: keys and values now have length == max_length, i.e. may be longer than xk/xv
    keys = lax.dynamic_update_slice(cache_k.value, xk, indices)
    values = lax.dynamic_update_slice(cache_v.value, xv, indices)
    cache_k.value = keys
    cache_v.value = values

    # take new xk and xv as a subarray of cached keys and values
    # of size (*bs, max_len, n_heads, head_dim)
    slices = tuple([slice(None, b) for b in bs] + [slice(None, None)] * 3)
    new_xk, new_xv = keys[slices], values[slices]

    # update causal mask to also black out tokens beyond current position
    pad_mask = jnp.broadcast_to(
        jnp.arange(max_length) < start_pos + q_len,
        tuple(bs) + (q_len, max_length),
    )
    attn_mask = combine_masks(pad_mask, attn_mask).astype(bool)  # type: ignore

    return new_xk, new_xv, attn_mask
