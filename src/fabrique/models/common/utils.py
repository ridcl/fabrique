import jax
import jax.numpy as jnp


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, jnp.newaxis, :], (1, 1, 1, n_rep, 1)).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


def padding_to_attention_mask(
    padding_mask: jax.Array, shape: tuple | None = None
) -> jax.Array:
    """
    Convert padding mask of shape (batch_size, seq_len) to attention
    mask of shape (batch_size, seq_len, seq_len).

    Args:
        padding_mask (jax.Array): Padding mask of shape (batch_size, seq_len).

    Returns:
        jas.Array: Attention mask of size (batch_size, seq_len, seq_len)
    """
    pad_attn_mask = jnp.einsum("...bi,...bj->bij", padding_mask, padding_mask)
    if shape is not None:
        # extend shape to be at least the same size as pad_attn_mask
        # e.g. if shape comes from causal_mask, its batch size will be 1, while
        # padding mask may have batch size > 1
        shape = tuple(max(s, ms) for s, ms in zip(shape, pad_attn_mask.shape))
        all_false = jnp.zeros(shape, dtype=bool)
        pad_attn_mask = jax.lax.dynamic_update_slice(
            all_false, pad_attn_mask, (0, 0, 0)
        )
    return pad_attn_mask
