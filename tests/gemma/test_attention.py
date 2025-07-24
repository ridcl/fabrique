import pytest
from typing import List

import jax
import jax.numpy as jnp
from flax import nnx
from gemma.gm.nn import _layers, _modules
from gemma.gm.utils import _dtype_params

from fabrique.loading import ConversionRule as R
from fabrique.loading import apply_rules
from fabrique.models.gemma.modules import Attention, AttentionType, Embedder, LayerCache


def update_module_from_params(module: nnx.Module, rules: List[R], params: dict):
    """
    Update Flax NNX module from a Flax Linen param tree
    """

    def keys_to_path(keys):
        return ".".join(key.key for key in keys)

    flat_with_keys, _ = jax.tree.flatten_with_path(params)
    flat = {keys_to_path(key): val for key, val in flat_with_keys}
    # TODO: check that shape and dtype match
    apply_rules(module, rules, flat)



# logic: define default arguments to Attention and then add various modifications
ATTN_KW = {
    "num_heads": 2,
    "num_kv_heads": 2,
    "features": 12,
    "head_dim": 16,
    "attn_type": AttentionType.LOCAL_SLIDING,
    "query_pre_attn_scalar": 1.0,
    "sliding_window_size": 3,
}

GQA_KW = {"num_heads": 8, "num_kv_heads": 4}
GLOBAL_ATTN_KW = {"attn_type": AttentionType.GLOBAL}



@pytest.mark.parametrize(
    "batch_size,seq_len,cache_size,dtype,use_cache,attn_kw_args",
    [
        (1, 10, 20, jnp.float32, True, ATTN_KW),
        (2, 10, 20, jnp.float32, True, ATTN_KW),
        (2, 10, 20, jnp.bfloat16, True, ATTN_KW),
        (2, 10, 20, jnp.bfloat16, True, ATTN_KW | GLOBAL_ATTN_KW),
        # (2, 10, 5, jnp.bfloat16, True, ATTN_KW), -- illegal: global cache < seq_len
        (2, 10, 20, jnp.bfloat16, True, ATTN_KW | GQA_KW),
        (2, 10, 20, jnp.bfloat16, True, ATTN_KW | GQA_KW | {"use_qk_norm": True}),
        (2, 10, 20, jnp.bfloat16, True, ATTN_KW | GQA_KW | {"attn_logits_soft_cap": 5.0}),
        (2, 10, 5, jnp.bfloat16, False, ATTN_KW | GQA_KW),
        (2, 10, 20, jnp.bfloat16, False, ATTN_KW | GQA_KW | GLOBAL_ATTN_KW),
        (2, 10, 20, jnp.bfloat16, False, ATTN_KW | GLOBAL_ATTN_KW),
    ]
)
def test_attention(
    batch_size: int, seq_len: int, cache_size: int, dtype, use_cache: bool, attn_kw_args: dict
):
    kw = attn_kw_args
    rngs = nnx.Rngs(params=101)
    key = rngs.params()
    if use_cache:
        cache: LayerCache = Attention.init_cache(
            cache_size=cache_size,
            num_heads=kw["num_kv_heads"],
            head_dim=kw["head_dim"],
            batch_size=batch_size,
            dtype=dtype,
        )
    else:
        cache = None
        cache_size = seq_len
    x = jax.random.normal(
        rngs.params(), (batch_size, seq_len, kw["features"]), dtype=dtype
    )
    segment_pos = jnp.tile(jnp.arange(seq_len), (batch_size, 1))

    attn_mask = jax.random.randint(
        rngs.params(), (batch_size, seq_len, cache_size), 0, 2
    )

    # overwrite self.param() to use specified dtype - just like in original code
    with _dtype_params.initialize_param_with_dtype(dtype):
        attn_nn = _modules.Attention(**kw)
        variables = attn_nn.init(key, x, segment_pos, cache, attn_mask)

    attn = Attention(**kw, param_dtype=dtype, rngs=rngs)
    rules = [
        R("attn_vec_einsum.w", "attn_vec_einsum.kernel"),
        R("qkv_einsum.w", "qkv_einsum.kernel"),
        R("q_einsum.w", "q_einsum.kernel"),
        R("kv_einsum.w", "kv_einsum.kernel"),
        R("_query_norm.scale", "_query_norm.scale"),
        R("_key_norm.scale", "_key_norm.scale"),
    ]
    update_module_from_params(attn, rules, variables["params"])

    # check model call
    new_cache_nn, out_nn = attn_nn.apply(variables, x, segment_pos, cache, attn_mask)
    new_cache, out = attn(x, segment_pos, cache, attn_mask)
    assert (out_nn == out).all()
    if use_cache:
        for p in ("k", "v", "end_index"):
            assert (new_cache_nn[p] == new_cache[p]).all()

    if use_cache:
        # check second call
        end_index = new_cache["end_index"][0].item()
        seq_len = 2
        x = jax.random.normal(
            rngs.params(), (batch_size, seq_len, kw["features"]), dtype=dtype
        )
        segment_pos = jnp.tile(jnp.array([end_index + 1, end_index + 2]), (batch_size, 1))

        attn_mask = jax.random.randint(
            rngs.params(), (batch_size, seq_len, cache_size), 0, 2
        )
        # new_cache_nn, out_nn =
        attn_nn.apply(variables, x, segment_pos, new_cache_nn, attn_mask)
        # new_cache, out =
        attn(x, segment_pos, new_cache, attn_mask)





def main():
    batch_size,seq_len,cache_size,dtype,use_cache,attn_kw_args = (2, 10, 5, jnp.bfloat16, False, ATTN_KW | GQA_KW)
    test_attention(batch_size,seq_len,cache_size,dtype,use_cache,attn_kw_args)

    rng = nnx.Rngs(0)
    head_dim = 5
    x = jax.random.normal(rng(), (3, 4, head_dim))
    norm_nn = _layers.RMSNorm()
    variables = norm_nn.init(rng(), x)
    out_nn = norm_nn.apply(variables, x)

    norm = nnx.RMSNorm(head_dim, rngs=rng)
    out = norm(x)