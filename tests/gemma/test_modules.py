from typing import List

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from gemma.gm.nn import _layers, _modules
from gemma.gm.utils import _dtype_params

from fabrique.loading import ConversionRule as R
from fabrique.loading import apply_rules
from fabrique.models.gemma.modules import (
    Attention,
    AttentionType,
    Block,
    Embedder,
    FeedForward,
    LayerCache,
)


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


def test_embedder():
    batch_size = 2
    seq_len = 5
    vocab_size = 32
    embed_dim = 16
    vision_proj_dim = 16
    param_dtype = jnp.bfloat16
    rngs = nnx.Rngs(params=102)
    key = rngs.params()
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    vectors = jax.random.randint(key, (batch_size, embed_dim), 0, vocab_size)

    emb_nn = _modules.Embedder(
        vocab_size=vocab_size, embed_dim=embed_dim, vision_proj_dim=vision_proj_dim
    )
    variables = emb_nn.init(key, jnp.ones((seq_len, vision_proj_dim)), method=_modules.Embedder.encode_vision)

    emb = Embedder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        vision_proj_dim=vision_proj_dim,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    rules = [
        R("input_embedding", "input_embedding_table"),
        R("mm_soft_embedding_norm.scale", "mm_soft_embedding_norm.scale"),
        R("mm_input_projection.w", "mm_input_projection.kernel"),
    ]
    update_module_from_params(emb, rules, variables["params"])

    # emb.input_embedding_table = params["params"]["input_embedding"]

    encoded_nn = emb_nn.apply(variables, tokens, method=_modules.Embedder.encode)
    encoded = emb.encode(tokens)
    assert (encoded_nn == encoded).all()

    decoded_nn = emb_nn.apply(variables, vectors, method=_modules.Embedder.decode)
    decoded = emb.decode(vectors)
    assert (decoded_nn == decoded).all()

    vis_encoded_nn = emb_nn.apply(variables, vectors, method=_modules.Embedder.encode_vision)
    vis_encoded = emb.encode_vision(vectors)
    assert (vis_encoded_nn == vis_encoded).all()


# attention accepts a whole lot of arguments, so we define the default ones
# and then modify them for various test cases
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
        (
            2,
            10,
            20,
            jnp.bfloat16,
            True,
            ATTN_KW | GQA_KW | {"attn_logits_soft_cap": 5.0},
        ),
        (2, 10, 5, jnp.bfloat16, False, ATTN_KW | GQA_KW),
        (2, 10, 20, jnp.bfloat16, False, ATTN_KW | GQA_KW | GLOBAL_ATTN_KW),
        (2, 10, 20, jnp.bfloat16, False, ATTN_KW | GLOBAL_ATTN_KW),
    ],
)
def test_attention(
    batch_size: int,
    seq_len: int,
    cache_size: int,
    dtype,
    use_cache: bool,
    attn_kw_args: dict,
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
        segment_pos = jnp.tile(
            jnp.array([end_index + 1, end_index + 2]), (batch_size, 1)
        )

        attn_mask = jax.random.randint(
            rngs.params(), (batch_size, seq_len, cache_size), 0, 2
        )
        # new_cache_nn, out_nn =
        attn_nn.apply(variables, x, segment_pos, new_cache_nn, attn_mask)
        # new_cache, out =
        attn(x, segment_pos, new_cache, attn_mask)


@pytest.mark.parametrize(
    "transpose_gating_einsum,dtype",
    [
        (True, jnp.float32),
        (True, jnp.bfloat16),
        (False, jnp.float32),
        (False, jnp.bfloat16),
    ],
)
def test_feedforward(transpose_gating_einsum: bool, dtype):
    transpose_gating_einsum = True  # TODO
    dtype = jnp.bfloat16  # TODO
    rngs = nnx.Rngs(params=14)
    batch_size = 1
    seq_len = 5
    features: int = 12
    hidden_dim = 6
    x = jax.random.normal(rngs.params(), (batch_size, seq_len, features), dtype=dtype)

    with _dtype_params.initialize_param_with_dtype(dtype):
        ffw_nn = _modules.FeedForward(features, hidden_dim, transpose_gating_einsum)
        variables = ffw_nn.init(rngs.params(), x)

    ffw = FeedForward(
        features, hidden_dim, transpose_gating_einsum, param_dtype=dtype, rngs=rngs
    )
    rules = [
        R("gating_einsum", "gating.kernel"),
        R("linear", "linear.kernel"),
    ]
    update_module_from_params(ffw, rules, variables["params"])

    out_nn = ffw_nn.apply(variables, x)
    out = ffw(x)
    assert (out_nn == out).all()
    assert out_nn.dtype == out.dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_block(dtype: jax.typing.DTypeLike):
    rngs = nnx.Rngs(params=101)
    key = rngs.params()
    batch_size = 2
    seq_len = 10
    cache_size = 20
    kw = ATTN_KW | {
        "hidden_dim": 32,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "transpose_gating_einsum": False,
        "embed_dim": ATTN_KW["features"],
    }
    del kw["features"]

    cache: LayerCache = Attention.init_cache(
        cache_size=cache_size,
        num_heads=kw["num_kv_heads"],
        head_dim=kw["head_dim"],
        batch_size=batch_size,
        dtype=dtype,
    )
    x = jax.random.normal(
        rngs.params(), (batch_size, seq_len, kw["embed_dim"]), dtype=dtype
    )
    segment_pos = jnp.tile(jnp.arange(seq_len), (batch_size, 1))
    attn_mask = jax.random.randint(
        rngs.params(), (batch_size, seq_len, cache_size), 0, 2
    )

    # overwrite self.param() to use specified dtype - just like in original code
    with _dtype_params.initialize_param_with_dtype(dtype):
        block_nn = _modules.Block(**kw)
        variables = block_nn.init(key, x, segment_pos, cache, attn_mask)

    block = Block(**kw, param_dtype=dtype, rngs=rngs)
    rules = [
        # attn
        R("attn.attn_vec_einsum.w", "attn.attn_vec_einsum.kernel"),
        R("attn.qkv_einsum.w", "attn.qkv_einsum.kernel"),
        R("attn.q_einsum.w", "attn.q_einsum.kernel"),
        R("attn.kv_einsum.w", "attn.kv_einsum.kernel"),
        R("attn._query_norm.scale", "attn._query_norm.scale"),
        R("attn._key_norm.scale", "attn._key_norm.scale"),
        # ffw
        R("mlp.gating_einsum", "mlp.gating.kernel"),
        R("mlp.linear", "mlp.linear.kernel"),
        # norms
        R("pre_attention_norm.scale", "pre_attention_norm.scale"),
        R("post_attention_norm.scale", "post_attention_norm.scale"),
        R("pre_ffw_norm.scale", "pre_ffw_norm.scale"),
        R("post_ffw_norm.scale", "post_ffw_norm.scale"),
    ]
    update_module_from_params(block, rules, variables["params"])

    # check model call
    new_cache_nn, out_nn = block_nn.apply(variables, x, segment_pos, cache, attn_mask)
    new_cache, out = block(x, segment_pos, cache, attn_mask)
    assert (out_nn == out).all()
    for p in ("k", "v", "end_index"):
        assert (new_cache_nn[p] == new_cache[p]).all()


def main():
    batch_size, seq_len, cache_size, dtype, use_cache, attn_kw_args = (
        2,
        10,
        5,
        jnp.bfloat16,
        False,
        ATTN_KW | GQA_KW,
    )
    test_attention(batch_size, seq_len, cache_size, dtype, use_cache, attn_kw_args)

    rng = nnx.Rngs(0)
    head_dim = 5
    x = jax.random.normal(rng(), (3, 4, head_dim))
    norm_nn = _layers.RMSNorm()
    variables = norm_nn.init(rng(), x)
    out_nn = norm_nn.apply(variables, x)

    norm = nnx.RMSNorm(head_dim, rngs=rng)
    out = norm(x)
