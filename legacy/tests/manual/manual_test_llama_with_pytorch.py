import flax.linen as nn
import jax
import jax.numpy as jnp
import torch
from multimethod import multimethod
from transformers.models.llama import modeling_flax_llama as hf
from transformers.models.llama import modeling_llama as pt

from fabrique.models.llama.modeling import (  # precompute_freqs_cis,; view_as_complex,; view_as_real,
    Attention,
    FeedForward,
    ModelArgs,
    RMSNorm,
    Transformer,
    TransformerBlock,
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
    repeat_kv,
)
from fabrique.utils import update_tree
from tests.manual.torch_utils import fill_pytorch, jax2pt, pt2jax

# reduces memory per layer from 2.2Gb to 0.8Gb
MODEL_ARGS = ModelArgs(max_batch_size=1, max_seq_len=512)

# args = ModelArgs(
#             dim=config["hidden_size"],
#             n_layers=config["num_hidden_layers"],
#             n_heads=config["num_attention_heads"],
#             n_kv_heads=config["num_key_value_heads"],
#             vocab_size=config["vocab_size"],
#             # multiple_of=
#             # ffn_dim_multiplier=
#             norm_eps=config["rms_norm_eps"],
#             # max_batch_size=
#             max_seq_len=config["max_position_embeddings"],
#         )


def args2config(args: ModelArgs):
    from transformers.models.llama.modeling_llama import LlamaConfig

    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        # hidden_act="silu",
        max_position_embeddings=args.max_seq_len,
        # initializer_range=0.02,
        rms_norm_eps=args.norm_eps,
        use_cache=args.use_cache,
        # pad_token_id=None,
        # bos_token_id=1,
        # eos_token_id=2,
        # pretraining_tp=1,
        # tie_word_embeddings=False,
        # rope_theta=10000.0,
        # rope_scaling=None,
        # attention_bias=False,
        # attention_dropout=0.0,
        # mlp_bias=False,
    )


@multimethod
def fill_pt(pt_attn: pt.LlamaAttention, params: dict):
    pt_attn.q_proj.weight.data = jax2pt(params["wq"]["kernel"].T)
    pt_attn.k_proj.weight.data = jax2pt(params["wk"]["kernel"].T)
    pt_attn.v_proj.weight.data = jax2pt(params["wv"]["kernel"].T)
    pt_attn.o_proj.weight.data = jax2pt(params["wo"]["kernel"].T)


@multimethod
def fill_pt(pt_ff: pt.LlamaMLP, params: dict):
    pt_ff.gate_proj.weight.data = jax2pt(params["w1"]["kernel"].T)
    pt_ff.down_proj.weight.data = jax2pt(params["w2"]["kernel"].T)
    pt_ff.up_proj.weight.data = jax2pt(params["w3"]["kernel"].T)


@multimethod
def fill_pt(norm: pt.LlamaRMSNorm, params: dict):
    norm.weight.data = jax2pt(params["weight"])


def test_rotary_embeddings():
    args = MODEL_ARGS
    rng = jax.random.PRNGKey(71)
    rng_q, rng_k = jax.random.split(rng, 2)

    batch_dim, seq_len, n_heads, head_dim = 1, 5, args.n_heads, args.dim // args.n_heads
    xq = jax.random.normal(rng_q, (batch_dim, seq_len, n_heads, head_dim))
    xk = jax.random.normal(rng_k, (batch_dim, seq_len, n_heads, head_dim))

    # jax
    sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
    out = apply_rotary_pos_emb(xq, xk, sincos, start_pos=0)

    # torch
    pt_rot_emb = pt.LlamaRotaryEmbedding(args.dim // args.n_heads, args.max_seq_len)
    pt_cos, pt_sin = pt_rot_emb(jax2pt(xq), torch.arange(seq_len).reshape(1, -1))
    # note: for torch, we move dimensions
    # (bs, seq_len, n_heads, head_dim) -> (bs, n_heads, seq_len, head_dim)
    pt_xq, pt_xk = jax2pt(xq).transpose(1, 2), jax2pt(xk).transpose(1, 2)
    pt_out = pt.apply_rotary_pos_emb(pt_xq, pt_xk, pt_cos, pt_sin)

    # test that sin and cos are calculated the same way
    assert jnp.allclose(pt2jax(pt_cos)[0, :, :], sincos[:seq_len, head_dim:])
    assert jnp.allclose(pt2jax(pt_sin)[0, :, :], sincos[:seq_len, :head_dim])
    # test that values are the same after RoPE
    # note: we move dimensions back
    assert jnp.allclose(out[0], pt2jax(pt_out[0].transpose(1, 2)), atol=1e-5)

    # test with start_pos > 0
    start_pos = 3
    out = apply_rotary_pos_emb(xq, xk, sincos, start_pos=start_pos)
    pt_cos, pt_sin = pt_rot_emb(
        jax2pt(xq), torch.arange(start_pos, start_pos + seq_len).reshape(1, -1)
    )
    pt_xq, pt_xk = jax2pt(xq).transpose(1, 2), jax2pt(xk).transpose(1, 2)
    pt_out = pt.apply_rotary_pos_emb(pt_xq, pt_xk, pt_cos, pt_sin)
    assert jnp.allclose(out[0], pt2jax(pt_out[0].transpose(1, 2)), atol=1e-5)


def test_repeat_kv():
    rng = jax.random.PRNGKey(134)
    x = jax.random.normal(rng, (5, 4, 3, 2))
    out = repeat_kv(x, 6)
    pt_x = jax2pt(x).transpose(1, 2)
    pt_out = pt.repeat_kv(pt_x, 6)
    assert jnp.allclose(out, pt2jax(pt_out.transpose(1, 2)))


# def test_attention():
#     args = MODEL_ARGS
#     args.dtype, args.param_dtype = jnp.float32, jnp.float32
#     args.use_cache = True
#     bsz, seqlen, dim = (1, 5, args.dim)
#     rng = jax.random.PRNGKey(925)
#     x = jax.random.normal(rng, (bsz, seqlen, dim))

#     sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
#     attn = Attention(args)
#     variables = attn.init(rng, x, 0, sincos, None)
#     attn = attn.bind(variables)

#     pt_config = args2config(args)
#     pt_attn = pt.LlamaAttention(pt_config)
#     params = variables["params"]
#     fill_pt(pt_attn, params)

#     pt_x = jax2pt(x)

#     start_pos = 0
#     position_ids = torch.arange(start_pos, start_pos + seqlen).unsqueeze(0)
#     out, v_upd = attn.apply(variables, x, start_pos, sincos, None, mutable=["cache"])
#     pt_out, _, pt_cache = pt_attn(pt_x, position_ids=position_ids)
#     assert jnp.allclose(pt2jax(pt_out), out, atol=5e-2)

#     start_pos = 3
#     position_ids = torch.arange(start_pos, start_pos + seqlen).unsqueeze(0)
#     update_tree(variables, v_upd)
#     out, v_upd = attn.apply(variables, x, start_pos, sincos, None, mutable=["cache"])
#     pt_out, _, pt_cache = pt_attn(pt_x, position_ids=position_ids)
#     assert jnp.allclose(pt2jax(pt_out), out, atol=5e-2)


def test_attention_vs_hf():
    args = MODEL_ARGS
    args.dtype, args.param_dtype = jnp.float32, jnp.float32
    args.use_cache = True
    bsz, seq_len, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seq_len, dim))

    sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
    full_causal_mask = nn.make_causal_mask(
        jnp.ones((1, args.max_seq_len), dtype="bool"), dtype="bool"
    )
    attn = Attention(args)
    variables = attn.init(rng, x, 0, sincos, full_causal_mask)
    attn = attn.bind(variables)

    hf_config = args2config(args)
    hf_attn = hf.FlaxLlamaAttention(hf_config)
    hf_attn_mask = jnp.ones((bsz, args.max_seq_len))
    position_ids = jnp.arange(0, args.max_seq_len).reshape(1, -1)
    # initialize with max length to get full-size cache
    hf_variables = hf_attn.init(
        rng,
        jax.random.normal(rng, (bsz, args.max_seq_len, dim)),
        hf_attn_mask,
        position_ids,
        init_cache=True,
    )

    hf_variables["params"]["q_proj"] = variables["params"]["wq"]
    hf_variables["params"]["k_proj"] = variables["params"]["wk"]
    hf_variables["params"]["v_proj"] = variables["params"]["wv"]
    hf_variables["params"]["o_proj"] = variables["params"]["wo"]

    def test_from_position(start_pos: int):
        position_ids = jnp.arange(start_pos, start_pos + seq_len).reshape(1, -1)
        # hf_variables["cache"]["cache_index"] = jnp.array(start_pos, dtype=jnp.int32)
        out, v_upd = attn.apply(
            variables, x, start_pos, sincos, full_causal_mask, mutable=["cache"]
        )
        (hf_out,), hf_v_upd = hf_attn.apply(
            hf_variables,
            x,
            hf_attn_mask,
            position_ids,
            init_cache=True,
            mutable=["cache"],
        )
        assert jnp.allclose(out, hf_out)
        assert jnp.allclose(
            v_upd["cache"]["cache_k"][:, :seq_len, :, :],
            hf_v_upd["cache"]["cached_key"][:, :seq_len, :, :],
        )
        assert jnp.allclose(
            v_upd["cache"]["cache_v"][:, :seq_len, :, :],
            hf_v_upd["cache"]["cached_value"][:, :seq_len, :, :],
        )
        update_tree(variables, v_upd)
        update_tree(hf_variables, hf_v_upd)

    test_from_position(0)
    test_from_position(5)


def test_feedforward():
    args = MODEL_ARGS
    args.dtype, args.param_dtype = jnp.float32, jnp.float32
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))

    ff = FeedForward(
        args.dim,
        args.dim // 2,
        args.multiple_of,
        args.ffn_dim_multiplier,
        dtype=args.dtype,
    )
    variables = ff.init(rng, x)
    out = ff.apply(variables, x)

    pt_config = args2config(args)
    pt_ff = pt.LlamaMLP(pt_config)
    params = variables["params"]
    fill_pt(pt_ff, params)

    pt_x = jax2pt(x)
    pt_out = pt_ff(pt_x)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)


def test_transformerblock():
    args = MODEL_ARGS
    args.dtype, args.param_dtype = jnp.float32, jnp.float32
    bsz, seq_len, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seq_len, dim))
    sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
    full_causal_mask = nn.make_causal_mask(
        jnp.ones((1, args.max_seq_len), dtype="bool"), dtype="bool"
    )
    block = TransformerBlock(0, args)
    variables = block.init(rng, x, 0, sincos, full_causal_mask)

    hf_config = args2config(args)
    hf_block = hf.FlaxLlamaDecoderLayer(hf_config)
    hf_attn_mask = jnp.ones((bsz, args.max_seq_len))
    position_ids = jnp.arange(0, args.max_seq_len).reshape(1, -1)
    hf_variables = hf_block.init(
        rng,
        jax.random.normal(rng, (bsz, args.max_seq_len, dim)),
        hf_attn_mask,
        position_ids,
        init_cache=True,
    )
    # pt_block = pt.LlamaDecoderLayer(pt_config, 0)
    # fill_pytorch(pt_block, variables["params"])

    hf_variables["params"]["self_attn"]["q_proj"] = variables["params"]["attention"][
        "wq"
    ]
    hf_variables["params"]["self_attn"]["k_proj"] = variables["params"]["attention"][
        "wk"
    ]
    hf_variables["params"]["self_attn"]["v_proj"] = variables["params"]["attention"][
        "wv"
    ]
    hf_variables["params"]["self_attn"]["o_proj"] = variables["params"]["attention"][
        "wo"
    ]

    hf_variables["params"]["mlp"]["gate_proj"] = variables["params"]["feed_forward"][
        "w1"
    ]
    hf_variables["params"]["mlp"]["down_proj"] = variables["params"]["feed_forward"][
        "w2"
    ]
    hf_variables["params"]["mlp"]["up_proj"] = variables["params"]["feed_forward"]["w3"]

    hf_variables["params"]["input_layernorm"]["weight"] = variables["params"][
        "attention_norm"
    ]["weight"]
    hf_variables["params"]["post_attention_layernorm"]["weight"] = variables["params"][
        "ffn_norm"
    ]["weight"]

    start_pos = 3
    out, v_upd = block.apply(
        variables, x, start_pos, sincos, full_causal_mask, mutable=["cache"]
    )
    (hf_out,), hf_v_upd = hf_block.apply(
        hf_variables,
        x,
        attention_mask=jnp.ones((bsz, args.max_seq_len)),
        position_ids=jnp.arange(start_pos, start_pos + seq_len).reshape(1, -1),
        mutable=("cache",),
    )
    assert jnp.allclose(hf_out, out, atol=5e-2)


def test_transformer():
    from fabrique.llama import Llama

    # testing on Llama 2 since Llama 3 doesn't load in HF Flax implementation
    model_id = "meta-llama/Llama-2-7b-hf"
    kwargs = {
        "max_seq_len": 512,
        "max_batch_size": 1,
        "use_cache": True,
        "dtype": jnp.float32,
        "param_dtype": jnp.float32,
    }
    llama = Llama.from_pretrained(model_id, **kwargs)
    tokenizer, model, variables = llama.tokenizer, llama.model, llama.variables
    tokens = tokenizer.encode("once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)

    # rng = jax.random.PRNGKey(925)
    # model = Transformer(args)
    # variables = model.init(rng, tokens, 0)
    # model = model.bind(variables)
    out, _v_upd = model.apply(variables, tokens, 0, mutable=["cache"])

    hf_kwargs = {
        "max_position_embeddings": kwargs["max_seq_len"],
        # "max_position_embeddings": 4096,
        "use_cache": model.args.use_cache,
    }
    # pt_model = pt.LlamaForCausalLM.from_pretrained(model_id, **pt_kwargs)
    # fill_pytorch(pt_model, variables["params"])
    # pt_out = pt_model(pt_tokens).logits
    # assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)

    from transformers.modeling_flax_outputs import FlaxMaskedLMOutput

    hf_model = hf.FlaxLlamaForCausalLM.from_pretrained(
        model_id, from_pt=True, **hf_kwargs
    )
    hf_attn_mask = jnp.ones_like(tokens)
    hf_position_ids = jnp.arange(tokens.shape[-1]).reshape(1, -1)
    hf_out = hf_model(tokens, hf_attn_mask, hf_position_ids)
    assert jnp.allclose(out, hf_out[0], atol=5e-2)

    start_pos = tokens.shape[-1]
    hf_position_ids = jnp.arange(start_pos, start_pos + start_pos).reshape(1, -1)
    hf_out = hf_model(tokens, hf_attn_mask, hf_position_ids)

    # TODO: update cache from previous call
    out, _v_upd = model.apply(variables, tokens, start_pos, mutable=["cache"])

    # --------------------------------------------------------------------
    import os

    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    model_id = "meta-llama/Llama-2-7b-hf"
    model_dir = snapshot_download(model_id, repo_type="model")
    tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_file)
    tokens = tokenizer.encode("once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)

    hf_model = hf.FlaxLlamaForCausalLM.from_pretrained(model_id, from_pt=True)
    hf_model.params = hf_model.to_fp32(hf_model.params)
    hf_out = hf_model.generate(tokens, max_length=512)
    tokenizer.decode_batch(hf_out.sequences)
