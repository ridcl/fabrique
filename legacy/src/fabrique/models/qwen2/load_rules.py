import jax.numpy as jnp

from fabrique.loading import IGNORE
from fabrique.loading import ConversionRule as R

# fmt: off
RULES = [
    R("model.embed_tokens.weight", "tok_embeddings.embedding"),
    R("model.layers.{n}.self_attn.rotary_emb.inv_freq", IGNORE),
    R("model.layers.{n}.self_attn.q_proj.weight", "layers[{n}].attention.wq.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.q_proj.bias", "layers[{n}].attention.wq.bias"),

    R("model.layers.{n}.self_attn.k_proj.weight", "layers[{n}].attention.wk.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.k_proj.bias", "layers[{n}].attention.wk.bias"),

    R("model.layers.{n}.self_attn.v_proj.weight", "layers[{n}].attention.wv.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.v_proj.bias", "layers[{n}].attention.wv.bias"),

    R("model.layers.{n}.self_attn.o_proj.weight", "layers[{n}].attention.wo.kernel", jnp.transpose),

    R("model.layers.{n}.mlp.gate_proj.weight", "layers[{n}].feed_forward.w1.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.down_proj.weight", "layers[{n}].feed_forward.w2.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.up_proj.weight", "layers[{n}].feed_forward.w3.kernel", jnp.transpose),
    R("model.layers.{n}.input_layernorm.weight", "layers[{n}].attention_norm.weight"),
    R("model.layers.{n}.post_attention_layernorm.weight", "layers[{n}].ffn_norm.weight"),
    R("lm_head.weight", "output.kernel", jnp.transpose),
    R("model.norm.weight", "norm.weight"),
]

# Qwen2's config on HF includes chat_template, so there's no need to specify it separately
# see https://huggingface.co/Qwen/Qwen2-7B/blob/main/tokenizer_config.json#L31
CHAT_TEMPLATE = None
# fmt: on
