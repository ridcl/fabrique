import jax.numpy as jnp

from fabrique.loading import IGNORE
from fabrique.loading import ConversionRule as R

# fmt: off
RULES = [
    R("model.embed_tokens.weight", "tok_embeddings.embedding"),
    R("model.layers.{n}.self_attn.rotary_emb.inv_freq", IGNORE),
    R("model.layers.{n}.self_attn.qkv_proj.weight", "layers[{n}].attention.wqkv.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.o_proj.weight", "layers[{n}].attention.wo.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.gate_up_proj.weight", "layers[{n}].feed_forward.w1.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.down_proj.weight", "layers[{n}].feed_forward.w2.kernel", jnp.transpose),
    R("model.layers.{n}.input_layernorm.weight", "layers[{n}].attention_norm.weight"),
    R("model.layers.{n}.post_attention_layernorm.weight", "layers[{n}].ffn_norm.weight"),
    R("lm_head.weight", "output.kernel", jnp.transpose),
    R("model.norm.weight", "norm.weight"),
]

CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|' + message['role'] + '|>\n' + message['content'] | trim + '<|end|>' + '\n' }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|assistant|>\n' }}
{% endif %}
""".strip()
# fmt: on
