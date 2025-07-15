import jax.numpy as jnp

from fabrique.loading import IGNORE
from fabrique.loading import ConversionRule as R

# fmt: off
RULES = [
    R("model.embed_tokens.weight", "tok_embeddings.embedding"),
    R("model.layers.{n}.self_attn.rotary_emb.inv_freq", IGNORE),
    R("model.layers.{n}.self_attn.q_proj.weight", "layers[{n}].attention.wq.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.k_proj.weight", "layers[{n}].attention.wk.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.v_proj.weight", "layers[{n}].attention.wv.kernel", jnp.transpose),
    R("model.layers.{n}.self_attn.o_proj.weight", "layers[{n}].attention.wo.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.gate_proj.weight", "layers[{n}].feed_forward.w1.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.down_proj.weight", "layers[{n}].feed_forward.w2.kernel", jnp.transpose),
    R("model.layers.{n}.mlp.up_proj.weight", "layers[{n}].feed_forward.w3.kernel", jnp.transpose),
    R("model.layers.{n}.input_layernorm.weight", "layers[{n}].attention_norm.weight"),
    R("model.layers.{n}.post_attention_layernorm.weight", "layers[{n}].ffn_norm.weight"),
    R("lm_head.weight", "output.kernel", jnp.transpose),
    R("model.norm.weight", "norm.weight"),
]

# from: https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-3-instruct.jinja
CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
{% endif %}
""".strip()
# fmt: on
