from fabrique.loading import LoadRule as R

# fmt: off
RULES = [
    # embedder
    R("embedder.input_embedding", "embedder.input_embedding_table"),
    R("embedder.mm_soft_embedding_norm.scale", "embedder.mm_soft_embedding_norm.scale"),
    R("embedder.mm_input_projection.w", "embedder.mm_input_projection.kernel"),

    # vision_encoder
    R("vision_encoder.siglip_encoder.Transformer.encoder_norm.bias", "vision_encoder.siglip_encoder.Transformer.encoder_norm.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoder_norm.scale", "vision_encoder.siglip_encoder.Transformer.encoder_norm.scale"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.LayerNorm_{j}.bias", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.LayerNorm_{j}.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.LayerNorm_{j}.scale", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.LayerNorm_{j}.scale"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MlpBlock_{j}.Dense_{k}.bias", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MlpBlock_{j}.Dense_{k}.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MlpBlock_{j}.Dense_{k}.kernel", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MlpBlock_{j}.Dense_{k}.kernel"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.query.bias", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.query.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.query.kernel", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.query.kernel"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.key.bias", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.key.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.key.kernel", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.key.kernel"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.value.bias", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.value.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.value.kernel", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.value.kernel"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.out.bias", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.out.bias"),
    R("vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.out.kernel", "vision_encoder.siglip_encoder.Transformer.encoderblock_{i}.MultiHeadDotProductAttention_{j}.out.kernel"),
    R("vision_encoder.siglip_encoder.embedding.bias", "vision_encoder.siglip_encoder.embedding.bias"),
    R("vision_encoder.siglip_encoder.embedding.kernel", "vision_encoder.siglip_encoder.embedding.kernel"),
    R("vision_encoder.siglip_encoder.pos_embedding", "vision_encoder.siglip_encoder.pos_embedding"),

    ## blocks
    # attn
    R("layer_{n}.attn.attn_vec_einsum.w", "blocks.{n}.attn.attn_vec_einsum.kernel"),
    R("layer_{n}.attn.qkv_einsum.w", "blocks.{n}.attn.qkv_einsum.kernel"),
    R("layer_{n}.attn.q_einsum.w", "blocks.{n}.attn.q_einsum.kernel"),
    R("layer_{n}.attn.kv_einsum.w", "blocks.{n}.attn.kv_einsum.kernel"),
    R("layer_{n}.attn._query_norm.scale", "blocks.{n}.attn._query_norm.scale"),
    R("layer_{n}.attn._key_norm.scale", "blocks.{n}.attn._key_norm.scale"),
    # ffw
    R("layer_{n}.mlp.gating_einsum", "blocks.{n}.mlp.gating.kernel"),
    R("layer_{n}.mlp.linear", "blocks.{n}.mlp.linear.kernel"),
    # norms
    R("layer_{n}.pre_attention_norm.scale", "blocks.{n}.pre_attention_norm.scale"),
    R("layer_{n}.post_attention_norm.scale", "blocks.{n}.post_attention_norm.scale"),
    R("layer_{n}.pre_ffw_norm.scale", "blocks.{n}.pre_ffw_norm.scale"),
    R("layer_{n}.post_ffw_norm.scale", "blocks.{n}.post_ffw_norm.scale"),

    # norms
    R("final_norm.scale", "final_norm.scale"),
]

# from: https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-3-instruct.jinja
CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] | trim + '\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}

    {% if (message['role'] == 'assistant') %}
        {% set role = 'model' %}
    {% else %}
        {% set role = message['role'] %}
    {% endif %}

    {{ '<start_of_turn>' + role + '\n' + content | trim + '<end_of_turn>\n' }}
{% endfor %}

{% if add_generation_prompt %}
    {{'<start_of_turn>model\n'}}
{% endif %}
""".strip()
# fmt: on
