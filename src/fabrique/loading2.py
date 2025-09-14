import re
import logging
from typing import Callable
from dataclasses import dataclass
import jax
from flax import nnx
from jax.sharding import NamedSharding
from fabrique.utils import set_by_path, get_by_path
from fabrique.loading import RuleIgnore, IGNORE


logger = logging.getLogger("fabrique")


def _pattern_to_regexp(pat: str) -> str:
    pat = pat.replace(".", "\\.")
    pat = pat.replace("{i}", "(?P<i>\\d+)")
    pat = pat.replace("{j}", "(?P<j>\\d+)")
    pat = pat.replace("{k}", "(?P<k>\\d+)")
    pat = pat.replace("{n}", "(?P<n>\\d+)")
    pat = "^" + pat + "$"
    return pat


def convert_path(path: str, in_pattern: str, out_pattern: str | RuleIgnore):
    """
    Convert path according to input and output patterns. Example:

    ```
    path = "layer_3.attn.attn_vec_einsum.w"
    in_pat = "layer_{n}.attn.attn_vec_einsum.w"
    out_pat = "blocks.{n}.attn.attn_vec_einsum.kernel"

    convert_path(path, in_pat, out_pat)
    # ==> 'blocks.{n}.attn.attn_vec_einsum.kernel'
    ```

    """
    pat_re = _pattern_to_regexp(in_pattern)
    if m := re.match(pat_re, path):
        if isinstance(out_pattern, RuleIgnore):
            return IGNORE
        return out_pattern.format(**m.groupdict())


def update_module_from_params(module: nnx.Module, rules: tuple[str, str], params: dict, *, mesh: jax.sharding.Mesh | None = None):
    """
    Update Flax NNX module from a Flax Linen param tree
    """

    def keys_to_path(keys):
        return ".".join(key.key for key in keys)

    state = nnx.state(module)                   # the model's state, a pure pytree
    pspecs = nnx.get_partition_spec(state)      # strip out the annotations from state
    for param_keys, val in jax.tree.flatten_with_path(params)[0]:
        param_path = keys_to_path(param_keys)
        for in_pattern, out_pattern, converter in rules:
            module_path = convert_path(param_path, in_pattern, out_pattern)
            if not module_path:
                continue
            # path is rules points to Param, but here we work with Array values
            module_path += ".value"
            if converter:
                val = converter(val)
            if mesh:
                pspec = get_by_path(pspecs.raw_mapping, module_path)
                val = jax.lax.with_sharding_constraint(val, NamedSharding(mesh, pspec))
            set_by_path(module, module_path, val)


@dataclass
class LoadRule:
    in_pattern: str
    out_pattern: str | RuleIgnore
    converter: Callable | None = None

    def __iter__(self):
        yield self.in_pattern
        yield self.out_pattern
        yield self.converter



R = LoadRule

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


def main():
    import numpy as np
    import jax.numpy as jnp
    from gemma import gm
    from fabrique.models.gemma.modeling import Transformer
    # in_path = "layer_3.attn.q_einsum.w"
    # in_pattern = "layer_{n}.attn.q_einsum.w"
    # out_pattern = "blocks.{n}.attn.q_einsum.kernel.value"

    # out_path = convert_path(in_path, in_pattern, out_pattern)


    config = gm.nn.Gemma3_4B.config
    ckpt = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    params = gm.ckpts.load_params(ckpt)
    module = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    )

    rules = RULES
    mesh = jax.sharding.Mesh(devices=np.array(jax.devices())[None, :], axis_names=("data", "model"))
    update_module_from_params(module, rules, params, mesh=mesh)

    param_keys, val = jax.tree.flatten_with_path(params)[0][8]