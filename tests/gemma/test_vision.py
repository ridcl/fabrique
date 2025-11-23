import jax
import jax.numpy as jnp
from flax import nnx
from gemma.multimodal import vision as v
from gemma.multimodal import vision_utils as vu
from fabrique.loading import update_module_from_params
from fabrique.loading import LoadRule as R
from fabrique.models.gemma.vision_utils import (
    MlpBlock,
    MAPHead,
    Encoder1DBlock,
    Encoder,
    ViTModel
)
from fabrique.models.gemma.vision import SigLiPFromPatches


def test_mlp_block():
    batch_size = 2
    length = 5
    input_dim = 4
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    mlp_nn = vu.MlpBlock(block_id=0, mlp_dim=mlp_dim)
    variables = mlp_nn.init(rngs.params(), x, deterministic=True)
    mlp = MlpBlock(block_id=0, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)
    rules = [
        R("Dense_0.kernel", "linear1.kernel"),
        R("Dense_0.bias", "linear1.bias"),
        R("Dense_1.kernel", "linear2.kernel"),
        R("Dense_1.bias", "linear2.bias"),
    ]
    update_module_from_params(mlp, rules, variables["params"])

    out_nn = mlp_nn.apply(variables, x)
    out = mlp(x)
    assert jnp.all(out_nn == out)


def test_map_head():
    batch_size = 2
    length = 5
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    map_head_nn = vu.MAPHead(block_id=0, mlp_dim=mlp_dim)
    variables = map_head_nn.init(rngs.params(), x)

    map_head = MAPHead(block_id=0, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)

    rules = [
        # probe
        R("probe", "probe"),
        # attn
        R("MultiHeadDotProductAttention_0.query.kernel", "attn.query.kernel"),
        R("MultiHeadDotProductAttention_0.query.bias", "attn.query.bias"),
        R("MultiHeadDotProductAttention_0.key.kernel", "attn.key.kernel"),
        R("MultiHeadDotProductAttention_0.key.bias", "attn.key.bias"),
        R("MultiHeadDotProductAttention_0.value.kernel", "attn.value.kernel"),
        R("MultiHeadDotProductAttention_0.value.bias", "attn.value.bias"),
        R("MultiHeadDotProductAttention_0.out.kernel", "attn.out.kernel"),
        R("MultiHeadDotProductAttention_0.out.bias", "attn.out.bias"),
        # norm
        R("LayerNorm_0.scale", "norm.scale"),
        R("LayerNorm_0.bias", "norm.bias"),
        # mlp
        R("MlpBlock_0.Dense_0.kernel", "mlp.linear1.kernel"),
        R("MlpBlock_0.Dense_0.bias", "mlp.linear1.bias"),
        R("MlpBlock_0.Dense_1.kernel", "mlp.linear2.kernel"),
        R("MlpBlock_0.Dense_1.bias", "mlp.linear2.bias"),
    ]
    update_module_from_params(map_head, rules, variables["params"])

    out_nn = map_head_nn.apply(variables, x)
    out = map_head(x)
    assert jnp.all(out == out_nn)


def test_encoder_1d_block():
    batch_size = 2
    length = 5
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    block_nn = vu.Encoder1DBlock(block_id=0, mlp_dim=mlp_dim)
    variables = block_nn.init(rngs.params(), x)
    block = Encoder1DBlock(block_id=0, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)

    rules = [
        # attn
        R("MultiHeadDotProductAttention_0.query.kernel", "attn.query.kernel"),
        R("MultiHeadDotProductAttention_0.query.bias", "attn.query.bias"),
        R("MultiHeadDotProductAttention_0.key.kernel", "attn.key.kernel"),
        R("MultiHeadDotProductAttention_0.key.bias", "attn.key.bias"),
        R("MultiHeadDotProductAttention_0.value.kernel", "attn.value.kernel"),
        R("MultiHeadDotProductAttention_0.value.bias", "attn.value.bias"),
        R("MultiHeadDotProductAttention_0.out.kernel", "attn.out.kernel"),
        R("MultiHeadDotProductAttention_0.out.bias", "attn.out.bias"),
        # mlp
        R("MlpBlock_0.Dense_0.kernel", "mlp.linear1.kernel"),
        R("MlpBlock_0.Dense_0.bias", "mlp.linear1.bias"),
        R("MlpBlock_0.Dense_1.kernel", "mlp.linear2.kernel"),
        R("MlpBlock_0.Dense_1.bias", "mlp.linear2.bias"),
        # pre-attn norm
        R("LayerNorm_0.scale", "pre_attn_norm.scale"),
        R("LayerNorm_0.bias", "pre_attn_norm.bias"),
        # post-attn norm
        R("LayerNorm_1.scale", "post_attn_norm.scale"),
        R("LayerNorm_1.bias", "post_attn_norm.bias"),
    ]
    update_module_from_params(block, rules, variables["params"])

    out_nn = block_nn.apply(variables, x)
    out = block(x)
    assert jnp.all(out_nn == out)


def test_encoder():
    batch_size = 2
    length = 5
    depth = 6
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    encoder_nn = vu.Encoder(depth=depth, mlp_dim=mlp_dim)
    variables = encoder_nn.init(rngs.params(), x)
    encoder = Encoder(depth=depth, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)

    rules = [
        # block: attn
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.query.kernel", "blocks.{n}.attn.query.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.query.bias", "blocks.{n}.attn.query.bias"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.key.kernel", "blocks.{n}.attn.key.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.key.bias", "blocks.{n}.attn.key.bias"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.value.kernel", "blocks.{n}.attn.value.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.value.bias", "blocks.{n}.attn.value.bias"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.out.kernel", "blocks.{n}.attn.out.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.out.bias", "blocks.{n}.attn.out.bias"),
        # block: mlp
        R("encoderblock_{n}.MlpBlock_0.Dense_0.kernel", "blocks.{n}.mlp.linear1.kernel"),
        R("encoderblock_{n}.MlpBlock_0.Dense_0.bias", "blocks.{n}.mlp.linear1.bias"),
        R("encoderblock_{n}.MlpBlock_0.Dense_1.kernel", "blocks.{n}.mlp.linear2.kernel"),
        R("encoderblock_{n}.MlpBlock_0.Dense_1.bias", "blocks.{n}.mlp.linear2.bias"),
        # block: pre-attn norm
        R("encoderblock_{n}.LayerNorm_0.scale", "blocks.{n}.pre_attn_norm.scale"),
        R("encoderblock_{n}.LayerNorm_0.bias", "blocks.{n}.pre_attn_norm.bias"),
        # block: post-attn norm
        R("encoderblock_{n}.LayerNorm_1.scale", "blocks.{n}.post_attn_norm.scale"),
        R("encoderblock_{n}.LayerNorm_1.bias", "blocks.{n}.post_attn_norm.bias"),
        # norm
        R("encoder_norm.scale", "norm.scale"),
        R("encoder_norm.bias", "norm.bias"),
    ]
    update_module_from_params(encoder, rules, variables["params"])

    out_nn = encoder_nn.apply(variables, x)
    out = encoder(x)
    assert jnp.allclose(out_nn, out, atol=1e-3)


def test_vit_model():
    batch_size = 2
    h, w = 896, 896
    in_channels = 3
    depth = 6
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, h, w, in_channels))

    vit_nn = vu.ViTModel(depth=depth, mlp_dim=mlp_dim)
    variables = vit_nn.init(rngs.params(), x)
    vit = ViTModel(depth=depth, mlp_dim=mlp_dim, in_channels=in_channels, rngs=rngs)

    rules = [
        # encoder: block: attn
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.query.kernel", "encoder.blocks.{n}.attn.query.kernel"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.query.bias", "encoder.blocks.{n}.attn.query.bias"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.key.kernel", "encoder.blocks.{n}.attn.key.kernel"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.key.bias", "encoder.blocks.{n}.attn.key.bias"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.value.kernel", "encoder.blocks.{n}.attn.value.kernel"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.value.bias", "encoder.blocks.{n}.attn.value.bias"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.out.kernel", "encoder.blocks.{n}.attn.out.kernel"),
        R("Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.out.bias", "encoder.blocks.{n}.attn.out.bias"),
        # encoder: block: mlp
        R("Transformer.encoderblock_{n}.MlpBlock_0.Dense_0.kernel", "encoder.blocks.{n}.mlp.linear1.kernel"),
        R("Transformer.encoderblock_{n}.MlpBlock_0.Dense_0.bias", "encoder.blocks.{n}.mlp.linear1.bias"),
        R("Transformer.encoderblock_{n}.MlpBlock_0.Dense_1.kernel", "encoder.blocks.{n}.mlp.linear2.kernel"),
        R("Transformer.encoderblock_{n}.MlpBlock_0.Dense_1.bias", "encoder.blocks.{n}.mlp.linear2.bias"),
        # encoder: block: pre-attn norm
        R("Transformer.encoderblock_{n}.LayerNorm_0.scale", "encoder.blocks.{n}.pre_attn_norm.scale"),
        R("Transformer.encoderblock_{n}.LayerNorm_0.bias", "encoder.blocks.{n}.pre_attn_norm.bias"),
        # encoder: block: post-attn norm
        R("Transformer.encoderblock_{n}.LayerNorm_1.scale", "encoder.blocks.{n}.post_attn_norm.scale"),
        R("Transformer.encoderblock_{n}.LayerNorm_1.bias", "encoder.blocks.{n}.post_attn_norm.bias"),
        # encoder: norm
        R("Transformer.encoder_norm.scale", "encoder.norm.scale"),
        R("Transformer.encoder_norm.bias", "encoder.norm.bias"),

        # conv
        R("embedding.kernel", "conv.kernel"),
        R("embedding.bias", "conv.bias"),

        # pos embedding
        R("pos_embedding", "pos_embedding"),
    ]
    update_module_from_params(vit, rules, variables["params"])

    out_nn = vit_nn.apply(variables, x)
    out = vit(x)
    assert jnp.allclose(out_nn, out, atol=1e-2)


def test_siglip_from_patches():
    batch_size = 2
    n_images = 1
    h, w = 896, 896
    in_channels = 3
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    images = jax.random.normal(rngs.data(), (batch_size, n_images, h, w, in_channels))

    siglip_nn = v.SigLiPFromPatches()
    siglip = SigLiPFromPatches(rngs=rngs)

    # First, check patchify, which doesn't require variable initialization
    patches_nn = siglip_nn.patchify_images(images)
    patches = siglip.patchify_images(images)
    assert jnp.all(patches_nn == patches)

    # Now, init Linen model and copy parameters to NNX model
    variables = siglip_nn.init(rngs.params(), patches=patches_nn, is_training=False)


    rules = [
        # encoder: block: attn
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.query.kernel", "siglip_encoder.encoder.blocks.{n}.attn.query.kernel"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.query.bias", "siglip_encoder.encoder.blocks.{n}.attn.query.bias"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.key.kernel", "siglip_encoder.encoder.blocks.{n}.attn.key.kernel"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.key.bias", "siglip_encoder.encoder.blocks.{n}.attn.key.bias"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.value.kernel", "siglip_encoder.encoder.blocks.{n}.attn.value.kernel"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.value.bias", "siglip_encoder.encoder.blocks.{n}.attn.value.bias"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.out.kernel", "siglip_encoder.encoder.blocks.{n}.attn.out.kernel"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MultiHeadDotProductAttention_0.out.bias", "siglip_encoder.encoder.blocks.{n}.attn.out.bias"),
        # encoder: block: mlp
        R("siglip_encoder.Transformer.encoderblock_{n}.MlpBlock_0.Dense_0.kernel", "siglip_encoder.encoder.blocks.{n}.mlp.linear1.kernel"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MlpBlock_0.Dense_0.bias", "siglip_encoder.encoder.blocks.{n}.mlp.linear1.bias"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MlpBlock_0.Dense_1.kernel", "siglip_encoder.encoder.blocks.{n}.mlp.linear2.kernel"),
        R("siglip_encoder.Transformer.encoderblock_{n}.MlpBlock_0.Dense_1.bias", "siglip_encoder.encoder.blocks.{n}.mlp.linear2.bias"),
        # encoder: block: pre-attn norm
        R("siglip_encoder.Transformer.encoderblock_{n}.LayerNorm_0.scale", "siglip_encoder.encoder.blocks.{n}.pre_attn_norm.scale"),
        R("siglip_encoder.Transformer.encoderblock_{n}.LayerNorm_0.bias", "siglip_encoder.encoder.blocks.{n}.pre_attn_norm.bias"),
        # encoder: block: post-attn norm
        R("siglip_encoder.Transformer.encoderblock_{n}.LayerNorm_1.scale", "siglip_encoder.encoder.blocks.{n}.post_attn_norm.scale"),
        R("siglip_encoder.Transformer.encoderblock_{n}.LayerNorm_1.bias", "siglip_encoder.encoder.blocks.{n}.post_attn_norm.bias"),
        # encoder: norm
        R("siglip_encoder.Transformer.encoder_norm.scale", "siglip_encoder.encoder.norm.scale"),
        R("siglip_encoder.Transformer.encoder_norm.bias", "siglip_encoder.encoder.norm.bias"),

        # conv
        R("siglip_encoder.embedding.kernel", "siglip_encoder.conv.kernel"),
        R("siglip_encoder.embedding.bias", "siglip_encoder.conv.bias"),

        # pos embedding
        R("siglip_encoder.pos_embedding", "siglip_encoder.pos_embedding"),
    ]
    update_module_from_params(siglip, rules, variables["params"])

    out_nn = siglip_nn.apply(variables, patches=patches, is_training=False)
    out = siglip(patches=patches)
    assert jnp.allclose(out_nn, out, atol=1e-2)