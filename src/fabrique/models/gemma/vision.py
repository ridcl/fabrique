import dataclasses
import functools
from typing import cast, Optional

import einops
import jax
from flax import nnx

from gemma.multimodal.vision import VisionInitEmbeddings
from gemma.gm.vision import _preprocess

from fabrique.models.gemma import vision_utils


class VisionExit(nnx.Module):
    """The vision exit layer.

    Possibly downsample the soft tokens to a required output length.

    Attributes:
        output_length: The embed will be spatially avg-pooled to this output length.
    """

    def __init__(self, output_length: int = 256):
        self.output_length = output_length

    def __call__(
        self, x: jax.Array  # Float["B INPUT_LENGTH D"]
    ) -> jax.Array:  # Float["B OUTPUT_LENGTH D"]:
        cur_length = x.shape[1]
        if cur_length == self.output_length:
            return x
        cur_width = int(cur_length**0.5)
        assert cur_width**2 == cur_length
        output_width = int(self.output_length**0.5)
        assert (
            output_width**2 == self.output_length
        ), f"Cannot pool {x.shape=} to {self.output_length}=!"
        x = einops.rearrange(x, " b (h w) d -> b h w d", h=cur_width, w=cur_width)
        assert not cur_width % output_width, f"{cur_width=} {output_width=}"
        window = cur_width // output_width
        window_shape = (window, window)
        x = nnx.avg_pool(x, window_shape=window_shape, strides=window_shape)
        return einops.rearrange(x, "b h w d -> b (h w) d")


class SigLiPFromPatches(nnx.Module):
    """SigLIP vision encoder forward pass from PatchifiedMedia."""

    def __init__(
        self,
        siglip_encoder: Optional[vision_utils.ViTModel] = None,
        siglip_exit: Optional[VisionExit] = None,
        num_mm_tokens_per_image_prepool: int = 4096,
        num_mm_tokens_per_image: int = 256,
        image_height: int = 896,
        image_width: int = 896,
        image_channels: int = 3,
        apply_stop_gradient: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        self.siglip_encoder = siglip_encoder or vision_utils.ViTModel(rngs=rngs)
        self.siglip_exit = siglip_exit or VisionExit()
        self.num_mm_tokens_per_image_prepool = num_mm_tokens_per_image_prepool
        self.num_mm_tokens_per_image = num_mm_tokens_per_image
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.apply_stop_gradient = apply_stop_gradient


    # @functools.partial(nnx.jit, static_argnames=("self", "is_training"))
    def __call__(
        self,
        *,
        patches: jax.Array,   # Float["B N P D"]
        is_training: bool,
    ) -> jax.Array:    # Float["B N siglip_embed_dim"]:
        # chex.assert_rank(patches, 4)
        batch_size, num_frames, num_patches, num_channels = patches.shape
        num_patches_one_side = self.image_height // self.siglip_encoder.patch_size[0]
        # chex.assert_equal(num_channels, 3 * self.siglip_encoder.patch_size[0] ** 2)
        # chex.assert_equal(num_patches, num_patches_one_side**2)
        flattened_images = einops.rearrange(
            patches,
            "b n (h w) c -> (b n) h w c",
            h=num_patches_one_side,
            w=num_patches_one_side,
            c=num_channels,
        )
        flattened_images = einops.rearrange(
            flattened_images,
            "b h w (p q c) -> b (h p) (w q) c",
            h=num_patches_one_side,
            w=num_patches_one_side,
            p=self.siglip_encoder.patch_size[0],
            q=self.siglip_encoder.patch_size[0],
            c=3,
        )

        soft_tokens = self.siglip_encoder(flattened_images)

        if self.num_mm_tokens_per_image_prepool != self.num_mm_tokens_per_image:
            soft_tokens = self.siglip_exit(soft_tokens)
            assert soft_tokens.shape[-2] == self.siglip_exit.output_length

        soft_tokens = einops.rearrange(
            soft_tokens, "(b n) ... -> b n ...", b=batch_size, n=num_frames
        )
        soft_tokens = cast(jax.Array, soft_tokens)

        if self.apply_stop_gradient:
            soft_tokens = jax.lax.stop_gradient(soft_tokens)
        return soft_tokens


    def patchify_images(
        self,
        images: jax.Array,   # Float["*B H W C"]
    ) -> jax.Array:    # Float["*B P D"]:
        """Patchify images.

        Args:
        images: The images to patchify.

        Returns:
        The patches of the images of shape (*batch, num_patches, patch_size *
        patch_size * channels)
        """
        *batch_dims, _, _, _ = images.shape
        images = einops.rearrange(images, "... h w c -> (...) h w c")

        preprocess_fn = functools.partial(
            _preprocess.pre_process_image,
            image_shape=(self.image_height, self.image_width, self.image_channels),
        )
        images = jax.vmap(preprocess_fn)(images)

        patches = _preprocess.patchify_images(
            images,
            patch_size=self.siglip_encoder.patch_size,
        )
        patches = patches.reshape((*batch_dims,) + patches.shape[1:])
        return patches



def test_siglip_from_patches():
    # test imports

    import jax.numpy as jnp
    from gemma.multimodal import vision as v
    from fabrique.loading import update_module_from_params
    from fabrique.loading import LoadRule as R


    batch_size = 2
    n_images = 1
    h, w = 896, 896
    in_channels = 3
    depth = 6
    input_dim = 24
    mlp_dim = 4 * input_dim
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
    out = siglip(patches=patches, is_training=False)
    assert jnp.allclose(out_nn, out, atol=1e-2)