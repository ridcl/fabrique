import einops
import jax
from flax import nnx

from gemma.multimodal.vision import VisionInitEmbeddings


class VisionExit(nnx.Module):
    """The vision exit layer.

    Possibly downsample the soft tokens to a required output length.

    Attributes:
        output_length: The embed will be spatially avg-pooled to this output length.
    """

    output_length: int = 256

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

    siglip_encoder: vision_utils.ViTModel = dataclasses.field(
        default_factory=vision_utils.ViTModel
    )
    siglip_exit: VisionExit = dataclasses.field(default_factory=VisionExit)
    num_mm_tokens_per_image_prepool: int = 4096
    num_mm_tokens_per_image: int = 256
    image_height: int = 896
    image_width: int = 896
    image_channels: int = 3
    apply_stop_gradient: bool = True

    @functools.partial(nn.jit, static_argnames=("self", "is_training"))
    @nn.compact
    def __call__(
        self,
        *,
        patches: Float["B N P D"],
        is_training: bool,
    ) -> Float["B N siglip_embed_dim"]:
        chex.assert_rank(patches, 4)
        batch_size, num_frames, num_patches, num_channels = patches.shape
        num_patches_one_side = self.image_height // self.siglip_encoder.patch_size[0]
        chex.assert_equal(num_channels, 3 * self.siglip_encoder.patch_size[0] ** 2)
        chex.assert_equal(num_patches, num_patches_one_side**2)
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

    # Could use `@_jax_utils.flatten_unflatten_batch_dim()` to simplify re-shape
    def patchify_images(self, images: Float["*B H W C"]) -> Float["*B P D"]:
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
