# based on:
# https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_transformer.py
from typing import Any, ClassVar

import einops
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import bridge
from gemma.gm.nn import _config, _modules, _transformer
from gemma.gm.nn._transformer import ModelInfo
from gemma.gm.utils import _types
from gemma.gm.vision import _token_utils
from gemma.multimodal import vision as gemma_vision

from fabrique.models.gemma.layers import GemmaRMSNorm
from fabrique.models.gemma.modules import Block, Embedder
from fabrique.models.gemma.vision import SigLiPFromPatches


class Transformer(nnx.Module):
    """Base transformer class.

    Attributes:
        return_last_only: If `True`, only compute and return the last token.
        Otherwise, return all logits. Default to `False`
        dtype: The parameter dtype. Default to `jnp.bfloat16`.
    """

    # Model info to specifiy the tokenizer version and default checkpoint.
    INFO: ClassVar[ModelInfo] = ModelInfo()

    def __init__(
        self,
        config: _config.TransformerConfig,
        *,
        param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.return_last_only: bool | None = None
        self.dtype: jnp.dtype = jnp.bfloat16
        self.config = config

        self.embedder = Embedder(
            vocab_size=self.config.num_embed,
            embed_dim=self.config.embed_dim,
            vision_proj_dim=(
                self.config.vision_encoder.siglip_encoder.width
                if self.config.vision_encoder
                else None
            ),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.blocks = nnx.List([
            Block(
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_kv_heads,
                embed_dim=self.config.embed_dim,
                head_dim=self.config.head_dim,
                hidden_dim=self.config.hidden_dim,
                sliding_window_size=self.config.sliding_window_size,
                use_post_attn_norm=self.config.use_post_attn_norm,
                use_post_ffw_norm=self.config.use_post_ffw_norm,
                attn_logits_soft_cap=self.config.attn_logits_soft_cap,
                attn_type=attn_type,
                query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
                transpose_gating_einsum=self.config.transpose_gating_einsum,
                use_qk_norm=self.config.use_qk_norm,
                rope_base_frequency=(
                    self.config.local_base_frequency
                    if attn_type == _modules.AttentionType.LOCAL_SLIDING
                    else self.config.global_base_frequency
                ),
                rope_scale_factor=(
                    self.config.local_scale_factor
                    if attn_type == _modules.AttentionType.LOCAL_SLIDING
                    else self.config.global_scale_factor
                ),
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for attn_type in self.config.attention_types
        ])
        self.final_norm = GemmaRMSNorm(
            config.embed_dim, param_dtype=param_dtype, rngs=rngs
        )
        # NOTE: TransformerConfig comes with original Linen vision encoder,
        # but we replace it with our NNX implementation (or None, if text-only model)
        if self.config.vision_encoder is not None:
            self.vision_encoder = SigLiPFromPatches(rngs=rngs)
        else:
            self.vision_encoder = None

    # def _wrap_and_init_vision_encoder(
    #     self, vision_encoder: gemma_vision.SigLiPFromPatches | None, rngs: nnx.Rngs
    # ) -> bridge.ToNNX:
    #     if vision_encoder is None:
    #         return None
    #     wrapped = bridge.ToNNX(vision_encoder, rngs=rngs)

    #     num_patches_one_side = (
    #         vision_encoder.image_height // vision_encoder.siglip_encoder.patch_size[0]
    #     )
    #     num_channels = 3 * vision_encoder.siglip_encoder.patch_size[0] ** 2
    #     dummy_patches = jnp.ones(
    #         (1, 1, num_patches_one_side**2, num_channels), dtype=jnp.uint8
    #     )
    #     # note: lazy_init() of vision encoder may take up to a few minutes
    #     wrapped.lazy_init(patches=dummy_patches, is_training=False)
    #     return wrapped

    def __repr__(self):
        return "Transformer[Gemma](...)"

    def __call__(
        self,
        tokens: jax.Array,  # Int['*B L']
        *,
        images: jax.Array | None = None,
        positions: jax.Array | None = None,
        cache: _config.Cache | None = None,
        attention_mask: jax.Array | None = None,
        return_last_only: bool | None = None,
        return_hidden_states: bool | None = None,
    ) -> _transformer.Output:  # Output['*B']
        """Transformer forward pass.

        You can run this forward pass two ways: with or without an attention kv
        cache.

        Args:
        tokens: input sequence of tokens, Int['*B L'].
        images: Images to feed to the vision encoder, UInt8['*B N H W C'] | UInt8['*B H W C'].
        positions: input absolute positions, Int['*B L_with_mm'].
            When provided, the positions and attention_mask should include
            the extra inserted multi-modal tokens.
        cache: Attention KV cache or None.
        attention_mask: transformer input mask, Bool['*B L_with_mm cache_length'].
            During training and pre-filling, the attention mask is `*B L L`
            When sampling (after prefilling), tokens are decoded one by one,
            so the attention mask is `*B 1 cache_length`
        return_last_only: If `True`, only compute and return the logits of the
            last input token in sequence. Useful for decoding where we don't need to
            compute logits for the whole sequence, but only for the last token.
            Otherwise, return all logits. Default to `False`.
        return_hidden_states: If `True`, return the hidden states of the model.
            Useful for developing custom models. Otherwise, return only the logits
            and the cache. Default to `False`.

        Returns:
            predicted_logits, new_cache

            predicted_logits: output logits predicted by the model
            new_cache: updated cache if the input cache is not None, None elsewhere.
        """

        return_last_only = return_last_only or self.return_last_only

        # Encode the text tokens, eventually including the vision embeddings.
        inputs = self._encode_and_get_inputs(
            tokens=tokens,
            images=images,
            positions=positions,
            attention_mask=attention_mask,
        )
        del positions, attention_mask

        x = inputs.embeddings

        old_cache = cache or {}
        new_cache = {}
        for i, block in enumerate(self.blocks):
            layer_name = f"layer_{i}"
            layer_cache, x = block(
                x,
                inputs.positions,
                old_cache.get(layer_name),
                inputs.attention_mask,
            )
            new_cache[layer_name] = (
                layer_cache  # pytype: disable=container-type-mismatch
            )

        x = self.final_norm(x)

        if return_last_only:
            last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
            x = x[jnp.arange(len(x)), last_input_token_idx, ...]
        elif images is not None:
            # Remove the MM extra tokens inserted.
            # During fine-tuning, the prompt is always masked, and the model cannot
            # generate images tokens, so the logits are meaningless anyway.
            x = _token_utils.remove_mm_logits(
                logits=x,
                tokens=tokens,
                num_tokens_per_image=self.config.vision_encoder.num_mm_tokens_per_image,  # pytype: disable=attribute-error
            )

        logits = self.embedder.decode(x)

        if self.config.final_logit_softcap is not None:
            logits /= self.config.final_logit_softcap
            logits = jnp.tanh(logits) * self.config.final_logit_softcap

        return _transformer.Output(
            logits=logits,
            cache=None if cache is None else new_cache,
            hidden_states=x if return_hidden_states else None,
        )

    def init_cache(
        self, *, batch_size: int, dtype: jnp.dtype[Any], cache_length: int
    ) -> _config.Cache:
        cache = self.config.init_cache(
            batch_size=batch_size,
            dtype=dtype,
            cache_length=cache_length,
        )
        return cache

    def _encode_and_get_inputs(
        self,
        *,
        tokens: jax.Array,
        images: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        positions: jax.Array | None = None,
    ) -> _transformer._Inputs:
        """
        Encode the text tokens, eventually including the vision embeddings.

        Args:
        tokens: Input sequence of tokens, Int['B L_no_mm'].
        images: Images to fit to he vision encoder, Bool['B L_with_mm cache_length'].
        attention_mask: Attention mask, Bool['B L_with_mm cache_length'].
        positions: Input absolute positions, Int['B L_with_mm'].
        """

        # If the model has images, we expand each `<start_of_image>` token to add
        # the image placeholder tokens.
        if images is not None:
            self._assert_support_mm()
            if len(images.shape) == 4:  # Expand optional `num_images` dimension
                images = einops.rearrange(images, "b h w c -> b 1 h w c")

        inputs = _types.Input(
            text=tokens,
            images=images,
            config=self.config.input_config,
        )
        del tokens, images

        # Encode the text tokens
        # Could this be optimized to filter out the `SOFT_TOKEN_PLACEHOLDER` ?
        # Currently, The placeholders are required so the mask, positions are
        # correctly computed.
        x = self.embedder.encode(inputs.tokens_with_mm)

        # Encode the vision tokens and merge them with the text embeddings.
        if inputs.images is not None:
            x = self._merge_mm_embeddings(
                tokens=inputs.tokens_with_mm, embeddings=x, images=inputs.images
            )
        elif self.vision_encoder is not None:
            # During initialization, call the vision encoder to ensure that the
            # params are correctly initialized.
            _ = self._encode_vision(_make_dummy_images(self.vision_encoder))

        # Note: When `positions` and `attention_mask` are explicitly provided,
        # it's the user responsibility to correctly take into account the extra
        # tokens inserted for the images.
        # This is what the `gm.text.Sampler` implementation does.
        if positions is None:
            positions = inputs.positions

        if attention_mask is None:
            attention_mask = inputs.attention_mask

        return _transformer._Inputs(
            embeddings=x,
            positions=positions,
            attention_mask=attention_mask,
            inputs_mask=inputs.inputs_mask,
        )

    def _merge_mm_embeddings(
        self,
        *,
        tokens: jax.Array,  # Int['B L'],
        embeddings: jax.Array,  # Float['B L D'],
        images: jax.Array,  # UInt8['B N H W C'],
    ) -> jax.Array:  # Float['B L D']
        """Update the embeddings to include the vision embeddings."""
        # Encode the images
        soft_embeddings = self._encode_vision(images)

        # Merge the soft tokens back with the text embeddings.
        merged_embeddings = _token_utils.merge_embeddings(
            text_embeddings=embeddings,
            vision_embeddings=soft_embeddings,
            mask=tokens == gemma_vision.TOKEN_PLACEHOLDER,
        )

        return merged_embeddings

    # images: UInt8['B N H W C']; -> Float['B N P D']
    def _encode_vision(self, images: jax.Array) -> jax.Array:
        """Encode the images into the same space as the text embeddings."""
        assert self.vision_encoder is not None
        patches = self.vision_encoder.patchify_images(images)
        soft_embeddings = self.vision_encoder(patches=patches)
        soft_embeddings = self.embedder.encode_vision(soft_embeddings)
        return soft_embeddings

    def _assert_support_mm(self) -> None:
        if self.config.vision_encoder is None:
            msg = ""
            if getattr(self, "text_only", False):
                msg = " The model was created with `text_only=True`."
            raise ValueError(
                f"The model {type(self).__name__!r} does not have vision encoder,"
                " yet images are provided." + msg
            )


def _make_dummy_images(
    vision_encoder: SigLiPFromPatches,
) -> jax.Array:  # Float['B L P D']
    """Make dummy images for initializing the vision encoder."""
    return jnp.zeros(
        (1, 1, vision_encoder.image_height, vision_encoder.image_width, 3),
        dtype=jnp.uint8,
    )
