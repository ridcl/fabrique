from functools import partial

import einops
import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import nnx, struct
from gemma.gm.data import _functional
from gemma.gm.text._prefill import _make_full_attention_mask
from gemma.gm.utils import _types
from PIL import Image

from fabrique.loading import load_model
from fabrique.tokenizer_utils import encode_batch

LayerCache = dict[str, jax.Array]  # gemma.gm.nn._modules.LayerCache
Cache = dict[str, LayerCache]  # gemma.gm.nn._config.Cache


def sample_token(
    rng, logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50
):
    """
    Sample next token using provided temperature, top_p and top_k.
    """
    ## TEMPERATURE
    logits = logits / temperature

    ## TOP P
    # sort logits, save original indices
    top_logits, top_indices = lax.top_k(logits, logits.shape[-1])

    # mask = jnp.full_like(logits, -float("inf"))
    cumulative_probs = jax.nn.softmax(top_logits, axis=-1).cumsum(axis=-1)
    top_p_mask = cumulative_probs < top_p

    # include the token that is higher than top_p as well
    top_p_mask = jnp.roll(top_p_mask, 1)
    top_p_mask |= top_p_mask.at[:, 0].set(True)

    ## TOP K
    top_k_mask = jnp.full_like(logits, False, dtype=bool)
    top_k_mask = top_k_mask.at[:, :top_k].set(True)

    # APPLY TOP P AND TOP K
    # combine masks (intersection - allow only logits that conform to both filters)
    mask = top_p_mask & top_k_mask

    # keep at least one value
    mask = mask.at[:, :1].set(True)

    top_new_logits = jnp.where(mask, top_logits, -float("inf"))
    new_logits = lax.sort_key_val(top_indices, top_new_logits)[-1]

    # SAMPLE
    next_token = jax.random.categorical(rng, new_logits, axis=-1)
    return next_token


@struct.dataclass(kw_only=True)
class SamplingState:
    """
    Internal sampling state.

    Attributes:
        step: Number of decoding steps taken so far (between [0,
        max_new_tokens]).
        done: For each sequence in the batch, `True` if the sequence is done (i.e
        has predicted a `<eos>` token).
        last_token: Model input for the next sampling step.
        last_token_pos: The RoPE position of the last token in the input. Used to
        compute the positional embedding, so includes MM tokens, but ignores all
        padding.
        predicted_tokens: Fixed-size buffer for accumulating the output tokens.
        cache: Attention KV cache.
        rng: Seed to use for sampling.
        init_cache_length: Length of the cache length in the pre-fill phase. Include
        the prompt, the MM tokens, and the previous turns.
        full_attention_mask: Pre-computed attention mask for the full sequence.
    """

    step: jax.Array  # Int['']
    done: jax.Array  # Bool['B']
    last_token: jax.Array  # Int['B']
    last_token_pos: jax.Array  # Int['B']
    predicted_tokens: jax.Array  # Int['B max_out_length']
    cache: Cache
    rng: jax.random.PRNGKey
    # static values:
    init_cache_length: jax.Array  # Int['']
    full_attention_mask: jax.Array  # Bool['B cache_length']

    @property
    def used_cache_length(self) -> jax.Array:  # Int['']
        """Length of the cache currently used."""
        return self.init_cache_length + self.step

    @property
    def attention_mask_for_step(self) -> jax.Array:  # Bool['B cache_length']
        """Attention mask for the current step."""
        # Select the slice of the attention mask for the current step.
        # For step == 2, init_cache_length == 5:
        # In:  [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, ..., 1, 1, 1]
        # Out: [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, ..., 0, 0, 0]

        cache_length = self.full_attention_mask.shape[-1]

        # +1 because the current step can be self-attended too.
        step_mask = jnp.arange(cache_length) < self.used_cache_length + 1
        attention_mask = self.full_attention_mask * step_mask
        return attention_mask


@partial(nnx.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def sample(
    model,
    prompt_tokens: jax.Array,
    images: jax.Array | None = None,
    eos_token_id: int | tuple[int] = 1,
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    cache_dtype: jnp.dtype = jnp.bfloat16,
    rng: jax.Array = jax.random.key(0),
):

    def sample_cond_fn(state: SamplingState):
        """state termination condition fn."""
        # has_reached_max_length = state.last_token_pos == max_length
        has_reached_max_length = state.init_cache_length + state.step >= max_length
        all_sequence_finished = jnp.all(state.done)
        finish_generation = jnp.logical_or(
            has_reached_max_length, all_sequence_finished
        )
        return ~finish_generation

    def sample_body_fn(
        state: SamplingState,
    ) -> SamplingState:
        """Single sampling step."""

        model = nnx.merge(static, model_state)
        # if hasattr(model, "vision_encoder") and isinstance(
        #     model.vision_encoder, nnx.bridge.ToNNX
        # ):
        #     model.vision_encoder.rngs = nnx.Rngs(
        #         state.rng
        #     )  # hack to fix tracing context error
        out = model(
            tokens=state.last_token[..., None],
            cache=state.cache,
            positions=state.last_token_pos[..., None],
            attention_mask=state.attention_mask_for_step[:, None, :],  # B 1 L
        )

        logits = out.logits
        # Logit is `B L V` with `L=1`, so collapse the L dimension.
        logits = einops.rearrange(logits, "B 1 V -> B V")
        # if self.forbidden_tokens:  # Eventually filter out the forbidden tokens.
        #     logits = logits.at[:, self.forbidden_tokens].set(-jnp.inf)

        # Sample next token.
        next_rng, curr_rng = jax.random.split(state.rng)
        next_token = sample_token(
            curr_rng,
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Update the buffers to save the outputs.
        predicted_tokens = state.predicted_tokens.at[:, state.step].set(next_token)
        # predicted_logits = state.predicted_logits.at[:, state.step].set(logits)

        # Check whether we have reached an end token.
        done = state.done | jnp.isin(next_token, jnp.asarray(eos_token_ids))

        return SamplingState(
            step=state.step + 1,
            done=done,
            last_token=next_token,
            # Only update the position if we are not done. The last predicted token
            # is still incremented, so we use previous `state.done`.
            last_token_pos=state.last_token_pos + ~state.done,
            predicted_tokens=predicted_tokens,
            # predicted_logits=predicted_logits,
            cache=out.cache,
            rng=next_rng,
            init_cache_length=state.init_cache_length,
            full_attention_mask=state.full_attention_mask,
        )

    # if model.vision_encoder is not None:
    #     model.vision_encoder.rngs = nnx.Rngs(0)  # hack to bypass TraceContextError

    bsz = prompt_tokens.shape[0]
    cache_length = max_length
    eos_token_ids = jnp.array(eos_token_id)

    # as a workaround for stateful models with Rngs (used in ToNNX submodule),
    # we split the graphdef and model state, and then just
    # merge them in sample_body_fn()
    static, model_state = nnx.split(model, ...)

    cache = model.init_cache(batch_size=bsz, dtype=cache_dtype, cache_length=max_length)

    # input = _types.Input(text=sequences, images=images, config=model.config.input_config)
    input = _types.Input(
        text=prompt_tokens, images=images, config=model.config.input_config
    )

    # prefill cache
    out = model(
        # tokens=input.tokens_with_mm,
        tokens=input.text,
        images=input.images,
        cache=cache,
        positions=input.positions,
        # attention_mask=input.attention_mask,
        attention_mask=_functional.pad(
            input.attention_mask, max_length=cache_length
        ).astype(int),
        return_last_only=True,
    )

    # We follow logic and notation of the original sampler implementation in Gemma,
    # but in our context they require some clarification. `full_attention_mask`
    # below is actually a padding mask (shape B x L) that is combined with
    # causal mask (a.k.a. step mask) in `SamplingState.attention_mask_for_step()`
    # to let model know what it's allowed to look at. Note that causal mask already
    # hides all future tokens, so this padding mask is only used to hide _past_
    # tokens, e.g. padded tokens in a batch of prompts of differrnt lengths.
    # See the original implementation for the details of what this
    # mask actually looks like for sequences of different length:
    # https://github.com/google-deepmind/gemma/blob/532b12d81ae077b7cd2f9d921ae50eceb9a83d9e/gemma/gm/text/_prefill.py#L283
    full_attention_mask = _make_full_attention_mask(
        input=input, prev_turns=None, cache_length=max_length
    )

    # initialize state
    state = SamplingState(
        step=jnp.asarray(0),
        done=jnp.zeros((input.batch_size,), dtype=jnp.bool_),
        # Last token for autoregressive sampling.
        last_token=input.last_token,
        last_token_pos=input.last_token_pos,
        predicted_tokens=jnp.zeros((bsz, max_length), dtype=jnp.int32),
        cache=out.cache,
        rng=rng,
        full_attention_mask=full_attention_mask,
        init_cache_length=out.cache["layer_0"]["end_index"][
            0
        ],  # jnp.asarray(init_cache_length),
    )

    state = jax.lax.while_loop(sample_cond_fn, sample_body_fn, state)
    return state.predicted_tokens


class Sampler:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def load_model(variant: str, mesh: jax.sharding.Mesh | None = None):
        tokenizer, model = load_model(variant, mesh=mesh)
        return Sampler(tokenizer=tokenizer, model=model)

    def __repr__(self):
        return "Sampler"

    def sample(
        self,
        prompt: str,
        images: jax.Array | Image.Image | list[Image.Image] | None = None,
        max_length: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        pad_to_multiple_of: int = 128,
        cache_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        prompt_tokens = encode_batch(
            self.tokenizer, [prompt], pad_to_multiple_of=pad_to_multiple_of
        )
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list):
            assert all(isinstance(img, Image.Image) for img in images)
            assert len(images) > 0
            img_size = images[0].size
            assert all(
                img.size == img_size for img in images
            ), "All images must have the same size"
            # array of size (B N H W C), where B=1
            images = jnp.stack([jnp.array(img) for img in images])[None, ...]
        st = self.tokenizer.special_tokens
        out_tokens = sample(
            self.model,
            prompt_tokens,
            images=images,
            eos_token_id=(
                st.EOS,
                st.END_OF_TURN,
            ),
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cache_dtype=cache_dtype,
            rng=rngs(),
        )
        completion: str = self.tokenizer.decode(out_tokens[0])
        # tokenizer doesn't remove <end_of_turn> in instruction-tuned models,
        # so we do it manually here
        completion = completion.removesuffix("<end_of_turn>")
        return completion
