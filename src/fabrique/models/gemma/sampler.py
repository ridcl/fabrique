from functools import partial

import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import einops
from flax import nnx, struct
from gemma import gm
from gemma.gm.utils import _types
from gemma.gm.text._prefill import _make_full_attention_mask

from fabrique.loading import update_module_from_params
from fabrique.models.gemma.load_rules import RULES
from fabrique.models.gemma.modeling import Transformer


LayerCache = dict[str, jax.Array]   # gemma.gm.nn._modules.LayerCache
Cache = dict[str, LayerCache]       # gemma.gm.nn._config.Cache


def load_gemma(variant: str):
    match variant.lower():
        case "1b":
            config = gm.nn.Gemma3_1B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_1B_IT
        case "4b":
            config = gm.nn.Gemma3_4B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
        case "12b":
            config = gm.nn.Gemma3_12B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_12B_IT
        case "27b":
            config = gm.nn.Gemma3_27B.config
            ckpt = gm.ckpts.CheckpointPath.GEMMA3_27B_IT
        case _:
            raise ValueError(f"Unknown Gemma variant: {variant}")
    param_dtype = jnp.bfloat16
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )
    params = gm.ckpts.load_params(ckpt)
    update_module_from_params(model, RULES, params)
    model.vision_encoder.rngs = nnx.Rngs(0)   # otherwise rngs will be abstract array
    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model



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


# @struct.dataclass
# class SampleState:
#     cur_len: jnp.ndarray
#     sequences: jnp.ndarray
#     running_token: jnp.ndarray
#     is_sent_finished: jnp.ndarray
#     # start_pos: int
#     cache: Cache
#     model_state: nnx.State
#     static: nnx.GraphDef
#     prng_key: jnp.ndarray


@struct.dataclass(kw_only=True)
class SamplingState:
    """Internal sampling state.

    Attributes:
        step: Number of steps decoding steps taken so far (between [0,
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

    step: jax.Array                 # Int['']
    done: jax.Array                 # Bool['B']
    last_token: jax.Array           # Int['B']
    last_token_pos: jax.Array       # Int['B']
    predicted_tokens: jax.Array     # Int['B max_out_length']
    cache: Cache
    rng: jax.random.PRNGKey
    # static values:
    init_cache_length: jax.Array    # Int['']
    full_attention_mask: jax.Array  # Bool['B cache_length']

    @property
    def used_cache_length(self) -> jax.Array:           # Int['']
        """Length of the cache currently used."""
        return self.init_cache_length + self.step

    @property
    def attention_mask_for_step(self) -> jax.Array:    # Bool['B cache_length']
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


@partial(nnx.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def sample(
    model,
    prompt_tokens: jax.Array,
    pad_token_id: int,
    eos_token_id: int | tuple[int],
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    rng: jax.Array = jax.random.key(0),
):

    def sample_cond_fn(state: SamplingState):
        """state termination condition fn."""
        has_reached_max_length = state.last_token_pos == max_length
        all_sequence_finished = jnp.all(state.done)
        finish_generation = jnp.logical_or(
            has_reached_max_length, all_sequence_finished
        )
        return ~finish_generation[0]

    def sample_body_fn(
        state: SamplingState,
    ) -> SamplingState:
        """Single sampling step."""

        model = nnx.merge(static, model_state)
        out = model(
            tokens=state.last_token[..., None],
            cache=state.cache,
            positions=state.last_token_pos[..., None],
            attention_mask=state.attention_mask_for_step[:, None, :],  # B 1 L
        )

        logits = out.logits
        # Logit is `B L V` with `L=1`, so collapse the L dimension.
        logits = einops.rearrange(logits, 'B 1 V -> B V')
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


    bsz = prompt_tokens.shape[0]
    eos_token_ids = jnp.array(eos_token_id)

    # per batch-item holding current token in loop
    sequences = jnp.full((bsz, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, prompt_tokens, (0, 0))

    # as a workaround for stateful models with Rngs (used in ToNNX submodule),
    # we split the graphdef and model state, and then just
    # merge them in sample_body_fn()
    static, model_state = nnx.split(model, ...)

    # TODO: accept dtype as arg
    cache = model.init_cache(batch_size=bsz, dtype=jnp.bfloat16, cache_length=max_length)

    input = _types.Input(text=sequences, images=None, config=model.config.input_config)
    prompt_input = _types.Input(text=prompt_tokens, images=None, config=model.config.input_config)


    # prefill cache
    out = model(
        tokens=input.tokens_with_mm,
        # images=prompt_input.images,
        cache=cache,
        positions=input.positions,
        attention_mask=input.attention_mask,
        return_last_only=True
    )

    used_cache_length = input.last_token_pos.max()  # TODO: is it a good choice?
    end_index = used_cache_length
    cache = out.cache
    for layer_data in cache.values():
        layer_data['end_index'] = jnp.full_like(
            layer_data['end_index'], fill_value=end_index
        )

    # We follow logic and notation of the original sampler implementation in Gemma,
    # but in our context they require some clarification. `full_attention_mask`
    # below is actually a padding mask (shape B x L) that is combined with
    # causal mask (a.k.a. step mask) in `SamplingState.attention_mask_for_step()`
    # to let model know what it's allowed to look at. Note that causal mask already
    # hides all future tokens, so this padding mask is only useful to hide _past_
    # tokens. See the original implementation for the details of what this
    # mask actually looks like for sequences of different length:
    # https://github.com/google-deepmind/gemma/blob/532b12d81ae077b7cd2f9d921ae50eceb9a83d9e/gemma/gm/text/_prefill.py#L283
    full_attention_mask = _make_full_attention_mask(input=prompt_input, prev_turns=None, cache_length=max_length)


    # initialize state
    state = SamplingState(
        step=jnp.asarray(0),
        done=jnp.zeros((input.batch_size,), dtype=jnp.bool_),
        # Last token for autoregressive sampling.
        last_token=input.last_token,
        last_token_pos=input.last_token_pos,
        # In theory, those values only need to be `B max_new_tokens`, however,
        # to avoid re-compilation when prompt length and max_new_tokens changes,
        # we set this to the fixed maximum static size.
        predicted_tokens=sequences,
        # predicted_logits=jnp.zeros(
        #     (batch_size, self.max_out_length, out.logits.shape[-1]),
        #     dtype=jnp.float32,
        # ),
        cache=cache,
        rng=rng,
        full_attention_mask=full_attention_mask,
        init_cache_length=jnp.asarray(used_cache_length),
    )

    state = jax.lax.while_loop(sample_cond_fn, sample_body_fn, state)
    # state = debug_while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
    return state.predicted_tokens


################################################################


def encode_batch(tokenizer: gm.text.Gemma3Tokenizer, prompts: list[str], add_bos=False, add_eos=False):
    bsz = len(prompts)
    token_lists = [tokenizer.encode(prompt, add_bos=add_bos, add_eos=add_eos) for prompt in prompts]
    max_prompt_length = max(len(lst) for lst in token_lists)
    tokens = np.full((bsz, max_prompt_length), fill_value=tokenizer.special_tokens.PAD)
    for i in range(bsz):
        length = len(token_lists[i])
        tokens[i][: length] = token_lists[i]
    return jnp.array(tokens)



def example():
    tokenizer, model = load_gemma("4b")
    prompts = [
        """<start_of_turn>user\nWrite a poem about a stool<end_of_turn>\n<start_of_turn>model\n""",
        """<start_of_turn>user\nWho is John Snow?<end_of_turn>\n<start_of_turn>model\n"""
    ]
    prompt_tokens = encode_batch(tokenizer, prompts, add_bos=True)

    # jax.config.update("jax_explain_cache_misses", True)

    rngs = nnx.Rngs(0)

    sequences = sample(
        model,
        prompt_tokens,
        pad_token_id=tokenizer.special_tokens.PAD,
        eos_token_id=tokenizer.special_tokens.EOS,
        max_length=512,
        temperature=1,
        # top_p=0.5,
        # top_k=3,
        rng=rngs()
    )
    print(tokenizer.decode(sequences[0]))
    print(tokenizer.decode(sequences[1]))


    pad_token_id=tokenizer.special_tokens.PAD
    eos_token_id=tokenizer.special_tokens.EOS
    max_length = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    rngs: nnx.Rngs = nnx.Rngs(0)
    rng = rngs()

    bsz, cur_len = prompt_tokens.shape
    eos_token_ids = jnp.array(eos_token_id)

    # per batch-item holding current token in loop
    sequences = jnp.full((bsz, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, prompt_tokens, (0, 0))

    # per batch-item state bit indicating if sentence has finished.
    is_sent_finished = jnp.zeros((bsz,), dtype=jnp.bool_)

    # cache = model.init_cache(batch_size=bsz, dtype=jnp.bfloat16, cache_length=max_length)
    cache = None

    from gemma.gm.utils._types import Input

    input = Input(text=sequences, images=None, config=model.config.input_config)

    # for i in range(max_length - prompt_tokens.shape[-1]):
    #     cur_len = prompt_tokens.shape[-1] + i
    #     # out = model(tokens=sequences, cache=cache)
    #     input_mask = (sequences != pad_token_id).astype(jnp.int32)
    #     attention_mask = make_attention_mask(sequences).astype(bool)
    #     positions = jnp.cumsum(input_mask)[None, :]
    #     out = model_nn.apply({"params": params}, tokens=sequences, cache=cache, attention_mask=attention_mask, positions=positions)
    #     # cache = out.cache
    #     next_token = out.logits[:, -1, :].argmax(axis=-1)
    #     # next_token = sample_token(rngs(), out.logits[:, -1, :], temperature=2)

    #     sequences = sequences.at[:, cur_len].set(next_token)
    #     print("-" * 20 + tokenizer.decode(sequences[0]))



    # causal_full = precompute_causal_mask(max_length)

    # for i in range(max_length - prompt_tokens.shape[1]):
    #     cur_len = prompt_tokens.shape[1] + i

    #     # 1 for valid tokens, 0 for pad
    #     padding_mask = (jnp.arange(max_length) < cur_len).astype(jnp.float32)  # [max_length]
    #     padding_mask = padding_mask[None, :]  # [1, max_length]
    #     padding_mask = jnp.repeat(padding_mask, sequences.shape[0], axis=0)  # [batch, max_length]

    #     # Combine causal and padding into full mask
    #     attn_mask = causal_full[None, :, :] * padding_mask[:, None, :]  # [batch, max_length, max_length]

    #     out = model(
    #         tokens=sequences,  # full buffer, fixed shape
    #         attention_mask=attn_mask
    #     )

    #     next_token = out.logits[:, cur_len - 1, :].argmax(axis=-1)
    #     sequences = sequences.at[:, cur_len].set(next_token)

    #     print("-" * 20 + tokenizer.decode(sequences[0]))


    # # model_nn = gm.nn.Gemma3_4B()
    # # params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

    # # this one actually works 0_o
    # # greedy_slow_prefilled(model_nn.apply, params, sequences, prompt_tokens.shape[-1], max_length=512, tokenizer=tokenizer)

    # greedy_slow_prefilled2(model, sequences, prompt_tokens.shape[-1], max_length=512, tokenizer=tokenizer)



def precompute_causal_mask(max_length):
    # Lower-triangular causal mask for the whole max_length
    return jnp.tril(jnp.ones((max_length, max_length), dtype=jnp.float32))


# def make_attention_mask(sequences, pad_token_id=0):
#     """
#     sequences: [batch, seq_len] int
#     Returns:   [batch, seq_len, seq_len] float32
#     """
#     batch, seq_len = sequences.shape

#     # Causal mask [seq_len, seq_len]: 1 below diagonal, 0 above
#     causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))

#     # Padding mask: 1 for real tokens, 0 for pad
#     padding = (sequences != pad_token_id).astype(jnp.float32)  # [batch, seq_len]

#     # Combine: each query token can only see non-pad keys up to itself
#     mask = causal[None, :, :] * padding[:, None, :]  # [batch, seq_len, seq_len]
#     return mask


# def greedy_slow_prefilled(model_apply, params, sequences, prompt_len, max_length, tokenizer=None, pad_token_id=0):
#     # sequences: [batch, max_length] with first prompt_len filled, rest = pad_token_id
#     cur_len = prompt_len
#     for i in range(max_length - prompt_len):
#         out = model_apply({"params": params}, tokens=sequences[:, :cur_len])   # IMPORTANT: slice
#         next_token = out.logits[:, -1, :].argmax(axis=-1)
#         sequences = sequences.at[:, cur_len].set(next_token)
#         cur_len += 1
#         if tokenizer:
#             print("-" * 20, tokenizer.decode(sequences[0]))
#     return sequences



# def greedy_slow_prefilled2(model, sequences, prompt_len, max_length, tokenizer=None, pad_token_id=0):
#     # sequences: [batch, max_length] with first prompt_len filled, rest = pad_token_id
#     cur_len = prompt_len
#     for i in range(max_length - prompt_len):
#         out = model(tokens=sequences[:, :cur_len])   # IMPORTANT: slice
#         next_token = out.logits[:, -1, :].argmax(axis=-1)
#         sequences = sequences.at[:, cur_len].set(next_token)
#         cur_len += 1
#         if tokenizer:
#             print("-" * 20, tokenizer.decode(sequences[0]))
#     return sequences

# TODO: so the problem is that we don't pass attention mask AND
# don't multiply it with input_mask (padding_mask)
# 1. we need to figure out how to do it in Gemma 3 with images. Use Input class?
# 2. we also need to fix it in the main generation loop OR migrate everything to
#    the new sampler