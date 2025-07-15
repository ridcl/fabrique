from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import nnx, struct


def debug_while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


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


@struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    start_pos: int
    model_state: nnx.State
    static: nnx.GraphDef
    prng_key: jnp.ndarray


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
    prng_key: jax.Array = jax.random.key(0),
):
    def sample_cond_fn(state: SampleState):
        """state termination condition fn."""
        has_reached_max_length = state.cur_len == max_length
        all_sequence_finished = jnp.all(state.is_sent_finished)
        finish_generation = jnp.logical_or(
            has_reached_max_length, all_sequence_finished
        )
        return ~finish_generation

    def sample_body_fn(state: SampleState):
        """state update fn."""
        bsz = state.sequences.shape[0]
        model_state = state.model_state
        static = state.static
        model = nnx.merge(static, model_state)
        logits = model(state.running_token, state.start_pos)
        # note: model can return > bsz sequences so we limit logits
        next_token_logits = logits[:bsz, -1]

        next_token = sample_token(
            state.prng_key,
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        next_token = (
            next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        )
        next_is_sent_finished = state.is_sent_finished | jnp.isin(
            next_token, eos_token_ids
        )
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(
            state.sequences, next_token, (0, state.cur_len)
        )
        next_start_pos = state.cur_len

        next_prng_key = jax.random.split(state.prng_key)[0]

        _, next_model_state = nnx.split(model, ...)
        return state.replace(  # type: ignore[attr-defined]
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            start_pos=next_start_pos,
            model_state=next_model_state,
            prng_key=next_prng_key,
        )

    bsz, cur_len = prompt_tokens.shape
    eos_token_ids = jnp.array(eos_token_id)

    # per batch-item holding current token in loop
    sequences = jnp.full((bsz, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, prompt_tokens, (0, 0))

    # per batch-item state bit indicating if sentence has finished.
    is_sent_finished = jnp.zeros((bsz,), dtype=jnp.bool_)

    static, model_state = nnx.split(model, ...)

    # initialize state
    state = SampleState(
        cur_len=cur_len,  # type: ignore
        sequences=sequences,
        running_token=prompt_tokens,
        is_sent_finished=is_sent_finished,
        start_pos=0,
        model_state=model_state,
        static=static,
        prng_key=prng_key,
    )

    # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
    if prompt_tokens.shape[1] > 1:
        state = sample_body_fn(state)
    state = lax.while_loop(sample_cond_fn, sample_body_fn, state)
    # state = debug_while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
    return state.sequences


################################################################


# def example():
#     from fabrique.models.llm import LLM

#     llm = LLM.from_pretrained(
#         "meta-llama/Meta-Llama-3.1-8B-Instruct",
#         max_batch_size=1,
#         max_seq_len=4096,
#         dtype=jnp.bfloat16,
#         param_dtype=jnp.bfloat16
#     )
#     model, tokenizer, hf_config = llm.model, llm.tokenizer, llm.hf_config

#     # prompt = """{"name": "Thomas", "surname": "Anderson", "age":"""
#     # prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
#     #     What is the capital of France?<|eot_id|>
#     #     <|start_header_id|>assistant<|end_header_id|>"""
#     prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat time is it?<|eot_id|>'
#     prompt_tokens = tokenizer.encode(prompt).ids
#     prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)

#     jax.config.update("jax_explain_cache_misses", True)

#     rngs = nnx.Rngs(0)
#     prng_key = rngs()

#     sequences = sample(
#         model,
#         prompt_tokens,
#         pad_token_id=llm.special_tokens.eos_id,
#         eos_token_id=llm.special_tokens.eos_ids,
#         max_length=128,
#         temperature=1,
#         # top_p=0.5,
#         # top_k=3,
#         prng_key=prng_key
#     )
#     out = tokenizer.decode(sequences[0])
#     print(out)


#     self = llm
#     sequences = sample(
#         model,
#         prompt_tokens,
#         pad_token_id=self.special_tokens.eos_id,
#         eos_token_id=self.special_tokens.eos_ids,
#         max_length=max_length,
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         prng_key=prng_key,
#     )
#     out = tokenizer.decode(sequences[0])
#     print(out)

#     pad_token_id = hf_config.get("pad_token_id") or hf_config["eos_token_id"][0]
#     eos_token_id = tuple(hf_config["eos_token_id"])
#     max_length = 512
#     temperature: float = 1.0
#     top_p: float = 1.0
#     top_k: int = 50
#     rngs: nnx.Rngs = nnx.Rngs(0)
#     prng_key = rngs()
