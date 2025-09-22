import jax
import jax.numpy as jnp

from fabrique import LLM


def check_padding(model, tokenizer, prompt):
    tokens = jnp.array(tokenizer.encode(prompt).ids).reshape(1, -1)
    # mask first tokens to avoid correlation with the causal mask
    pad_positions = (slice(None), slice(0, 3))
    not_pad_positions = (slice(None), slice(3, None))
    padding_mask = jnp.ones(tokens.shape, dtype=bool).at[pad_positions].set(False)
    tokens_modified = jnp.where(padding_mask, tokens, tokens + 1)

    # w/o padding mask, modified tokens should lead to modified out
    out = model(tokens, 0)
    out_mod = model(tokens_modified, 0)
    assert (out != out_mod).sum() / out.size > 0.99  # almost all not equal

    # w/ padding mask, changes in the padded part should have no effect
    out = model(tokens, 0, padding_mask=padding_mask)
    out_mod = model(tokens_modified, 0, padding_mask=padding_mask)

    assert (out[not_pad_positions] == out_mod[not_pad_positions]).all()
    assert (out[pad_positions] != out_mod[pad_positions]).sum() / out[
        pad_positions
    ].size > 0.99


def load_and_check(model_id: str, revision: str, prompt: str, expected: str):
    kwargs = {
        "max_seq_len": 32,
        "max_batch_size": 1,
        "dtype": jnp.bfloat16,
    }  # limit cache size
    llm = LLM.from_pretrained(model_id, revision=revision, **kwargs)
    key = jax.random.key(818)
    result = llm.generate(prompt, new_only=False, max_length=32, prng_key=key)
    assert isinstance(result, str)
    assert result == expected

    model, tokenizer = llm.model, llm.tokenizer
    check_padding(model, tokenizer, prompt)


def test_llama():
    model_id = "meta-llama/Meta-Llama-3-8B"
    prompt = "Once upon a time"
    revision = "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    expected = "Once upon a time the British government established a board charged with running the country, the British economy and the welfare of her citizens.\nThis board was duly composed"
    load_and_check(model_id, revision, prompt, expected)


def test_phi():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    revision = "c1358f8"
    prompt = "Once upon a time"
    expected = "Once upon a time in the magical land of Wordspell, every word had its unique spell, and words were connected with pathways of letters that dan"
    load_and_check(model_id, revision, prompt, expected)


def test_qwen2():
    model_id = "Qwen/Qwen2-7B-Instruct"
    revision = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
    prompt = "Once upon a time"
    expected = "Once upon a time, I was a young man studying at a well-known university in Hong Kong. As a student there, I became quite acquainted with the work"
    load_and_check(model_id, revision, prompt, expected)


def test_bert():
    from fabrique.loading import from_pretrained

    model_id = "google-bert/bert-base-uncased"

    tokenizer, model, _hf_config = from_pretrained(model_id)
    tokens = tokenizer.encode("Once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)
    out = model(tokens)

    # from huggingface model
    expected_slice = jnp.asarray(
        [
            0.06543218,
            -0.35718504,
            -0.3604128,
            0.0042748,
            0.23516399,
            -0.09302475,
            -0.2931913,
            0.6673709,
            -0.02174259,
            -0.38075727,
        ]
    )
    atol = 5e-2
    assert jnp.allclose(out[0, 0, :10], expected_slice, atol=atol)
    assert jnp.allclose(out.sum(), -80.01314, atol=atol)
    assert jnp.allclose(out.var(), 0.3342754, atol=atol)

    mask = jnp.ones_like(tokens)
    mask = mask.at[0, -3:].set(0)
    segments = jnp.zeros_like(tokens)
    out = model(tokens, mask, segments)

    expected_slice = jnp.asarray(
        [
            -0.2556453,
            0.12366249,
            -0.70183635,
            0.26455373,
            -0.14038257,
            -0.27187574,
            -0.35302785,
            0.5766474,
            -0.55448484,
            -0.573969,
        ]
    )
    print(out[0, 0, :10] - expected_slice)
    print(jnp.abs(out[0, 0, :10] - expected_slice) < atol)
    assert jnp.allclose(out[0, 0, :10], expected_slice, atol=atol)
    assert jnp.allclose(out.sum(), -50.359936, atol=atol)
    assert jnp.allclose(out.var(), 0.25323534, atol=atol)
