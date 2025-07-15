import jax
import jax.numpy as jnp

from fabrique import LLM, ChatMessage


def test_inference():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    kwargs = {
        "dtype": jnp.bfloat16,
        # limit cache size
        "max_seq_len": 32,
        "max_batch_size": 2,
    }
    llm = LLM.from_pretrained(model_id, **kwargs)
    key = jax.random.key(94)

    prompts = ["What is the meaning of life?", "Write a tanku"]
    chats = [[ChatMessage(role="user", content=prompt)] for prompt in prompts]

    result = llm.generate(prompts[0], max_length=32, prng_key=key)
    assert isinstance(result, str)

    result = llm.generate(prompts, max_length=32, prng_key=key)
    assert isinstance(result, list)

    result = llm.generate(chats[0], max_length=32, prng_key=key)
    assert isinstance(result, ChatMessage)

    result = llm.generate(chats, max_length=32, prng_key=key)
    assert isinstance(result, list)
