import re
import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from PIL import Image

from fabrique.sampling import Sampler, sample
from fabrique.tokenizer_utils import encode_batch

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


def similar_texts(text1, text2, threshold=0.5, common_prefix=30):
    """
    Fuzzy comparison between two text. In these tests, this functions
    ensures that the LLM doesn't generate garbage and generally follows
    the given topic.

    Note that precise comparison of long generated text is impossible
    because of slight differences in installed packages.
    """
    pattern = r"[.,\ \n\t]"
    words1 = set(re.split(pattern, text1))
    words2 = set(re.split(pattern, text2))
    score = len(words1 & words2) / len(words1 | words2)
    return text1[:common_prefix] == text2[:common_prefix] and score >= threshold


@pytest.fixture(scope="module")
def sampler():
    sampler = Sampler.load_model("gemma-3-4b-it")
    yield sampler
    # unload
    sampler.model = None
    sampler.tokenizer = None


def test_single_prompt_with_image(sampler):
    tokenizer, model = sampler.tokenizer, sampler.model

    prompts = [
        """<start_of_turn>user\n<start_of_image>Describe the image in a few sentences<end_of_turn>\n<start_of_turn>model\n"""
    ]
    prompt_tokens = encode_batch(tokenizer, prompts)
    images = jnp.array(Image.open("tests/bird.jpg"))[None, None, ...]

    out_tokens = sample(
        model,
        prompt_tokens,
        images=images,
        eos_token_id=(
            tokenizer.special_tokens.EOS,
            tokenizer.special_tokens.END_OF_TURN,
        ),
        max_length=512,
        temperature=1,
        rng=jax.random.key(0),
    )
    completion = tokenizer.decode(out_tokens[0])
    target = "Here's a description of the image:\n\nThe image showcases a vibrant European Robin perched on a thin, gray branch. The bird has a distinctive orange breast and red face, contrasting beautifully with its gray and white plumage. It features a bright black eye and a short, pointed beak, creating a charming and detailed portrait of this iconic songbird.<end_of_turn>"
    assert similar_texts(completion, target)


def test_text_only_batch(sampler):
    tokenizer, model = sampler.tokenizer, sampler.model

    prompts = [
        """<start_of_turn>user\nWrite a tanku about a stool<end_of_turn>\n<start_of_turn>model\n""",
        """<start_of_turn>user\nWho is John Snow? Reply in one sentence\n<start_of_turn>model\n""",
    ]
    prompt_tokens = encode_batch(tokenizer, prompts)
    out_tokens = sample(
        model,
        prompt_tokens,
        eos_token_id=(
            tokenizer.special_tokens.EOS,
            tokenizer.special_tokens.END_OF_TURN,
        ),
        max_length=512,
        temperature=1,
        rng=jax.random.key(0),
    )
    completions = [tokenizer.decode(t) for t in out_tokens]
    targets = [
        "A humble, wooden seat,\nSupporting weary, tired feet,\nSilence in its wait. \n\nSimple, strong, and true,\nA grounding point for you,\nJust a stool, for few.<end_of_turn>",
        "John Snow was a 19th-century English anesthesiologist and physician who is best known for his pioneering work in antiseptic surgery and for documenting the first modern outbreak of cholera in London.<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>",
    ]
    assert similar_texts(completions[0], targets[0])
    assert similar_texts(completions[1], targets[1])


def test_prompts_with_lots_of_padding(sampler):
    tokenizer, model = sampler.tokenizer, sampler.model
    # sampling with lots of padding tokens (fails if we mess up attention)
    TMPL = """<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"""
    prompts = [
        TMPL.format("How much is 2 + 2?"),
        TMPL.format(
            "You are the best mathematician in history. Your solve "
            + "the most complicated tasks. Now you need to answer the "
            + "following question in the most concise way: how much is 2 + 2?"
        ),
    ]
    tokens_batch = encode_batch(tokenizer, prompts)
    tokens_pad = encode_batch(tokenizer, prompts[0:1], pad_to_multiple_of=128)

    # batch sampling
    out_batch = sample(model, tokens_batch)
    out_text = tokenizer.decode(out_batch[0, :])
    assert out_text.replace("<end_of_turn>", "").replace("\n", "") == "2 + 2 = 4"

    # single seq batch sampling
    out_batch_0 = sample(model, tokens_batch[0:1, :])
    out_text = tokenizer.decode(out_batch_0[0, :])
    assert out_text.replace("<end_of_turn>", "").replace("\n", "") == "2 + 2 = 4"

    out_pad = sample(model, tokens_pad)
    out_text = tokenizer.decode(out_pad[0, :])
    assert out_text.replace("<end_of_turn>", "").replace("\n", "") == "2 + 2 = 4"


def test_sampler_class(sampler):
    prompt = """<start_of_turn>user\n<start_of_image>Describe the image in a few sentences<end_of_turn>\n<start_of_turn>model\n"""
    image = Image.open("tests/bird.jpg")

    rngs = nnx.Rngs(0)

    completion = sampler.sample(
        prompt, images=[image], max_length=512, temperature=1, rngs=rngs
    )
    target = "Here's a description of the image:\n\nThe image showcases a vibrant red robin perched on a thin, gray branch. The bird has a striking orange breast and reddish-brown back, contrasted with a gray head and a bright black eye. The background is softly blurred, creating a shallow depth of field that emphasizes the robin as the central focus of the photograph."
    # weird case of test <> REPL incinsistency, need to find a better way to compare completion and target
    assert similar_texts(completion[:100], target[:100])

    completion = sampler.sample(
        prompt,
        images=[image],
        max_length=1024,
        temperature=1,
        pad_to_multiple_of=512,
        rngs=rngs,
    )
    target = "Here's a description of the image:\n\nThe photo showcases a vibrant red robin perched on a weathered branch. It has a distinctive orange breast and cap, contrasting beautifully with its gray-brown back and fluffy white underparts. The robin’s dark, alert eye and delicate beak add to its charming appearance, and the soft, natural lighting gives the image a peaceful and detailed quality."
    assert similar_texts(completion[:100], target[:100])
