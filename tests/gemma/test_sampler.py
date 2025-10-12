import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from PIL import Image

from fabrique.loading import load_model
from fabrique.sampling import Sampler, sample
from fabrique.tokenizer_utils import encode_batch


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
    assert completion == target


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
        "A simple, sturdy form,\nHolding weight with quiet grace,\nBeneath a weary form. \n\nJust wood, or metal cold,\nA brief support, then released,\nA silent, humble role.<end_of_turn>",
        "John Snow was a 19th-century English anesthesiologist and physician who is best known for his pioneering work in antiseptic surgery and for documenting the first modern outbreak of cholera in London.<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>",
    ]
    assert completions == targets


def test_prompts_with_lots_of_padding(sampler):
    tokenizer, model = sampler.tokenizer, sampler.model
    # sampling with lots of padding tokens (fails if we mess up attention)
    TMPL = """<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"""
    prompts = [
        TMPL.format("How much is 2 + 2?"),
        TMPL.format(
            "You are the best mathematician in history. Your solve " +
            "the most complicated tasks. Now you need to answer the " +
            "following question in the most concise way: how much is 2 + 2?"
        )
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
    assert completion == target

    completion = sampler.sample(
        prompt, images=[image], max_length=1024, temperature=1, pad_to_multiple_of=512, rngs=rngs
    )
    target = "Here's a description of the image:\n\nThe image showcases a vibrant red robin perched on a weathered branch. The bird's plumage displays a beautiful mix of orange, gray, and white feathers, with a distinctive red breast. It has a dark, alert eye and a pointed beak, and the soft lighting highlights its fluffy appearance against the blurred background of twigs and branches."
    assert completion == target