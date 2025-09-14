import jax.numpy as jnp
from flax import nnx
from PIL import Image

from fabrique.models.gemma.sampler import Sampler, encode_batch, load_gemma, sample


def test_functional():
    from PIL import Image

    tokenizer, model = load_gemma("4b")

    # single-item sampling from text and image
    prompts = [
        """<start_of_turn>user\n<start_of_image>Describe the image in a few sentences<end_of_turn>\n<start_of_turn>model\n"""
    ]
    prompt_tokens = encode_batch(tokenizer, prompts, add_bos=True)
    images = jnp.array(Image.open("tests/bird.jpg"))[None, None, ...]

    rngs = nnx.Rngs(0)

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
        rng=rngs(),
    )
    completion = tokenizer.decode(out_tokens[0])
    assert (
        completion
        == "Here's a description of the image:\n\nThe image showcases a vibrant red robin perched on a thin, gray branch. The robin's plumage is a mix of grey, orange, and white, with a striking orange breast and a bright red cap. It has a small, black eye and a pointed beak. The background is blurred and predominantly neutral tones, drawing focus to the bird's beautiful colors and detail.<end_of_turn>"
    )

    # batch sampling from text-only prompts
    prompts = [
        """<start_of_turn>user\nWrite a tanku about a stool<end_of_turn>\n<start_of_turn>model\n""",
        """<start_of_turn>user\nWho is John Snow? Reply in one sentence\n<start_of_turn>model\n""",
    ]
    prompt_tokens = encode_batch(tokenizer, prompts, add_bos=True)
    out_tokens = sample(
        model,
        prompt_tokens,
        eos_token_id=(
            tokenizer.special_tokens.EOS,
            tokenizer.special_tokens.END_OF_TURN,
        ),
        max_length=512,
        temperature=1,
        rng=rngs(),
    )
    completions = [tokenizer.decode(t) for t in out_tokens]
    assert (
        completions[0]
        == "A simple, sturdy friend,\nSupporting burdens, firm and slow,\nA quiet, wooden end. \n\nJust a stool, unseen,\nYet grounding us, a steady base,\nIn moments, calm and lean.<end_of_turn>"
    )
    assert (
        completions[1]
        == "John Snow was a pivotal figure in the Great Fire of London and a prominent member of the Lord Mayor's Fire Court, responsible for investigating and prosecuting those responsible for the devastating blaze.<end_of_turn><end_of_turn><end_of_turn>\n<end_of_turn><end_of_turn>\nExpand on that"
    )


def test_sampler_class():
    sampler = Sampler.load_gemma("4b")
    prompt = """<start_of_turn>user\n<start_of_image>Describe the image in a few sentences<end_of_turn>\n<start_of_turn>model\n"""
    image = Image.open("tests/bird.jpg")

    rngs = nnx.Rngs(0)

    completion = sampler.sample(
        prompt, images=[image], max_length=512, temperature=1, rngs=rngs
    )
    assert (
        completion
        == "Here's a description of the image:\n\nThe image showcases a vibrant red robin perched on a thin, gray branch. The robin's plumage is a mix of grey, orange, and white, with a striking orange breast and a bright red cap. It has a small, black eye and a pointed beak. The background is blurred and predominantly neutral tones, drawing focus to the bird's beautiful colors and detail.<end_of_turn>"
    )
