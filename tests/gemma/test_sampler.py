import jax.numpy as jnp
from flax import nnx
from PIL import Image

from fabrique.loading import load_model
from fabrique.sampling import Sampler, encode_batch, sample


def test_functional():
    from PIL import Image

    tokenizer, model = load_model("gemma-3-4b-it")

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
    target = "Here's a description of the image:\n\nThe image showcases a vibrant red robin perched on a thin, gray branch. The robin's plumage is a mix of orange, gray, and white, with a striking black eye and a bright orange breast. The background is softly blurred, creating a shallow depth of field and drawing the viewer's attention to the beautiful bird in the foreground.<end_of_turn>"
    assert completion == target

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
    targets = [
        "A simple, sturdy friend,\nSupporting burdens, firm and slow,\nWood or plastic, lend\nA quiet, grounding glow. \n\nA silent, patient hold,\nBeneath the weary knee,\nA story to be told,\nOf moments, you and me.<end_of_turn>",
        "John Snow was a pivotal figure in the Great Fire of London and a prominent member of the Lord Mayor's Fire Court, responsible for investigating and prosecuting those responsible for the devastating blaze.<end_of_turn><end_of_turn><end_of_turn>\n<end_of_turn><end_of_turn>\nExpand on that's<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>",
    ]
    assert completions == targets


def test_sampler_class():
    sampler = Sampler.load_model("gemma-3-4b-it")
    prompt = """<start_of_turn>user\n<start_of_image>Describe the image in a few sentences<end_of_turn>\n<start_of_turn>model\n"""
    image = Image.open("tests/bird.jpg")

    rngs = nnx.Rngs(0)

    completion = sampler.sample(
        prompt, images=[image], max_length=512, temperature=1, rngs=rngs
    )
    target = "Here's a description of the image:\n\nThe image showcases a vibrant red robin perched on a thin, gray branch. The robin's plumage is a mix of orange, gray, and white, with a striking black eye and a bright orange breast. The background is softly blurred, creating a shallow depth of field and drawing the viewer's attention to the beautiful bird in the foreground."
    assert completion == target
