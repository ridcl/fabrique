Quick Start
===========

*fabrique* uses Gemma 3 as its go-to LLM and VLM implementation. To sample from it, you can use a dedicated ``Sampler`` class::

    from flax import nnx
    from fabrique import Sampler

    sampler = Sampler.load_model("gemma-3-4b-it")
    prompt = """<start_of_turn>user\nWrite a poem about a stool<end_of_turn>\n<start_of_turn>model\n"""
    rngs = nnx.Rngs(95)
    completion = sampler.sample(prompt, rngs=rngs)
    print(completion)


You can also analyze images using ``<start_of_image>`` token and ``images`` keyword::

    import requests
    from PIL import Image

    image_url = "https://www.tracyvets.com/files/Parakeets.jpeg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    prompt = """<start_of_turn>user\n<start_of_image>Describe the image in a few sentences<end_of_turn>\n<start_of_turn>model\n"""

    completion = sampler.sample(prompt, images=[image], rngs=rngs)
    print(completion)


In addition to sampling capabilities, ``Sampler`` serves as a convenient container for the tokenizer and the model. This gives us access to the low level details such as model's logits::

    import jax.numpy as jnp

    tokenizer, model = sampler.tokenizer, sampler.model

    prompt = """<start_of_turn>user\nWrite a poem about a stool<end_of_turn>\n<start_of_turn>model\n"""
    tokens = jnp.array(tokenizer.encode(prompt))[None, :]  # make it 2D
    out = model(tokens, return_hidden_states=True)
    print(out.logits)
    print(out.hidden_states)
