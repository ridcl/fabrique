===========
Multi-modal
===========

Gemma 3 with 4B, 12B and 27B parameters supports images as its input.
To use it in sampler, add special ``<start_of_image>`` token to prompt
and ``images`` keyword argument to the call::

    import requests
    from PIL import Image
    from fabrique.sampler import Sampler

    sampler = Sampler.load_model("gemma-3-4b-it")

    image_url = "https://imgs.xkcd.com/comics/linear_regression.png"
    image = Image.open(requests.get(image_url, stream=True).raw)

    # prompt: use <start_of_image> token
    prompt = """<start_of_turn>user\n<start_of_image>Explain the joke on the image<end_of_turn>\n<start_of_turn>model\n"""
    # model call: pass images keyword argument
    completion = sampler.sample(prompt, images=[image])
    print(completion)

In sampler, you can pass images as a PIL image, list of PIL images or a JAX array.

For training or fine-grained control, you can also pass images directly to the model::

    import jax.numpy as jnp

    tokenizer, model = sampler.tokenizer, sampler.model
    tokens = jnp.array(tokenizer.encode(prompt))[None, :]  # make it 2D

    # model expects JAX array of shape (B N H W C)
    # where B is batch size and N is number of images
    image_arr = jnp.array(image)[None, None, ...]
    out = model(tokens, images=image_arr)
    print(out)

Internally, Gemma encodes image(s) using SigLip2 vision encoder and projects
them to 256 "soft tokens". The model then inserts these soft tokens in place
of ``<start_of_image>`` token in the prompt.

Gemma was trained with images of size 896x896 pixels and will use
adaptive window algorithm on high-resolution and non-square images.