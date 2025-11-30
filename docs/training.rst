Training
========

Unlike many other libraries that provide predefined "trainers" for a fixed set of ML problems, *fabrique* gives users complete control over the training loop, while offering utilities for routine tasks. Let’s look at an example of fine-tuning an LLM for prompt–completion generation.

We’ll start by loading the model and tokenizer as usual::

    from fabrique.loading import load_model

    tokenizer, model = load_model("gemma-3-1b-it")

Next, we’ll load the dataset that we’ll use for training with the Hugging Face *datasets* library::

    from datasets import load_dataset

    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    print(dataset[0])
    # {'type': 'MATH_AnsAug',
    # 'query': "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
    # 'original_question': "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
    # 'response': "The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.\nIn this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.\nSo the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}

We’ll use the ``query`` field as the prompt and fine-tune the model to predict the ``response`` field as the completion. Since we’re using an instruction-tuned model (hence the "-it" in "gemma-3-1b-it"), we first need to format the fields accordingly::

    PROMPT_TEMPLATE = """<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"""
    COMPLETION_TEMPLATE = """{{"answer": "{}"}}<end_of_turn>"""

    prompt = PROMPT_TEMPLATE.format(dataset[0]["query"])
    completion = COMPLETION_TEMPLATE.format(dataset[0]["response"])

Tokenization for prompt–completion fine-tuning is a bit tricky. For the prompt, we need to add a ``<bos>`` (beginning-of-sequence) token at the start, while the completion requires an ``<eos>`` (end-of-sequence) token at the end. We also want to optimize the model only with respect to the completion, masking out the prompt tokens. Moreover, to make training more efficient, we should batch multiple prompt–completion pairs and pad sequences.

While this can be done with a plain tokenizer, *fabrique* provides a few utilities to simplify the process::

    from fabrique.tokenizer_utils import encode_batch_for_prompt_completion

    tokens, completion_mask = encode_batch_for_prompt_completion(
        tokenizer,
        prompts=[prompt],
        completions=[completion],
        pad_to_multiple_of=128,
    )
    print(tokens)
    # [[     2    105   2364    107   9322 ... 0      0      0      0]]
    # 0 is a PAD token

    print(completion_mask.astype(int))
    # [[0 0 0 0 ... 1 1 1 1 ... 0 0 0 0]]
    #   ^ ^ ^ ^     ^ ^ ^ ^     ^ ^ ^ ^
    #   prompt      compl'n     padding

Now that we’re satisfied with the tokenization, we can define our loss function::

    import jax
    import optax

    def loss_fn(model, tokens: jax.Array, completion_mask: jax.Array):
        # In prompt–completion prediction, the objective is to predict
        # the next token, so the labels are the same as the inputs,
        # but shifted by one token.
        inputs = tokens[:, :-1]    # [batch, seq_len-1]
        labels = tokens[:, 1:]     # [batch, seq_len-1]

        # Apply the model to the input tokens. The model recognizes the padding mask
        # from PAD tokens in the input and applies the causal mask automatically.
        # Note that we don’t pass the completion mask here because the model *should*
        # attend to prompt tokens during the forward pass.
        logits = model(inputs).logits  # [batch, seq_len-1, vocab_size]

        # However, the model *should not* be optimized with respect to prompt tokens,
        # so we apply the completion mask to the loss.
        mask = completion_mask[:, 1:]   # [batch, seq_len-1]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        return loss, logits


At each iteration of training, we will take a batch of data, calculate gradient of the loss function and update parameters using SGD optimizer::

    from flax import nnx

    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, tokens, completion_mask, optimizer: nnx.Optimizer):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grad = grad_fn(model, tokens, completion_mask)
        optimizer.update(model, grad)
        return loss


Training loop, although conceptually simple, may be quite verbose. To address it, *fabrique* provides a special :function:`train_iterator` class, which automatically handles loop boundaries, shows progress bar, etc. For example, the following code::

    from fabrique.training import train_iterator

    for batch, ts in train_iterator(dataset, max_epochs=10, max_steps=1000, batch_size=8):
        loss = train_step(...)


is roughly equivalent to the following double loop::

    step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(dataset.iter(batch_size=8)):
            loss = train_step(...)
            step += 1
            if step == max_steps:
                break

Of course, beyond providing a flat iterator, :function:`train_iterator` doesn't restrict the training loop in any way. You can still track metrics to your favorite tool, save checkpoints, or break the loop early.

Putting it all together, we get::

    import jax
    import optax
    from flax import nnx
    from datasets import load_dataset
    from fabrique.loading import load_model
    from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
    from fabrique.training import train_iterator


    PROMPT_TEMPLATE = """<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"""
    COMPLETION_TEMPLATE = """{{"answer": "{}"}}<end_of_turn>"""


    def loss_fn(model, tokens: jax.Array, completion_mask: jax.Array):
        # In prompt–completion prediction, the objective is to predict
        # the next token, so the labels are the same as the inputs,
        # but shifted by one token.
        inputs = tokens[:, :-1]    # [batch, seq_len-1]
        labels = tokens[:, 1:]     # [batch, seq_len-1]

        # Apply the model to the input tokens. The model recognizes the padding mask
        # from PAD tokens in the input and applies the causal mask automatically.
        # Note that we don’t pass the completion mask here because the model *should*
        # attend to prompt tokens during the forward pass.
        logits = model(inputs).logits  # [batch, seq_len-1, vocab_size]

        # However, the model *should not* be optimized with respect to prompt tokens,
        # so we apply the completion mask to the loss.
        mask = completion_mask[:, 1:]   # [batch, seq_len-1]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        return loss, logits


    @nnx.jit
    def train_step(model, tokens, completion_mask, optimizer: nnx.Optimizer):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grad = grad_fn(model, tokens, completion_mask)
        optimizer.update(model, grad)
        return loss

    tokenizer, model = load_model("gemma-3-1b-it")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)

    for batch in train_iterator(dataset, max_epochs=10, max_steps=1000, batch_size=1):
        prompts = [PROMPT_TEMPLATE.format(rec["query"]) for rec in batch]
        completions = [COMPLETION_TEMPLATE.format(rec["response"]) for rec in batch]
        tokens, completion_mask = encode_batch_for_prompt_completion(
            tokenizer, prompts, completions, pad_to_multiple_of=128,
        )
        loss = train_step(model, tokens, completion_mask, optimizer)


.. note::
   If this is the first time you train a model in JAX/Flax, system behavior
   might surprise you. Some iterations will be very quick, while others
   will take much longer. Most often this happens because of JIT compilation:
   ``jax.jit`` (and, inherently, ``nnx.jit``) re-compiles computation graph
   for every new shape of input arrays. For example, above we set ``pad_to_multiple_of=128``
   to align sequence lengths to multiple of 128 and thus reduce number of
   re-compilations. To persist compiled functions, you can enable `compilation cache`_.

.. _compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html#setting-cache-directory

Once training is done, we can instantiate a :class:`Sampler` a sample a completion.
Note that we only trained the model for 1000 steps and don't expect high-quality answers,
but at least the model now should follow the format (JSON with "answer" field)::

    from fabrique.sampling import Sampler

    sampler = Sampler(tokenizer, model)
    completion = sampler.sample(PROMPT_TEMPLATE.format(dataset[0]["query"]))
    # '{"answer": "The distance between two points ... "}'