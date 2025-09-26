Training
========

Unlike many other libraries that provide predefined "trainers" for a fixed
set of ML problems, *fabrique* leaves complete control over training loop
to the user and instead provides utilities for the routine tasks.
Let's look at the example of VLM fine-tuning::

    from fabrique.training import TrainIterator, encode_batch_for_prompt_completion
    from fabrique.loading import load_model

    -- todo: adjust sampler import, add note on load_model
    tokenizer, model = load_model(1b)
    ...
    for batch in TrainIter(lambda: dataset.iter(), max_epochs=1):
    ...