import math

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf  # for logging
from flax import nnx

from fabrique import LLM, ChatMessage

BATCH_SIZE = 2
TOTAL_STEPS = 1000
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 1024


summary_writer = tf.summary.create_file_writer("/tmp/tensorboard")


def tokenize_batch(
    llm: LLM, batch: dict, pad_to_multiple_of=128, truncate=MAX_SEQ_LENGTH
):
    texts = []
    for sys, inst, outp in zip(batch["system"], batch["instruction"], batch["output"]):
        text = llm.apply_chat_template(
            [
                ChatMessage(role="system", content=sys),
                ChatMessage(role="user", content=inst),
                ChatMessage(role="assistant", content=outp),
            ]
        )
        texts.append(text)
    token_lists = [enc.ids for enc in llm.tokenizer.encode_batch(texts)]
    max_length = max(len(token_list) for token_list in token_lists)
    # align to multiple of certain value to minimize re-compilation for every length
    max_length = math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of
    pad_token_id = llm.special_tokens.pad_id or llm.special_tokens.eos_id
    token_lists = [
        token_list + [pad_token_id] * (max_length - len(token_list))
        for token_list in token_lists
    ]
    token_lists = [token_list[:truncate] for token_list in token_lists]
    tokens = jnp.array(token_lists)
    return {"tokens": tokens, "pad_mask": tokens != pad_token_id}


def loss_fn(model, batch: dict):
    tokens = batch["tokens"]
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    logits = model(inputs, 0)
    mask = batch["pad_mask"][:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = (loss * mask).sum() / mask.sum()  # ignore loss at padding
    return loss, logits


@nnx.jit
def train_step(model, batch: dict, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grad = grad_fn(model, batch)
    optimizer.update(grad)
    metrics.update(loss=loss)
    return loss


def train(llm: LLM, ds: datasets.Dataset):
    model = llm.model
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    step = 0
    for epoch in range(NUM_EPOCHS):
        if step == TOTAL_STEPS:
            break
        metrics.reset()
        for i, orig_batch in enumerate(ds.iter(batch_size=BATCH_SIZE)):
            batch = tokenize_batch(llm, orig_batch)
            loss = train_step(model, batch, optimizer, metrics)
            print(
                f"Epoch {epoch}, step {step}: avg_loss = {metrics.compute()['loss']:.2f}; batch_loss = {loss:.2f}"
            )
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, metrics.compute()["loss"])
            step += 1
            if step == TOTAL_STEPS:
                print("Finished training!")
                break


def main():
    example_chat = [
        ChatMessage(role="system", content="You are a professinal Python programmer."),
        ChatMessage(
            role="user",
            content="Write a function to retrieve title of the Wikipedia's main page",
        ),
    ]
    llm = LLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LENGTH,
        dtype=jnp.bfloat16,
        use_cache=True,
    )
    output_before_training = llm.generate(example_chat, max_length=512)

    ds = datasets.load_dataset("jtatman/python-code-dataset-500k")["train"]
    train(llm, ds)

    output_after_training = llm.generate(example_chat, max_length=512)
    print(f"-" * 30 + " before training " + "-" * 30)
    print(output_before_training.content)
    print(f"-" * 30 + " after training " + "-" * 30)
    print(output_after_training.content)
