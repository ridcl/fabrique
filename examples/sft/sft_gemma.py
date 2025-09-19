import math

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf  # for logging
from flax import nnx
from datasets import load_dataset, Dataset
from jinja2 import Environment

from fabrique import LLM, ChatMessage
from fabrique.models.gemma.load_rules import CHAT_TEMPLATE
from fabrique.models.gemma.sampler import Sampler, encode_batch
from fabrique.lora import LoRAEinsum


BATCH_SIZE = 1
TOTAL_STEPS = 1000
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 4096


summary_writer = tf.summary.create_file_writer("/tmp/tensorboard")

chat_template = Environment().from_string(CHAT_TEMPLATE)


def tokenize_batch(
    sampler: Sampler, batch: dict, pad_to_multiple_of=128, truncate=MAX_SEQ_LENGTH
):
    texts = []
    for img, question, answer in zip(batch["image"], batch["question"], batch["answer"]):
        text = chat_template.render(
            messages=[
                ChatMessage(role="user", content=img),
                ChatMessage(role="user", content=question),
                ChatMessage(role="assistant", content=answer),
            ]
        )
        texts.append(text)
    token_lists = [sampler.tokenizer.encode(text) for text in texts]
    max_length = max(len(token_list) for token_list in token_lists)
    # align to multiple of certain value to minimize re-compilation for every length
    max_length = math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of
    pad_token_id = sampler.tokenizer.special_tokens.PAD
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
    mask = batch["pad_mask"][:, 1:]
    # logits = model(inputs, attention_mask=mask)
    logits = model(inputs).logits

    mask = batch["pad_mask"][:, 1:]  # TODO: also mask out prompt
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = (loss * mask).sum() / mask.sum()  # ignore loss at padding
    return loss, logits



trainable = nnx.All(
    nnx.Param,
    nnx.Any(nnx.PathContains("lora_a"), nnx.PathContains("lora_b"))
)


@nnx.jit
def train_step(model, batch: dict, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, trainable))
    (loss, _), grad = grad_fn(model, batch)
    optimizer.update(model, grad)
    metrics.update(loss=loss)
    return loss


def train(sampler: Sampler, ds: Dataset):
    model = sampler.model
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=trainable)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    step = 0
    for epoch in range(NUM_EPOCHS):
        if step == TOTAL_STEPS:
            break
        metrics.reset()
        for i, orig_batch in enumerate(ds.iter(batch_size=BATCH_SIZE)):
            batch = tokenize_batch(sampler, orig_batch, truncate=2048)
            loss = train_step(model, batch, optimizer, metrics)
            print(
                f"Epoch {epoch}, step {step}: avg_loss = {metrics.compute()['loss'].item():.2f}; batch_loss = {loss.item():.2f}"
            )
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, metrics.compute()["loss"])
            step += 1
            if step == TOTAL_STEPS:
                print("Finished training!")
                break


def main():
    # TODO: train on VQA
    prompt = """<start_of_turn>user\nWrite a function to retrieve title of the Wikipedia's main page\n<start_of_turn>model\n"""
    device_arr = np.array(jax.devices())[None, :]
    # mesh = jax.sharding.Mesh(devices=device_arr, axis_names=("data", "model"))
    mesh = None
    sampler = Sampler.load_gemma("4b", mesh=mesh)
    model = sampler.model

    output_before_training = sampler.sample(prompt, max_length=512)

    rngs = nnx.Rngs(0)
    for i in range(len(model.blocks)):
        model.blocks[i].attn.q_einsum = LoRAEinsum(
            rank=16, base_module=model.blocks[i].attn.q_einsum, rngs=rngs
        )
        model.blocks[i].attn.kv_einsum = LoRAEinsum(
            rank=16, base_module=model.blocks[i].attn.kv_einsum, rngs=rngs
        )

    #  TODO: use full dataset
    ds = load_dataset("flaviagiammarino/vqa-rad", split="train[:200]")
    train(sampler, ds)

    output_after_training = sampler.sample(prompt, max_length=512)

    print("\033[94m" + output_before_training[:400] + "\033[0m")
    print("\033[92m" + output_after_training[:400] + "\033[0m")

