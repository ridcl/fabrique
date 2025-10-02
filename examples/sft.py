from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf  # for logging
from flax import nnx
from datasets import load_dataset, Dataset

from fabrique.sampling import Sampler
from fabrique.lora import LoRAEinsum
from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
# from fabrique.training import TrainIterator


BATCH_SIZE = 2
IMG_SHAPE = (896, 896)
MAX_STEPS = 1000
MAX_EPOCHS = 10
MAX_SEQ_LENGTH = 4096

PROMPT_TEMPLATE = """<start_of_turn>user\n<start_of_image>{}<end_of_turn>\n<start_of_turn>model\n"""
COMPLETION_TEMPLATE = """{{"answer": "{}"}}<end_of_turn>"""


summary_writer = tf.summary.create_file_writer("/tmp/tensorboard")



# def encode_batch(
#     tokenizer,
#     prompts: list[str],
#     completions: list[str],
#     pad_to_multiple_of: int | None = None,
#     truncate: int | None = None,
# ) -> Tuple[jax.Array, jax.Array]:
#     """
#     Encode batched prompts and completions with completion masking.

#     Args:
#         prompts: List of prompt strings
#         completions: List of completion strings (same length as prompts)

#     Returns:
#         Tuple of:
#         - tokenized_sequences: JAX array of shape (batch_size, max_seq_len) with token IDs
#         - completion_mask: JAX array of shape (batch_size, max_seq_len) with boolean mask
#                           (True for completion tokens, False for prompt/padding tokens)
#     """
#     assert len(prompts) == len(completions), "Prompts and completions must have same length"

#     # batch_size = len(prompts)
#     tokenized_sequences = []
#     completion_masks = []

#     # Encode each prompt + completion pair
#     for prompt, completion in zip(prompts, completions):
#         # Tokenize prompt and completion separately
#         prompt_tokens = tokenizer.encode(prompt, add_bos=True)
#         completion_tokens = tokenizer.encode(completion, add_eos=True)

#         # Combine tokens
#         full_sequence = prompt_tokens + completion_tokens

#         # Create completion mask (False for prompt tokens, True for completion tokens)
#         mask = [False] * len(prompt_tokens) + [True] * len(completion_tokens)

#         tokenized_sequences.append(full_sequence)
#         completion_masks.append(mask)

#     # Find maximum sequence length for padding
#     max_len = max(len(seq) for seq in tokenized_sequences)
#     if pad_to_multiple_of:
#         # align to multiple of certain value to minimize re-compilation for every length
#         max_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of

#     # Pad sequences and masks
#     padded_sequences = []
#     padded_masks = []

#     for seq, mask in zip(tokenized_sequences, completion_masks):
#         # Pad sequence with PAD tokens
#         padding_length = max_len - len(seq)
#         padded_seq = seq + [tokenizer.special_tokens.PAD] * padding_length

#         # Pad mask with False (padding tokens are not completion tokens)
#         padded_mask = mask + [False] * padding_length

#         padded_sequences.append(padded_seq)
#         padded_masks.append(padded_mask)

#     if truncate:
#         padded_sequences = [seq[:truncate] for seq in padded_sequences]
#         padded_masks = [seq[:truncate] for seq in padded_sequences]

#     # Convert to JAX arrays
#     tokenized_sequences = jnp.array(padded_sequences, dtype=jnp.int32)
#     completion_mask = jnp.array(padded_masks, dtype=jnp.bool_)

#     return tokenized_sequences, completion_mask


def loss_fn(model, tokens: jax.Array, images: jax.Array, completion_mask: jax.Array):
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]    # same tokens, but shifted by one
    # note: the model knows about padding via PAD tokens in the input
    logits = model(inputs, images=images).logits

    mask = completion_mask[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = (loss * mask).sum() / mask.sum()  # ignore loss at padding
    return loss, logits



trainable = nnx.All(
    nnx.Param,
    nnx.Any(nnx.PathContains("lora_a"), nnx.PathContains("lora_b"))
)


@nnx.jit
def train_step(model, tokens, images, completion_mask, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, trainable))
    (loss, _), grad = grad_fn(model, tokens, images, completion_mask)
    optimizer.update(model, grad)
    metrics.update(loss=loss)
    return loss


def train(sampler: Sampler, dataset: Dataset):
    tokenizer = sampler.tokenizer
    model = sampler.model
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=trainable)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    # ti = TrainIterator(dataset, max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS)
    step = 0
    for epoch in range(MAX_EPOCHS):
        if step == MAX_STEPS:
            break
        metrics.reset()
        for i, batch in enumerate(dataset.iter(batch_size=BATCH_SIZE)):
            images, questions, answers = batch["image"], batch["question"], batch["answer"]
            prompts = [PROMPT_TEMPLATE.format(q) for q in questions]
            completions = [COMPLETION_TEMPLATE.format(a) for a in answers]
            tokens, completion_mask = encode_batch_for_prompt_completion(
                tokenizer, prompts, completions, pad_to_multiple_of=32
            )
            # array of size (B N H W C), where N=1 - number of images per prompt
            images = jnp.stack([jnp.array(img.resize(IMG_SHAPE)) for img in images])[:, None, ...]
            loss = train_step(model, tokens, images, completion_mask, optimizer, metrics)
            print(
                f"Epoch {epoch}, step {step}: avg_loss = {metrics.compute()['loss'].item():.2f}; batch_loss = {loss.item():.2f}"
            )
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, metrics.compute()["loss"])
            step += 1
            if step == MAX_STEPS:
                print("Finished training!")
                break



COLORS = [
    '\033[95m',
    '\033[94m',
    '\033[96m',
    '\033[92m',
    '\033[93m',
    '\033[91m',
]
ENDC = '\033[0m'


def show_batch(sampler, batch):
    for i in range(len(batch["question"])):
        image, question, answer = batch["image"][i], batch["question"][i], batch["answer"][i]
        out = sampler.sample(PROMPT_TEMPLATE.format(question), images=[image])
        color = COLORS[i % len(COLORS)]
        print(color + f"example {i}: " + out + ENDC)


def main():
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
    device_arr = np.array(jax.devices())[None, :]
    mesh = jax.sharding.Mesh(devices=device_arr, axis_names=("data", "model"))
    sampler = Sampler.load_model("gemma-3-4b-it", mesh=mesh)
    model = sampler.model

    rngs = nnx.Rngs(0)
    for i in range(len(model.blocks)):
        model.blocks[i].attn.q_einsum = LoRAEinsum(
            rank=16, base_module=model.blocks[i].attn.q_einsum, rngs=rngs
        )
        model.blocks[i].attn.kv_einsum = LoRAEinsum(
            rank=16, base_module=model.blocks[i].attn.kv_einsum, rngs=rngs
        )

    batch = next(dataset.iter(batch_size=8))
    # check output before training
    show_batch(sampler, batch)

    train(sampler, dataset)

    # check output after training
    # now it should follow the format in the training set
    show_batch(sampler, batch)