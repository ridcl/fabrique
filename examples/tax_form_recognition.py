from dataclasses import dataclass
import os
import random
import json
from datetime import datetime


import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from datasets import Dataset, load_dataset
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image

from fabrique import lora
from fabrique.sampling import Sampler
from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
from fabrique.export import to_huggingface
from fabrique.training import train_iterator


def _metrics(expected: str, actual: str) -> dict[str, float]:
    expected_dict = json.loads(expected)["gt_parse"]
    expected_kv = set(expected_dict.items())
    try:
        actual_dict = json.loads(actual)["gt_parse"]
        actual_kv = set(actual_dict.items())
    except:
        actual_kv = set([])
    tp = len(expected_kv & actual_kv)
    fp = len(actual_kv - expected_kv)
    fn = len(expected_kv - actual_kv)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn)
    }


def evaluate(sampler: Sampler, examples) -> dict[str, float]:
    metrics = []
    output = []
    for example in tqdm(examples):
        image, ground_truth = example["image"], example["ground_truth"]
        out = sampler.sample(PROMPT_TEMPLATE, images=[image])
        metrics.append(_metrics(ground_truth, out))
        output.append(out)

    avg_metrics = {
        "precision": sum([m["precision"] for m in metrics]) / len(examples),
        "recall": sum([m["recall"] for m in metrics]) / len(examples),
        "f1": sum([m["f1"] for m in metrics]) / len(examples),
    }
    return avg_metrics, output


# =============================
# Dataset preparation
# =============================


def create_dataset():
    return load_dataset("singhsays/fake-w2-us-tax-form-dataset")


# =====================
# Training
# =====================


BATCH_SIZE = 1
IMG_SHAPE = (896, 896)
MAX_STEPS = 10000
MAX_EPOCHS = 10
MAX_SEQ_LENGTH = 4096


PROMPT_TEMPLATE = """
<start_of_turn>user
Extract form data from the image
<start_of_image>
<end_of_turn>
<start_of_turn>model
"""
COMPLETION_TEMPLATE = """{}<end_of_turn>"""


def loss_fn(model, tokens: jax.Array, images: jax.Array, completion_mask: jax.Array):
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]  # same tokens, but shifted by one
    # note: the model knows about padding via PAD tokens in the input
    logits = model(inputs, images=images).logits

    mask = completion_mask[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = (loss * mask).sum() / mask.sum()  # ignore loss at padding
    return loss, logits


trainable = lora.ALL_LORA_PARAMS


@nnx.jit
def train_step(
    model,
    tokens,
    images,
    completion_mask,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
):
    grad_fn = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=nnx.DiffState(0, trainable)
    )
    (loss, _), grad = grad_fn(model, tokens, images, completion_mask)
    optimizer.update(model, grad)
    metrics.update(loss=loss)
    return loss


def _show_progress(sampler, trainset, testset):
    print("----------------------- train metrics ---------------------")
    train_metrics, out = evaluate(sampler, trainset.select(range(10)))
    idx = random.randint(0, len(out) - 1)
    print(f"\n\t{'\n\t'.join(k + ': ' + str(v) for k, v in train_metrics.items())}")
    print(f"Example output: {out[idx]}")
    print("----------------------- test metrics ---------------------")
    test_metrics, out = evaluate(sampler, testset.select(range(10)))
    idx = random.randint(0, len(out) - 1)
    print(f"\n\t{'\n\t'.join(k + ': ' + str(v) for k, v in test_metrics.items())}")
    # print(f"Example output: {out[idx]}")
    expected_dict = json.loads(testset[idx]["ground_truth"])["gt_parse"]
    try:
        actual_dict = json.loads(out[idx])["gt_parse"]
    except:
        actual_dict = {}
    end_color = "\033[0m"
    for k, ev in expected_dict.items():
        av = actual_dict.get(k)
        color = "\033[92m" if av == ev else "\x1b[31m"
        print(f"{color}{k}: {ev} <> {av}{end_color}")



def train(sampler: Sampler, trainset: Dataset, testset: Dataset, ckpt_base_path: str):
    tokenizer = sampler.tokenizer
    model = sampler.model
    if ckpt_path := lora.latest_checkpoint_path(ckpt_base_path):
        print(f"Loading LoRA checkpoint from {ckpt_path}")
        model = lora.load(model, ckpt_path)
        sampler.model = model

    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=trainable)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    test_metrics, _ = evaluate(sampler, testset.select(range(10)))
    print(f"Metrics:\n\t{'\n\t'.join(k + ': ' + str(v) for k, v in test_metrics.items())}")

    for batch, ts in train_iterator(trainset, batch_size=BATCH_SIZE, max_steps=MAX_STEPS, max_epochs=MAX_EPOCHS):
        if ts.new_epoch:
            metrics.reset()
        images, ground_truth = batch["image"], batch["ground_truth"]
        prompts = [PROMPT_TEMPLATE for _ in range(len(images))]
        completions = [COMPLETION_TEMPLATE.format(gt) for gt in ground_truth]
        tokens, completion_mask = encode_batch_for_prompt_completion(
            tokenizer, prompts, completions, pad_to_multiple_of=64
        )
        # array of size (B N H W C), where N=1 - number of images per prompt
        images = [jnp.array(img.resize(IMG_SHAPE)) for img in images]
        images = jnp.stack(images)[:, None, ...]
        loss = train_step(
            model, tokens, images, completion_mask, optimizer, metrics
        )
        print(
            f"Epoch {ts.epoch}, step {ts.step}: avg_loss = {metrics.compute()['loss'].item():.2f}; batch_loss = {loss.item():.2f}"
        )
        if ts.step == MAX_STEPS:
            _show_progress(sampler, trainset, testset)
            print("Finished training!")
            break
        if ts.step % 250 == 0 and ts.step != 0:
            _show_progress(sampler, trainset, testset)
            ckpt_path = os.path.join(
                f"{ckpt_base_path}/{datetime.now().strftime('%Y-%m_%H-%M-%S')}.ckpt"
            )
            print(f"Saving LoRA checkpoint to {ckpt_path}")
            lora.save(model, ckpt_path)



# ===========
# Main
# ===========


def main(ckpt_base_path: str = "output/tax_forms"):
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    # jax.config.update("jax_explain_cache_misses", True)

    dataset = create_dataset()
    trainset, testset = dataset["train"], dataset["test"]

    mesh = jax.make_mesh((1, len(jax.devices())), ("data", "model"))
    sampler = Sampler.load_model("gemma-3-4b-it", mesh=mesh)

    lora_sharding = NamedSharding(mesh, P())
    lora.apply(sampler.model, rank=64, sharding=lora_sharding, rngs=nnx.Rngs(0))

    train(sampler, trainset, testset, ckpt_base_path)
    lora.merge(sampler.model)
    to_huggingface(sampler.model, "google/gemma-3-4b-it", "output/gemma-3-4b-tax-forms")
