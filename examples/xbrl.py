import os
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
from PIL import Image
from datasets import Dataset

from fabrique import lora
from fabrique.models.gemma.modeling import Transformer
from fabrique.sampling import Sampler
from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
# from fabrique.training import TrainIterator


# =====================
# Dataset preparation
# =====================


def get_top_concepts(df, n):
    counts = df.groupby("concept").concept.count()
    return counts.sort_values(ascending=False).head(n).index.values


def screenshot_path(screenshot_dir, row):
    return f"{screenshot_dir}/{row.report_id}/page_{row.page:03}.png"


def create_dataset():
    path = os.path.expanduser("output/ml-datasets/xbrl_facts_filtered.parquet")
    screenshot_dir = os.path.dirname(path) + "/screenshots"
    df = pd.read_parquet(path)
    # collect top concepts
    top_concepts = get_top_concepts(df, 20)
    # filter out rows with other concepts
    df = df[df.concept.isin(top_concepts)]
    # add screenshots and filter out invalid rows
    df["screenshot"] = df.apply(lambda row: screenshot_path(screenshot_dir, row), axis=1)
    df = df[df.apply(lambda row: os.path.exists(row.screenshot), axis=1)]

    # collect target dataset
    gb = df.groupby(["concept", "screenshot"])["content"]
    dataset = gb.agg(lambda x: list(set(x))).reset_index()

    assert dataset.shape[0] > 0
    return Dataset.from_pandas(dataset).train_test_split(test_size=0.1)


# ===========
# Training
# ===========


BATCH_SIZE = 2
IMG_SHAPE = (896, 896)
MAX_STEPS = 10000
MAX_EPOCHS = 10
MAX_SEQ_LENGTH = 4096


PROMPT_TEMPLATE = """<start_of_turn>user\n<start_of_image>Extract values of the following IFRS concept: {}<end_of_turn>\n<start_of_turn>model\n"""
COMPLETION_TEMPLATE = """{}<end_of_turn>"""


def loss_fn(model, tokens: jax.Array, images: jax.Array, completion_mask: jax.Array):
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]    # same tokens, but shifted by one
    # note: the model knows about padding via PAD tokens in the input
    logits = model(inputs, images=images).logits

    mask = completion_mask[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = (loss * mask).sum() / mask.sum()  # ignore loss at padding
    return loss, logits



trainable = lora.ALL_LORA_PARAMS


@nnx.jit
def train_step(model, tokens, images, completion_mask, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, trainable))
    (loss, _), grad = grad_fn(model, tokens, images, completion_mask)
    optimizer.update(model, grad)
    metrics.update(loss=loss)
    return loss


def train(sampler: Sampler, dataset: list[dict]):
    tokenizer = sampler.tokenizer
    model = sampler.model
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=trainable)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    step = 0
    for epoch in range(MAX_EPOCHS):
        if step == MAX_STEPS:
            break
        metrics.reset()
        for i, batch in enumerate(dataset.iter(batch_size=BATCH_SIZE)):
            image_paths, questions, answers = batch["screenshot"], batch["concept"], batch["content"]
            images = [Image.open(path) for path in image_paths]
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
            step += 1
            if step == MAX_STEPS:
                print("Finished training!")
                break



# ==============
# Save/Load
# ==============

# TODO: move to lora module, adding arg filter = lora.ALL_LORA_PARAMS
def save_lora(model, ckpt_path: str):
    ckpt_path = os.path.abspath(ckpt_path)
    checkpointer = ocp.StandardCheckpointer()
    _graphdef, lora_state, _other_state = nnx.split(model, trainable, ...)
    checkpointer.save(ckpt_path, lora_state)


def load_lora(model, ckpt_path: str) -> Transformer:
    ckpt_path = os.path.abspath(ckpt_path)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, lora_state, other_state = nnx.split(model, trainable, ...)
    loaded_state = checkpointer.restore(
        ckpt_path,
        lora_state
    )
    model = nnx.merge(graphdef, loaded_state, other_state)
    return model


# ===============
# Visualization
# ===============

COLORS = [
    '\033[95m',
    '\033[94m',
    '\033[96m',
    '\033[92m',
    '\033[93m',
    '\033[91m',
]
ENDC = '\033[0m'



def show_batch(sampler: Sampler, batch):
    for i in range(len(batch["screenshot"])):
        image_path, question, answer = batch["screenshot"][i], batch["concept"][i], batch["content"][i]
        image = Image.open(image_path)
        out = sampler.sample(PROMPT_TEMPLATE.format(question), images=[image])
        color = COLORS[i % len(COLORS)]
        print(f"{color}example {i}: expected = {answer}; actual = {out}{ENDC}")


# ===========
# Main
# ===========


def main(training=False, ckpt_path: str = "output/vlm-xbrl-lora.ckpt"):
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    jax.config.update("jax_explain_cache_misses", True)

    if training and os.path.exists(ckpt_path):
        raise ValueError(
            f"You asked to train, but the checkpoint path {ckpt_path} "
            "already exists. If you meant to sample pretrained model, use train=False. " +
            "Otherwise, specify a different checkpoint path"
        )

    dataset = create_dataset()
    trainset, testset = dataset["train"], dataset["test"]

    mesh = jax.make_mesh((1, len(jax.devices())), ("data", "model"))
    sampler = Sampler.load_model("gemma-3-4b-it", mesh=mesh)
    model = sampler.model

    lora_sharding = NamedSharding(mesh, P())
    lora.apply(model, rank=64, sharding=lora_sharding, rngs=nnx.Rngs(0))

    if training:
        # check output before training
        batch = next(trainset.iter(batch_size=8))
        show_batch(sampler, batch)

        training(sampler, trainset)
        save_lora(sampler.model, ckpt_path)
    else:
        sampler.model = load_lora(sampler.model, ckpt_path)

    # check output after training
    # now it should follow the format in the training set
    show_batch(sampler, next(testset.iter(8)))