import os
from multimethod import multimethod
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image
from datasets import Dataset

from fabrique.sampling import Sampler
from fabrique.lora import LoRAEinsum
from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
# from fabrique.training import TrainIterator


# ===================
# Dataset preparation
# ===================


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
    return Dataset.from_pandas(dataset)


# ========
# Training
# ========

BATCH_SIZE = 2
IMG_SHAPE = (896, 896)
MAX_STEPS = 100# 00
MAX_EPOCHS = 10
MAX_SEQ_LENGTH = 4096

COLORS = [
    '\033[95m',
    '\033[94m',
    '\033[96m',
    '\033[92m',
    '\033[93m',
    '\033[91m',
]
ENDC = '\033[0m'



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


def show_batch(sampler, batch):
    for i in range(len(batch["screenshot"])):
        image_path, question, answer = batch["screenshot"][i], batch["concept"][i], batch["content"][i]
        image = Image.open(image_path)
        out = sampler.sample(PROMPT_TEMPLATE.format(question), images=[image])
        color = COLORS[i % len(COLORS)]
        print(f"{color}example {i}: expected = {answer}; actual = {out}{ENDC}")




def load_sampler_and_update_lora():
    device_arr = np.array(jax.devices())[None, :]
    mesh = jax.sharding.Mesh(devices=device_arr, axis_names=("data", "model"))
    sampler = Sampler.load_model("gemma-3-4b-it", mesh=mesh)
    model = sampler.model

    apply_lora(model, rank=64, rngs=nnx.Rngs(0))
    # for i in range(len(model.blocks)):
    #     model.blocks[i].attn.q_einsum = LoRAEinsum(
    #         rank=lora_rank, base_module=model.blocks[i].attn.q_einsum, rngs=rngs
    #     )
    #     model.blocks[i].attn.kv_einsum = LoRAEinsum(
    #         rank=lora_rank, base_module=model.blocks[i].attn.kv_einsum, rngs=rngs
    #     )
    return sampler



from flax.nnx.filterlib import Filter, OfType, Any as AnyOf


@multimethod
def wrap_in_lora(base_module: nnx.Einsum, rank: int, *, rngs: nnx.Rngs):
    return LoRAEinsum(rank=rank, base_module=base_module, rngs=rngs)

# TODO: add methods for other LoRA layers


LORA_COMPATIBLE = AnyOf(OfType(nnx.Einsum))
LORA_MODULE = AnyOf(OfType(LoRAEinsum))


def apply_lora(root: nnx.Module, rank: int, filter: Filter = LORA_COMPATIBLE, *, rngs: nnx.Rngs):
    matching = []  # list of (parent_module, lora_compatible_attr_name)
    for path, module in root.iter_modules():
        for attr_name, child in module.iter_children():
            # if child passes filter and is not LoRA module yet
            if filter(path, child) and not LORA_MODULE(path, child):
                matching.append((module, attr_name))
    for module, attr_name in matching:
        base_module = getattr(module, attr_name)
        lora_module = wrap_in_lora(base_module, rank, rngs=rngs)
        setattr(module, attr_name, lora_module)


def merge_lora(root: nnx.Module):
    raise NotImplementedError("Merging LoRA parameters is not implemented yet")
    for path, module in root.iter_modules():
        for attr_name, child in module.iter_children():
            # if child passes filter and is not LoRA module yet
            if LORA_MODULE(path, child):
                base_module, adapter = child.base_module, child.adapter
                # TODO: this doesn't work for Einsum. Instead, we
                # need smth like (assuming lora_einsum_str = 'BTD,Dr,rNH->BTNH')
                # adapter_kernel = jnp.einsum("Dr,rNH->NDH")
                # base_module.kernel += adapter.lora_a @ adapter.lora_b
                setattr(module, attr_name, base_module)



def main():
    dataset = create_dataset()
    sampler = load_sampler_and_update_lora()

    batch = next(dataset.iter(batch_size=8))
    # check output before training
    show_batch(sampler, batch)

    train(sampler, dataset)

    # check output after training
    # now it should follow the format in the training set
    show_batch(sampler, batch)

    # checkpoints
    # ckpt_path = os.path.abspath("output/vlm-xbrl.ckpt")  # Oct 4, 16:30
    ckpt_path = os.path.abspath("output/vlm-xbrl-lora.ckpt")
    checkpointer = ocp.StandardCheckpointer()

    # save
    graphdef, lora_state, other_state = nnx.split(model, trainable, ...)
    checkpointer.save(ckpt_path, lora_state)

    # load
    graphdef, lora_state, other_state = nnx.split(sampler.model, trainable, ...)
    # abstract_state = jax.tree_util.tree_map(
    #     ocp.utils.to_shape_dtype_struct, state
    # )
    loaded_state = checkpointer.restore(
        ckpt_path,
        lora_state
    )
    model = nnx.merge(graphdef, loaded_state, other_state)
    sampler.model = model
