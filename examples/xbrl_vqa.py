import os

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
from datasets import Dataset
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image

from fabrique import lora
from fabrique.models.gemma.modeling import Transformer
from fabrique.sampling import Sampler
from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
from fabrique.rouge import rouge_n, rouge_l

# from fabrique.training import TrainIterator


# =====================
# Dataset preparation
# =====================


def create_dataset():
    path = os.path.expanduser("output/vqa/dataset_1k.parquet")
    screenshot_dir = os.path.dirname(path) + "/screenshots"
    df = pd.read_parquet(path)
    df["image_path"] = df.image_path.apply(
        lambda p: p.replace("/ml-datasets/xbrl_facts/screenshots", screenshot_dir)
    )
    df = df[df.apply(lambda row: os.path.exists(row.image_path), axis=1)]

    # since this is SFT, take only good evidence
    df = df[df.is_good]

    assert df.shape[0] > 0
    return Dataset.from_pandas(df).train_test_split(test_size=0.1)


# ===========
# Training
# ===========


BATCH_SIZE = 1
IMG_SHAPE = (896, 896)
MAX_STEPS = 10000
MAX_EPOCHS = 10
MAX_SEQ_LENGTH = 4096


PROMPT_TEMPLATE = """
<start_of_turn>user
Given a page image and its <MARKDOWN>, extract evidence for the <QUESTION>
<start_of_image>
<MARKDOWN>
{markdown}
</MARKDOWN>
<QUESTION>
{question}
<QUESTION>
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


def train(sampler: Sampler, trainset: Dataset, testset: Dataset):
    tokenizer = sampler.tokenizer
    model = sampler.model
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=trainable)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    gen_metrics, _ = calc_gen_metrics(sampler, testset)
    print(f"!!! Before training: gen_metrics = {gen_metrics}")
    step = 0
    for epoch in range(MAX_EPOCHS):
        if step == MAX_STEPS:
            break
        metrics.reset()
        for i, batch in enumerate(trainset.iter(batch_size=BATCH_SIZE)):
            image_paths, markdowns, questions, answers = (
                batch["image_path"],
                batch["markdown"],
                batch["question"],
                batch["evidence"],
            )
            images = [Image.open(path) for path in image_paths]
            prompts = [PROMPT_TEMPLATE.format(question=q, markdown=md) for q, md in zip(questions, markdowns)]
            completions = [COMPLETION_TEMPLATE.format(a) for a in answers]
            tokens, completion_mask = encode_batch_for_prompt_completion(
                tokenizer, prompts, completions, pad_to_multiple_of=32
            )
            # array of size (B N H W C), where N=1 - number of images per prompt
            images = jnp.stack([jnp.array(img.resize(IMG_SHAPE)) for img in images])[
                :, None, ...
            ]
            loss = train_step(
                model, tokens, images, completion_mask, optimizer, metrics
            )
            print(
                f"Epoch {epoch}, step {step}: avg_loss = {metrics.compute()['loss'].item():.2f}; batch_loss = {loss.item():.2f}"
            )
            step += 1
            if step == MAX_STEPS:
                print("Finished training!")
                break
            if step % 200 == 0:
                gen_metrics, _ = calc_gen_metrics(sampler, testset)
                print(f"!!! Epoch {epoch}, step {step}: gen_metrics = {gen_metrics}")

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
    loaded_state = checkpointer.restore(ckpt_path, lora_state)
    model = nnx.merge(graphdef, loaded_state, other_state)
    return model


# ===============
# Visualization
# ===============

COLORS = [
    "\033[95m",
    "\033[94m",
    "\033[96m",
    "\033[92m",
    "\033[93m",
    "\033[91m",
]
ENDC = "\033[0m"


def show_batch(sampler: Sampler, batch):
    for i in range(len(batch["image_path"])):
        image_path, markdown, question, answer = (
            batch["image_path"][i],
            batch["markdown"][i],
            batch["question"][i],
            batch["evidence"][i],
        )
        image = Image.open(image_path).resize(IMG_SHAPE)
        out = sampler.sample(PROMPT_TEMPLATE.format(question=question, markdown=markdown), images=[image])
        color = COLORS[i % len(COLORS)]
        print(
            f"""\n{color}-------------- example {i} -------------
            EXPECTED: {answer}
            ACTUAL: {out}{ENDC}
            """
        )


def show_metrics(sampler: Sampler, batch):
    metrics = []
    for i in range(len(batch["image_path"])):
        image_path, markdown, question, answer = (
            batch["image_path"][i],
            batch["markdown"][i],
            batch["question"][i],
            batch["evidence"][i],
        )
        image = Image.open(image_path).resize(IMG_SHAPE)
        out = sampler.sample(PROMPT_TEMPLATE.format(question=question, markdown=markdown), images=[image])
        ans_tokens = sampler.tokenizer.encode(answer)
        out_tokens = sampler.tokenizer.encode(out)
        print("-" * 20 + f" {i:03} " + "-" * 20)
        print("ROUGE-1:", rouge_n(ans_tokens, out_tokens, n=1))
        print("ROUGE-2:", rouge_n(ans_tokens, out_tokens, n=2))
        print("ROUGE-3:", rouge_n(ans_tokens, out_tokens, n=3))
        print("ROUGE-L:", rouge_l(ans_tokens, out_tokens))
        print(f"""EXPECTED: {answer[:100]}\nACTUAL: {out[:100]}""")
        # usign ROUGE-L as the main metric
        metrics.append(rouge_l(ans_tokens, out_tokens))
    avg_precision = sum([m["precision"] for m in metrics]) / len(batch)
    avg_recall = sum([m["recall"] for m in metrics]) / len(batch)
    avg_f1 = sum([m["f1"] for m in metrics]) / len(batch)



def calc_gen_metrics(sampler: Sampler, testset) -> dict[str, float]:
    from tqdm import tqdm
    metrics = []
    answers = []
    output = []
    for batch in tqdm(testset.iter(batch_size=BATCH_SIZE), total=len(testset) // BATCH_SIZE):
        for i in range(len(batch["image_path"])):
            image_path, markdown, question, answer = (
                batch["image_path"][i],
                batch["markdown"][i],
                batch["question"][i],
                batch["evidence"][i],
            )
            image = Image.open(image_path).resize(IMG_SHAPE)
            out = sampler.sample(PROMPT_TEMPLATE.format(question=question, markdown=markdown), images=[image])
            ans_tokens = sampler.tokenizer.encode(answer)
            out_tokens = sampler.tokenizer.encode(out)
            # print("-" * 20 + f" {i:03} " + "-" * 20)
            # print("ROUGE-1:", rouge_n(ans_tokens, out_tokens, n=1))
            # print("ROUGE-2:", rouge_n(ans_tokens, out_tokens, n=2))
            # print("ROUGE-3:", rouge_n(ans_tokens, out_tokens, n=3))
            # print("ROUGE-L:", rouge_l(ans_tokens, out_tokens))
            # print(f"""EXPECTED: {answer[:100]}\nACTUAL: {out[:100]}""")
            # usign ROUGE-L as the main metric
            metrics.append(rouge_l(ans_tokens, out_tokens))
            answers.append(answer)
            output.append(out)
    avg_metrics = {
        "precision": sum([m["precision"] for m in metrics]) / len(testset),
        "recall": sum([m["recall"] for m in metrics]) / len(testset),
        "f1": sum([m["f1"] for m in metrics]) / len(testset)
    }
    details = {
        "answers": answers,
        "output": output,
    }
    return avg_metrics, details


# ===========
# Main
# ===========


def main(training=False, ckpt_path: str = "output/vqa/lora_v2.ckpt"):
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    # jax.config.update("jax_explain_cache_misses", True)

    if training and os.path.exists(ckpt_path):
        raise ValueError(
            f"You asked to train, but the checkpoint path {ckpt_path} "
            "already exists. If you meant to sample pretrained model, use train=False. "
            + "Otherwise, specify a different checkpoint path"
        )

    dataset = create_dataset()
    trainset, testset = dataset["train"], dataset["test"]

    mesh = jax.make_mesh((1, len(jax.devices())), ("data", "model"))
    sampler = Sampler.load_model("gemma-3-4b-it", mesh=mesh)
    model = sampler.model
    model.vision_encoder.rngs = nnx.Rngs(0)  # hack to work around NNX <> Linen interop

    lora_sharding = NamedSharding(mesh, P())
    lora.apply(model, rank=64, sharding=lora_sharding, rngs=nnx.Rngs(0))

    # TODO: check sharding, investigate why GPUs are not loaded equally

    # TODO: evaluate similarity using ROUGE

    if training:
        # # check output before training
        # batch = next(trainset.iter(batch_size=8))
        # show_batch(sampler, batch)

        train(sampler, trainset, testset)
        save_lora(sampler.model, ckpt_path)
    else:
        sampler.model = load_lora(sampler.model, ckpt_path)

    # check output after training
    # now it should follow the format in the training set
    show_metrics(sampler, next(testset.iter(8)))
