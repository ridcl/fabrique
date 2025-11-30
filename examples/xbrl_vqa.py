import os
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from datasets import Dataset
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image

from fabrique import lora
from fabrique.sampling import Sampler
from fabrique.tokenizer_utils import encode_batch_for_prompt_completion
from fabrique.export import to_huggingface
from examples.rouge import rouge_n, rouge_l

# from fabrique.training import TrainIterator


# ===============================
# Interactive helpers
# ===============================

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
        out = sampler.sample(
            PROMPT_TEMPLATE.format(question=question, markdown=markdown), images=[image]
        )
        color = COLORS[i % len(COLORS)]
        print(
            f"""\n{color}-------------- example {i} -------------
            EXPECTED: {answer}

            <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            ACTUAL: {out}{ENDC}
            """
        )


def calc_gen_metrics(sampler: Sampler, testset) -> dict[str, float]:
    from tqdm import tqdm

    metrics = []
    answers = []
    output = []
    testset = testset.select(range(10))  # hack for faster metric calculation
    for batch in tqdm(
        testset.iter(batch_size=BATCH_SIZE), total=len(testset) // BATCH_SIZE
    ):
        for i in range(len(batch["image_path"])):
            image_path, markdown, question, answer = (
                batch["image_path"][i],
                batch["markdown"][i],
                batch["question"][i],
                batch["evidence"][i],
            )
            image = Image.open(image_path).resize(IMG_SHAPE)
            out = sampler.sample(
                PROMPT_TEMPLATE.format(question=question, markdown=markdown),
                images=[image],
            )
            ans_tokens = sampler.tokenizer.encode(answer)
            out_tokens = sampler.tokenizer.encode(out)
            metrics.append(rouge_l(ans_tokens, out_tokens))
            answers.append(answer)
            output.append(out)
    avg_metrics = {
        "precision": sum([m["precision"] for m in metrics]) / len(testset),
        "recall": sum([m["recall"] for m in metrics]) / len(testset),
        "f1": sum([m["f1"] for m in metrics]) / len(testset),
    }
    details = {
        "answers": answers,
        "output": output,
    }
    return avg_metrics, details


# =============================
# Dataset preparation
# =============================


def create_dataset(max_rows=None):
    path = os.path.expanduser("output/vqa/dataset_1k.parquet")
    screenshot_dir = os.path.dirname(path) + "/screenshots"
    df = pd.read_parquet(path)
    df["image_path"] = df.image_path.apply(
        lambda p: p.replace("/ml-datasets/xbrl_facts/screenshots", screenshot_dir)
    )
    df = df[df.apply(lambda row: os.path.exists(row.image_path), axis=1)]
    if max_rows:
        df = df.iloc[0:max_rows]

    # replace bad evidence with "(none)"
    df["evidence"] = df.apply(
        lambda row: row["evidence"] if row["is_good"] else "(none)", axis=1
    )

    assert df.shape[0] > 0
    return Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=108)


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
    # print(f"==== memory before gradient calc: {get_memory_gb()}")
    (loss, _), grad = grad_fn(model, tokens, images, completion_mask)
    # print(f"==== memory before update: {get_memory_gb()}")
    optimizer.update(model, grad)
    # print(f"==== memory before metric update: {get_memory_gb()}")
    metrics.update(loss=loss)
    return loss


def train(sampler: Sampler, trainset: Dataset, testset: Dataset, ckpt_base_path: str):
    # print(f"==== memory at start: {get_memory_gb()}")
    tokenizer = sampler.tokenizer
    model = sampler.model
    if ckpt_path := lora.latest_checkpoint_path(ckpt_base_path):
        print(f"Loading LoRA checkpoint from {ckpt_path}")
        model = lora.load(model, ckpt_path)
        sampler.model = model

    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=trainable)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    gen_metrics, _ = calc_gen_metrics(sampler, testset)
    print(f"!!! Before training: gen_metrics = {gen_metrics}")
    step = 0
    # print(f"==== memory before train loop: {get_memory_gb()}")
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
            prompts = [
                PROMPT_TEMPLATE.format(question=q, markdown=md)
                for q, md in zip(questions, markdowns)
            ]
            completions = [COMPLETION_TEMPLATE.format(a) for a in answers]
            tokens, completion_mask = encode_batch_for_prompt_completion(
                tokenizer, prompts, completions, pad_to_multiple_of=64
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
                gen_metrics, _ = calc_gen_metrics(sampler, testset)
                print(f"!!! Epoch {epoch}, step {step}: gen_metrics = {gen_metrics}")
                print("Finished training!")
                break
            if step % 100 == 0:
                gen_metrics, _ = calc_gen_metrics(sampler, testset)
                print(f"!!! Epoch {epoch}, step {step}: gen_metrics = {gen_metrics}")
                ckpt_path = os.path.join(
                    f"{ckpt_base_path}/{datetime.now().strftime('%Y-%m_%H-%M-%S')}.ckpt"
                )
                print(f"Saving LoRA checkpoint to {ckpt_path}")
                lora.save(model, ckpt_path)
            # print(f"==== memory at step {step}: {get_memory_gb()}")


# ===========
# Main
# ===========


def get_memory_gb():
    """Get current GPU memory usage in MB"""
    out = {}
    for i, device in enumerate(jax.devices()):
        stats = device.memory_stats()
        out[i] = stats["bytes_in_use"] / (1024**3)
    return out


def show_result(sampler, testset):
    avg_metrics, details = calc_gen_metrics(sampler, testset)
    for question, answer, output in zip(
        testset["question"], details["answers"], details["output"]
    ):
        print(f"{COLORS[0]}Q: {question}\n{ENDC}")
        print(f"{COLORS[1]}A: {answer}\n{ENDC}")
        print(f"{COLORS[2]}O: {output}{ENDC}")
        print("-" * 40)


def main(ckpt_base_path: str = "output/vqa_lora"):
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
    # sampler.model.vision_encoder.rngs = nnx.Rngs(0)  # hack around NNX <> Linen interop

    lora_sharding = NamedSharding(mesh, P())
    lora.apply(sampler.model, rank=64, sharding=lora_sharding, rngs=nnx.Rngs(0))

    train(sampler, trainset, testset, ckpt_base_path)
    lora.merge(sampler.model)
    to_huggingface(sampler.model, "google/gemma-3-4b-it", "output/gemma-3-4b-audit-vqa")

    from vllm import LLM

    llm = LLM(model="output/gemma-3-4b-audit-vqa")
    batch = next(testset.iter(batch_size=1))
    image_path, markdown, question, answer = (
        batch["image_path"][0],
        batch["markdown"][0],
        batch["question"][0],
        batch["evidence"][0],
    )
    image = Image.open(image_path)
    out = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given a page image and its <MARKDOWN>, extract evidence for the <QUESTION>",
                    },
                    {"type": "image_pil", "image_pil": image},
                    {
                        "type": "text",
                        "text": f"<MARKDOWN>{markdown}</MARKDOWN>"
                        + f"<QUESTION>{question}</QUESTION>",
                    },
                ],
            }
        ]
    )
    out[0].outputs[0].text
