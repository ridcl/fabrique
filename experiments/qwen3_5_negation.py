"""Fine-tune Qwen3.5 on a trivial task, then export for serving.

The point is not the task -- it is an end-to-end check that this JAX port is
correct in a way a forward-pass comparison cannot show: if the model *learns*
and the exported weights reproduce the learned behaviour in an independent
runtime (vLLM), then the forward pass, the gradients, the optimiser wiring and
the weight round-trip are all right together.

Task: negate the user's statement.
    "The sky is blue."  ->  "No, the sky isn't blue."

Both sides are generated from templates, so a held-out split of unseen subjects
tells generalisation apart from memorisation.

    python experiments/qwen3_5_negation.py --steps 200 --out /data/qwen3_5-negation

Then serve the exported directory with vLLM and ask it to negate something.
"""

import argparse
import dataclasses
import logging
import os
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from fabrique.models.qwen3_5 import params as params_lib
from fabrique.models.qwen3_5.loading import load_model, resolve_model_dir

# force=True: importing jax/tunix installs a root logging handler, which makes a
# plain basicConfig() a silent no-op and hides every logger.info below.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-0.8B"

SUBJECTS_TRAIN = [
    ("the sky", "blue"),
    ("the grass", "green"),
    ("this room", "warm"),
    ("the coffee", "hot"),
    ("the road", "wet"),
    ("her answer", "correct"),
    ("the door", "open"),
    ("the soup", "salty"),
    ("my laptop", "fast"),
    ("the movie", "boring"),
    ("the bread", "fresh"),
    ("that idea", "sensible"),
    ("the water", "cold"),
    ("his story", "true"),
    ("the paint", "dry"),
    ("the meeting", "useful"),
    ("the cat", "hungry"),
    ("this chair", "comfortable"),
    ("the train", "late"),
    ("the light", "bright"),
    ("the river", "deep"),
    ("the joke", "funny"),
    ("the book", "long"),
    ("the air", "fresh"),
]
SUBJECTS_HELD_OUT = [
    ("the ceiling", "white"),
    ("the engine", "loud"),
    ("this puzzle", "hard"),
    ("the milk", "sour"),
    ("the garden", "tidy"),
    ("his coat", "heavy"),
]


def make_pair(subject: str, adjective: str) -> tuple[str, str]:
    return (
        f"{subject.capitalize()} is {adjective}.",
        f"No, {subject} isn't {adjective}.",
    )


@dataclasses.dataclass
class Example:
    tokens: np.ndarray  # [L]
    completion_mask: np.ndarray  # [L] 1 where loss applies


def build_dataset(tokenizer, subjects, max_len: int) -> list[Example]:
    """Tokenise with the model's own chat template.

    Using the real template matters: vLLM will apply it at serving time, so
    training on a different format would look fine here and fail there.
    """
    examples = []
    for subject, adjective in subjects:
        user, assistant = make_pair(subject, adjective)
        msgs = [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        full = prompt + assistant + "<|im_end|>"
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(f_ids) > max_len:
            raise ValueError(f"example longer than max_len: {len(f_ids)} > {max_len}")
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        tokens = np.full(max_len, pad_id, dtype=np.int32)
        tokens[: len(f_ids)] = f_ids
        mask = np.zeros(max_len, dtype=np.int32)
        mask[len(p_ids) : len(f_ids)] = 1  # loss on the answer only
        examples.append(Example(tokens, mask))
    return examples


def batches(examples, batch_size, rng):
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            chunk = [examples[j] for j in order[i : i + batch_size]]
            yield (
                jnp.asarray(np.stack([c.tokens for c in chunk])),
                jnp.asarray(np.stack([c.completion_mask for c in chunk])),
            )


def loss_fn(model, tokens, completion_mask):
    logits, _ = model(tokens)
    logits = logits[:, :-1].astype(jnp.float32)
    targets = tokens[:, 1:]
    mask = completion_mask[:, 1:].astype(jnp.float32)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.sum(ce * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@nnx.jit
def train_step(model, optimizer, tokens, completion_mask):
    loss, grads = nnx.value_and_grad(loss_fn)(model, tokens, completion_mask)
    optimizer.update(model, grads)
    return loss


def greedy_generate(
    model, tokenizer, prompt: str, max_new: int, buffer_len: int
) -> str:
    """Cache-free greedy decode.

    There is no KV cache in this port yet, so each step re-runs the whole
    prefix.  Writing into a fixed-size buffer keeps the shape constant, so this
    compiles once instead of once per length -- O(L^2) compute but only one
    compile, which is what makes it usable for a smoke test.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    buf = np.full((1, buffer_len), pad_id, dtype=np.int32)
    buf[0, : len(ids)] = ids
    n = len(ids)

    @nnx.jit
    def step(m, toks):
        logits, _ = m(toks)
        return logits

    out = []
    for _ in range(max_new):
        logits = step(model, jnp.asarray(buf))
        nxt = int(jnp.argmax(logits[0, n - 1]))
        if nxt == tokenizer.eos_token_id or n >= buffer_len:
            break
        out.append(nxt)
        buf[0, n] = nxt
        n += 1
    return tokenizer.decode(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="/data/qwen3_5-negation")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--gen-buffer", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mesh", action="store_true", help="shard across devices (see note in main)"
    )
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if not getattr(tokenizer, "chat_template", None):
        jinja = os.path.join(model_dir, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja) as f:
                tokenizer.chat_template = f.read()

    # Single device by default.  0.8B in bf16 plus AdamW state fits in 24 GB,
    # and under the default ("tp", fsdp) embedding sharding jax cannot resolve
    # an output sharding for the embedding gather ("use .at[...].get(
    # out_sharding=)"), so a mesh needs that addressed first.
    mesh = None
    if args.mesh:
        mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
    config, model = load_model(model_dir, dtype=jnp.bfloat16, mesh=mesh)
    logger.info("loaded %s (%d layers)", args.model, config.num_layers)

    train = build_dataset(tokenizer, SUBJECTS_TRAIN, args.max_len)
    logger.info("train %d examples, held-out %d", len(train), len(SUBJECTS_HELD_OUT))

    optimizer = nnx.Optimizer(
        model, optax.adamw(args.lr, b1=0.9, b2=0.95, weight_decay=0.0), wrt=nnx.Param
    )

    def show_samples(tag):
        for subject, adjective in SUBJECTS_HELD_OUT[:3]:
            user, want = make_pair(subject, adjective)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            got = greedy_generate(
                model, tokenizer, prompt, max_new=24, buffer_len=args.gen_buffer
            )
            logger.info("  [%s] %-28s -> %-42r (want %r)", tag, user, got, want)

    logger.info("--- before training (held-out prompts) ---")
    show_samples("base")

    rng = random.Random(args.seed)
    stream = batches(train, args.batch_size, rng)
    t0 = time.perf_counter()
    for step_i in range(1, args.steps + 1):
        tokens, mask = next(stream)
        loss = float(train_step(model, optimizer, tokens, mask))
        if step_i == 1 or step_i % 10 == 0:
            logger.info(
                "step %4d/%d  loss %.4f  (%.1fs)",
                step_i,
                args.steps,
                loss,
                time.perf_counter() - t0,
            )
        if args.eval_every and step_i % args.eval_every == 0:
            show_samples(f"step{step_i}")

    logger.info("--- after training (held-out prompts) ---")
    show_samples("tuned")

    params_lib.save_model_as_safetensors(model, config, model_dir, args.out)
    logger.info("exported to %s", args.out)
    logger.info("serve with:  vllm serve %s", args.out)


if __name__ == "__main__":
    main()
