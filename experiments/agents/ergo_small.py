"""Rejection-sampling fine-tuning (RFT) of Qwen3.5-0.8B on agent trajectories.

Trajectories come from a Qwen3.5-4B agent, each with a judge verdict and
per-step credits.  We keep the "good" / "great" ones, and train with LoRA on the
agent's own actions only -- its thoughts and tool calls; system, user and tool
messages are context, not targets.  Data-parallel over all visible GPUs.

    python experiments/agents/ergo_small.py --steps 500 --out /data/ergo-small

Device-dependent settings live in PROFILES below and are picked from the GPU
automatically (``--profile`` to force one, ``--max-len`` to override length).
"""

import os

# Pallas/Triton kernels need some GPU memory *outside* XLA's preallocated pool
# when they launch; with the devcontainer's 0.95 they fail with "Failed to
# launch CUDA kernel ... CUDA_ERROR_OUT_OF_MEMORY".  0.90 is measured to work.
# Must run before JAX initialises a backend.
if float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")) > 0.90:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse  # noqa: E402
import dataclasses
import json
import logging
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from fabrique import attention as attention_lib
from fabrique.models.qwen3_5 import model as model_lib
from fabrique.models.qwen3_5 import params as params_lib
from fabrique.models.qwen3_5.loading import (
    load_model,
    load_tokenizer,
    resolve_model_dir,
)

# force=True: importing jax installs a root logging handler, which makes a plain
# basicConfig() a silent no-op and hides every logger.info below.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATA = "/data/simple-100-trajectories-2026-19-11_01-10.json"
KEEP_VERDICTS = ("good", "great")


# ===========================================================================
# Device profiles
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class DeviceProfile:
    max_len: int  # trajectories are truncated here (tokens)
    gdn_chunk_size: int  # DeltaNet chunking; a pure speed knob
    gdn_scan_unroll: int  # >1 trades compile time for (little) speed
    # What to recompute in the backward; see qwen3_5.model.RematConfig.  "auto"
    # recomputes only the DeltaNet core below remat_block_len tokens (~22%
    # faster) and whole layers from there on (much less memory).
    remat: str
    remat_block_len: int
    note: str


# Trajectory lengths (551 good/great): p50 10.9k, p90 17.0k, max 31.2k tokens;
# 16k fits 88% whole, 32k fits all.
PROFILES = {
    # 2x RTX 3090, LoRA, per GPU: 8k 1.25 s (core-only remat, 20.8 GiB -- too
    # close to the pool to use at 8k itself) / 1.6 s (full remat, 6.1 GiB);
    # 16k 4.1 s, 8.2 GiB; 24k 8.8 s, 10.1 GiB; 32k 17.0 s, 12.0 GiB.
    "rtx3090": DeviceProfile(
        max_len=16384,
        gdn_chunk_size=64,
        gdn_scan_unroll=1,
        remat="auto",
        remat_block_len=8192,
        note="24 GiB; measured",
    ),
    # Extrapolated, not measured: core-only remat at ~1.1 GiB per 1k tokens
    # would be ~48 GiB at 32k, so it is used at every length.
    "h100": DeviceProfile(
        max_len=32768,
        gdn_chunk_size=64,
        gdn_scan_unroll=1,
        remat="auto",
        remat_block_len=65536,
        note="80 GiB; NOT yet measured -- extrapolated from the 3090",
    ),
    "cpu": DeviceProfile(
        max_len=1024,
        gdn_chunk_size=64,
        gdn_scan_unroll=1,
        remat="block",
        remat_block_len=0,
        note="smoke tests only",
    ),
}


def detect_profile() -> str:
    d = jax.devices()[0]
    if d.platform != "gpu":
        return "cpu"
    kind = d.device_kind.lower()
    if any(x in kind for x in ("h100", "h200", "h800", "gh200")):
        return "h100"
    if any(x in kind for x in ("3090", "4090", "a5000", "l4")):
        return "rtx3090"
    gib = d.memory_stats().get("bytes_limit", 0) / 2**30
    return "h100" if gib >= 70 else "rtx3090"


# ===========================================================================
# Data preparation
# ===========================================================================

# Every assistant turn renders as `<|im_start|>assistant\n<think>...</think>...`.
# Targets start right after this header, so the think block is learned too.
# (The generation prompt's *empty* think block is not a prefix of turns whose
# content carries real reasoning before a `</think>`, so it can't be the split.)
ASSISTANT_HEADER = "<|im_start|>assistant\n"


@dataclasses.dataclass
class Example:
    tokens: np.ndarray  # [L] int32
    loss_mask: np.ndarray  # [L] 1 on agent action tokens


def normalize_message(m: dict) -> dict:
    """OpenAI-style message -> what the Qwen3.5 chat template expects."""
    m = {k: v for k, v in m.items() if v is not None}
    if m.get("tool_calls"):
        # Stored as JSON strings; the template iterates them as a mapping.
        m["tool_calls"] = [
            {
                **tc,
                "function": {
                    **tc["function"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                },
            }
            for tc in m["tool_calls"]
        ]
    return m


def encode_trajectory(
    tokenizer, traj: dict, max_len: int, min_credit: float | None
) -> Example:
    """Render with the model's own chat template; mask in assistant turns only.

    Spans are found at the character level by rendering message prefixes, then
    mapped to tokens via offsets -- so the full conversation is tokenised once,
    exactly as it will be at serving time.
    """
    msgs = [normalize_message(m) for m in traj["messages"]]
    credit = {c["index"]: c["value"] for c in traj["assignment"]["credits"]}
    render = lambda ms: tokenizer.apply_chat_template(ms, tokenize=False)  # noqa: E731

    full = render(msgs)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if min_credit is not None and credit.get(i, 1.0) < min_credit:
            continue
        start = len(render(msgs[:i])) + len(ASSISTANT_HEADER)
        spans.append((start, len(render(msgs[: i + 1]))))

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    tokens = np.asarray(enc["input_ids"], dtype=np.int32)[:max_len]
    starts = np.asarray([s for s, _ in enc["offset_mapping"]])[:max_len]
    mask = np.zeros(len(tokens), dtype=np.int32)
    for s, e in spans:
        mask[(starts >= s) & (starts < e)] = 1
    return Example(tokens, mask)


def load_examples(
    tokenizer, path: str, max_len: int, min_credit: float | None
) -> list[Example]:
    with open(path) as f:
        trajs = json.load(f)
    kept = [t for t in trajs if t["assignment"]["verdict"] in KEEP_VERDICTS]
    examples = [encode_trajectory(tokenizer, t, max_len, min_credit) for t in kept]
    n_trunc = sum(len(e.tokens) == max_len for e in examples)
    logger.info(
        "kept %d/%d trajectories (%s); %d truncated to %d tokens; %d target tokens",
        len(kept), len(trajs), "/".join(KEEP_VERDICTS), n_trunc, max_len,
        sum(int(e.loss_mask.sum()) for e in examples),
    )  # fmt: skip
    return examples


def bucket_lengths(max_len: int, base: int, ratio: float) -> list[int]:
    """Padded lengths: multiples of `base`, growing geometrically up to max_len.

    Every distinct length is a separate ~40 s compile.  Linear 1k buckets meant
    up to 32 compiles per run at 32k; geometric buckets bound that at ~10 with
    ratio 1.25, for at most 25% padding (about 11% on average).
    """
    sizes, n = [], base
    while n < max_len:
        sizes.append(n)
        n = max(n + base, -(-int(n * ratio) // base) * base)
    return sizes + [max_len]


def batches(examples, batch_size: int, buckets: list[int], pad_id: int, rng):
    """Shuffled batches, right-padded to the smallest bucket that fits.

    Right padding is harmless here: attention and the DeltaNet recurrence are
    both causal, so real tokens never see it.
    """
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            chunk = [examples[j] for j in order[i : i + batch_size]]
            n = max(len(c.tokens) for c in chunk)
            L = next(b for b in buckets if b >= n)
            tokens = np.full((batch_size, L), pad_id, dtype=np.int32)
            mask = np.zeros((batch_size, L), dtype=np.int32)
            for b, c in enumerate(chunk):
                tokens[b, : len(c.tokens)] = c.tokens
                mask[b, : len(c.tokens)] = c.loss_mask
            yield tokens, mask


# ===========================================================================
# Training
# ===========================================================================

LORA_TARGETS = (
    r".*(q_proj|k_proj|v_proj|o_proj"  # full attention
    r"|in_proj_qkv|in_proj_z|out_proj"  # gated DeltaNet (not the conv, not a/b)
    r"|gate_proj|up_proj|down_proj)"  # MLP
)


def apply_lora(model, rank: int, alpha: float):
    provider = qwix.LoraProvider(module_path=LORA_TARGETS, rank=rank, alpha=alpha)
    return qwix.apply_lora_to_model(
        model, provider, **model.get_model_input(), rngs=nnx.Rngs(0)
    )


def merge_lora(model, rank: int, alpha: float) -> None:
    """Fold LoRA deltas into the base weights, in place, for export.

    qwix stores `{w}_lora_a` / `{w}_lora_b` next to each adapted weight; the
    delta is a @ b over the rank axis, scaled by alpha/rank.  Verified to match
    the adapted forward pass, including the 3-D attention Einsum weights.  The
    model must not be run afterwards -- the adapters are still attached.
    """
    targets = [
        (mod, name[: -len("_lora_a")])
        for _, mod in nnx.iter_graph(model)
        if isinstance(mod, nnx.Module)
        for name in list(vars(mod))
        if name.endswith("_lora_a")
    ]
    for mod, w in targets:
        a, b, base = (
            getattr(mod, w + "_lora_a"),
            getattr(mod, w + "_lora_b"),
            getattr(mod, w),
        )
        delta = jnp.tensordot(a[...], b[...], axes=1).reshape(base.shape) * (
            alpha / rank
        )
        base[...] = (base[...] + delta).astype(base.dtype)
    logger.info("merged %d LoRA adapters", len(targets))


def project(model, x):
    """Final hidden states -> logits, as the model's own head does."""
    return model.embedder.decode(x) if model.lm_head is None else model.lm_head(x)


def loss_fn(model, tokens, loss_mask, chunk: int):
    """Next-token CE on action tokens, without materialising full logits.

    At ~16k tokens x 248k vocab, fp32 logits alone are ~16 GB.  So we take the
    final hidden states (the model's own logits are unused, and XLA drops them)
    and project + score `chunk` positions at a time under remat.
    """
    B, L = tokens.shape
    _, hidden = model(tokens, output_hidden_states=True)  # right padding: see batches()
    x = model.final_norm(hidden[-1])  # [B, L, D]
    # Shift left: position t predicts token t+1; the last position predicts nothing.
    targets = jnp.concatenate([tokens[:, 1:], jnp.zeros((B, 1), tokens.dtype)], axis=1)
    mask = jnp.concatenate(
        [loss_mask[:, 1:], jnp.zeros((B, 1), loss_mask.dtype)], axis=1
    )

    n = L // chunk
    split = lambda a: a.reshape(B, n, chunk, *a.shape[2:]).swapaxes(0, 1)  # noqa: E731

    @jax.checkpoint
    def body(total, xs):
        xc, yc, mc = xs
        logits = project(model, xc).astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, yc)
        return total + jnp.sum(ce * mc), None

    total, _ = jax.lax.scan(body, 0.0, (split(x), split(targets), split(mask)))
    return total / jnp.maximum(jnp.sum(mask), 1)


@nnx.jit(static_argnames="chunk")
def train_step(model, optimizer, tokens, loss_mask, chunk):
    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam))
    loss, grads = grad_fn(model, tokens, loss_mask, chunk)
    optimizer.update(model, grads)
    return loss


@nnx.jit(static_argnames="chunk")
def eval_step(model, tokens, loss_mask, chunk):
    return loss_fn(model, tokens, loss_mask, chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="/data/ergo-small")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--profile", default=None, choices=sorted(PROFILES))
    ap.add_argument("--max-len", type=int, default=None, help="override the profile")
    ap.add_argument(
        "--bucket", type=int, default=1024, help="padded lengths are multiples of this"
    )
    ap.add_argument(
        "--bucket-ratio", type=float, default=1.25, help="geometric bucket growth"
    )
    ap.add_argument("--loss-chunk", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument(
        "--min-credit", type=float, default=None, help="drop steps credited below this"
    )
    ap.add_argument("--n-eval", type=int, default=16, help="held-out trajectories")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    profile_name = args.profile or detect_profile()
    profile = PROFILES[profile_name]
    args.max_len = args.max_len or profile.max_len
    assert args.max_len % args.bucket == 0 and args.bucket % args.loss_chunk == 0
    buckets = bucket_lengths(args.max_len, args.bucket, args.bucket_ratio)
    logger.info(
        "device %s -> profile %r (%s); max_len %d",
        jax.devices()[0].device_kind, profile_name, profile.note, args.max_len,
    )  # fmt: skip

    # --- data -------------------------------------------------------------
    model_dir = resolve_model_dir(args.model)
    tokenizer = load_tokenizer(model_dir)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    examples = load_examples(tokenizer, args.data, args.max_len, args.min_credit)
    random.Random(args.seed).shuffle(examples)
    eval_set, train_set = examples[: args.n_eval], examples[args.n_eval :]

    # --- model ------------------------------------------------------------
    config = model_lib.ModelConfig.from_hf_config(model_dir)
    config.remat_config = model_lib.RematConfig(profile.remat)
    config.remat_auto_block_len = profile.remat_block_len
    config.gdn_chunk_size = profile.gdn_chunk_size
    config.gdn_scan_unroll = profile.gdn_scan_unroll
    config, model = load_model(model_dir, dtype=jnp.bfloat16, config=config)
    logger.info(
        "attention backend: %s",
        attention_lib.select_backend(jnp.bfloat16, config.head_dim, causal=True),
    )
    model = apply_lora(model, args.lora_rank, args.lora_alpha)

    # Data parallel: replicate the model, split the batch.  Gradients of the
    # replicated LoRA params are all-reduced by XLA inside the jitted step.
    n_dev = jax.device_count()
    # Auto axes: with the default Explicit ones, the embedding gather needs an
    # explicit out_sharding ("use .at[...].get(out_sharding=)").
    mesh = jax.make_mesh((n_dev,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    replicate = lambda tree: jax.device_put(tree, NamedSharding(mesh, P()))  # noqa: E731
    nnx.update(model, replicate(nnx.state(model)))
    shard = lambda a: jax.device_put(a, NamedSharding(mesh, P("data")))  # noqa: E731
    # The mesh context lets kernels XLA cannot partition (Pallas attention) run
    # per batch shard via shard_map instead of being all-gathered onto every GPU.
    jax.sharding.set_mesh(mesh)

    tx = optax.MultiSteps(
        optax.adamw(args.lr, weight_decay=0.0), every_k_schedule=args.accum
    ).gradient_transformation()
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.LoRAParam)
    # Replicate the optimizer state too: created on device 0, it would come back
    # replicated after step 1 and change the jit signature -- a full recompile
    # at step 2.
    nnx.update(optimizer, replicate(nnx.state(optimizer)))
    logger.info(
        "training on %d device(s), global batch %d x accum %d", n_dev, n_dev, args.accum
    )

    def evaluate():
        stream = batches(eval_set, n_dev, buckets, pad_id, random.Random(0))
        losses = [
            float(eval_step(model, shard(t), shard(m), args.loss_chunk))
            for t, m in (next(stream) for _ in range(len(eval_set) // n_dev))
        ]
        return float(np.mean(losses))

    logger.info("eval loss before training: %.4f", evaluate())
    stream = batches(train_set, n_dev, buckets, pad_id, random.Random(args.seed))
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        tokens, mask = next(stream)
        loss = float(
            train_step(model, optimizer, shard(tokens), shard(mask), args.loss_chunk)
        )
        if step == 1 or step % 10 == 0:
            logger.info("step %4d/%d  len %5d  loss %.4f  (%.0fs)",
                        step, args.steps, tokens.shape[1], loss, time.perf_counter() - t0)  # fmt: skip
        if args.eval_every and step % args.eval_every == 0:
            logger.info("step %4d  eval loss %.4f", step, evaluate())
    logger.info("eval loss after training: %.4f", evaluate())

    # --- export -------------------------------------------------------------
    merge_lora(model, args.lora_rank, args.lora_alpha)
    params_lib.save_model_as_safetensors(model, config, model_dir, args.out)
    logger.info(
        "exported merged model to %s  (serve: vllm serve %s)", args.out, args.out
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
