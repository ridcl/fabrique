"""Gemma 4 sampler with optional vision support.

Supports greedy and top-p sampling for both text-only and vision-language
inputs using the Gemma 4 JAX model.

Usage::

    # Text-only (greedy):
    python -m fabrique.models.gemma4.sampler \\
        --model_id_or_dir google/gemma-4-e4b-it \\
        --prompt "<start_of_turn>user\\nWhat is the capital of France?<end_of_turn>\\n<start_of_turn>model\\n"

    # With an image (top-p):
    python -m fabrique.models.gemma4.sampler \\
        --model_id_or_dir google/gemma-4-e4b-it \\
        --image_path /path/to/image.jpg \\
        --top_p 0.9 --temperature 0.7
"""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Sequence
from typing import Optional

import flax
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx import graph, statelib
from transformers import AutoProcessor

from fabrique.models.gemma4 import model as model_lib
from fabrique.models.gemma4 import params as params_lib
from fabrique.models.gemma4.utils import encode_batch, load_processor


# ---------------------------------------------------------------------------
# Model ID / directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(model_id_or_dir: str) -> str:
    """Return a local directory path for the given model ID or local path."""
    if os.path.isdir(model_id_or_dir):
        return model_id_or_dir
    print(f'Downloading snapshot for "{model_id_or_dir}" from HuggingFace Hub…')
    return huggingface_hub.snapshot_download(model_id_or_dir)


# ---------------------------------------------------------------------------
# Sampling state
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class _SamplingState:
    """Internal state carried through the decode loop."""

    # Current decode step (counts up from 0 after prefill, pointing to the last
    # written position in token_buffer).
    decoding_step: jnp.int32

    # Token buffer: filled left-to-right, prompt first then generated tokens.
    token_buffer: jnp.ndarray  # [B, total_steps]

    # Position of the *next* token to generate (one per batch item).
    next_position: jnp.ndarray  # [B]

    # Per-layer KV cache.
    cache: dict

    # Whether each sequence has finished (hit EOS or step limit).
    done: jnp.ndarray  # [B]

    # Accumulated logits (None when not requested).
    logits_buffer: jnp.ndarray | None  # [B, total_steps, V]

    # Random key threaded through top-p sampling.
    seed: jax.Array

    # --- static (not traced by JAX) ---
    total_sampling_steps: int = flax.struct.field(pytree_node=False)
    num_input_tokens: int = flax.struct.field(pytree_node=False)
    sampling_mode: str = flax.struct.field(pytree_node=False)
    temperature: float = flax.struct.field(pytree_node=False)
    top_p: float = flax.struct.field(pytree_node=False)
    top_k: int | None = flax.struct.field(pytree_node=False)
    forbidden_token_ids: tuple[int, ...] | None = flax.struct.field(
        pytree_node=False
    )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class Gemma4Sampler:
    """Autoregressive sampler for Gemma 4 with optional image support.

    Differences from the generic Tunix ``Sampler``:

    * Model call takes ``(tokens, positions, pixel_values, pixel_position_ids,
      cache, attention_mask)`` — positions are 1-D cumsum-based (no M-RoPE).
    * Vision tokens are injected during prefill; decode steps are text-only.
    * Attention mask during prefill is ``[B, L, L]`` causal & padding; decode
      steps pass ``None`` (single real token, no masking needed).
    """

    def __init__(
        self,
        model: model_lib.Gemma4,
        processor: AutoProcessor,
        cache_size: int,
    ):
        self._model_graphdef: graph.NodeDef = nnx.graphdef(model)
        self._model_state: statelib.State = nnx.variables(model)
        self._flattened_model_state = jax.tree.leaves(
            self._model_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        self._cache_size = cache_size
        self._config: model_lib.ModelConfig = model.config
        self._processor = processor
        self._tokenizer = processor.tokenizer

        # Gemma4.__call__ is decorated with @nnx.jit, which rejects mixed-device
        # parameters (embedding on CPU, rest on GPU when cpu_embed=True).  Bypass
        # the nnx.jit wrapper so that:
        #   • _prefill runs eagerly — JAX handles cross-device ops without issue.
        #   • _decode_step runs inside the outer jax.jit(self._decode_fn), which
        #     traces through the raw function and compiles the whole decode loop.
        _call = type(model).__call__
        self._forward = getattr(_call, "__wrapped__", _call)

        self._compiled_decode_fn = jax.jit(self._decode_fn)

    @property
    def _model(self) -> model_lib.Gemma4:
        return nnx.merge(self._model_graphdef, self._flattened_model_state)

    @property
    def _dtype(self) -> jnp.dtype:
        return self._flattened_model_state[0].dtype

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill(
        self,
        input_ids: np.ndarray,  # [B, L]
        attention_mask: np.ndarray,  # [B, L]
        pixel_values: np.ndarray | None,  # [B, N, C]
        pixel_position_ids: np.ndarray | None,  # [B, N, 2]
        total_sampling_steps: int,
        sampling_mode: str,
        temperature: float,
        top_p: float,
        top_k: int | None,
        forbidden_token_ids: tuple[int, ...] | None,
        seed: jax.Array,
        include_logits: bool,
    ) -> _SamplingState:
        """Run the prefill pass and initialise the sampling state."""
        batch_size, seq_len = input_ids.shape

        input_ids_jax = jnp.array(input_ids)
        attn_mask_jax = jnp.array(attention_mask, dtype=jnp.bool_)  # [B, L]

        # Build 2-D causal + padding attention mask [B, L, L].
        pos = jnp.arange(seq_len)[None, :]  # [1, L]
        causal = pos[:, :, None] >= pos[:, None, :]  # [1, L, L]
        padding = attn_mask_jax[:, None, :]  # [B, 1, L]
        attn_mask_2d = causal & padding  # [B, L, L]

        # 1-D positions: cumulative count of valid tokens seen so far.
        cum = jnp.cumsum(attn_mask_jax, axis=1) - 1  # 0-indexed
        positions_jax = jnp.where(attn_mask_jax, cum, 0).astype(jnp.int32)

        pixel_values_jax = (
            jnp.array(pixel_values, dtype=jnp.float32)
            if pixel_values is not None
            else None
        )
        pixel_position_ids_jax = (
            jnp.array(pixel_position_ids, dtype=jnp.int32)
            if pixel_position_ids is not None
            else None
        )

        # Initialise the token buffer (prompt + space for generated tokens).
        pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        )
        token_buffer = np.full(
            (batch_size, total_sampling_steps), pad_id, dtype=np.int32
        )
        token_buffer[:, :seq_len] = input_ids
        token_buffer_jax = jnp.array(token_buffer)

        # Initialise KV cache.
        cache = self._model.init_cache(batch_size, self._cache_size, self._dtype)

        # Full-sequence prefill forward pass.  Use the unwrapped (non-nnx.jit)
        # forward function so that mixed-device parameters (cpu_embed=True) are
        # handled by eager JAX rather than rejected by nnx.jit's device check.
        model = nnx.merge(self._model_graphdef, self._flattened_model_state)
        logits, cache = self._forward(
            model,
            input_ids_jax,
            positions_jax,
            pixel_values_jax,
            pixel_position_ids_jax,
            cache,
            attn_mask_2d,
        )

        # Sample the first generated token from the last-position logits.
        first_token, seed = self._sample_token(
            logits[:, -1:, :],
            sampling_mode,
            temperature,
            top_p,
            top_k,
            forbidden_token_ids,
            seed,
            step=0,
        )
        token_buffer_jax = token_buffer_jax.at[:, seq_len].set(first_token)

        # next_position = number of valid input tokens (= position of the next
        # token to generate, 0-indexed).
        next_position = jnp.sum(attn_mask_jax, axis=-1).astype(jnp.int32)  # [B]

        if include_logits:
            logits_buffer = jnp.zeros(
                (batch_size, total_sampling_steps, self._config.num_embed),
                dtype=jnp.float32,
            )
            logits_buffer = logits_buffer.at[:, seq_len - 1].set(
                logits[:, -1, :].astype(jnp.float32)
            )
        else:
            logits_buffer = None

        eos_id = self._tokenizer.eos_token_id
        done = jnp.isin(first_token, jnp.array([eos_id]))

        return _SamplingState(
            decoding_step=jnp.int32(seq_len),
            token_buffer=token_buffer_jax,
            next_position=next_position,
            cache=cache,
            done=done,
            logits_buffer=logits_buffer,
            seed=seed,
            total_sampling_steps=total_sampling_steps,
            num_input_tokens=seq_len,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            forbidden_token_ids=forbidden_token_ids,
        )

    # ------------------------------------------------------------------
    # Decode loop
    # ------------------------------------------------------------------

    def _decode_fn(
        self,
        params: statelib.State,
        state: _SamplingState,
        eos_ids: jax.Array,
    ) -> _SamplingState:
        """JIT-compiled decode loop (``jax.lax.while_loop``)."""

        def cond(s: _SamplingState) -> jnp.ndarray:
            return (s.decoding_step < s.total_sampling_steps - 1) & jnp.any(
                ~s.done
            )

        def step(s: _SamplingState) -> _SamplingState:
            return self._decode_step(params, s, eos_ids)

        return jax.lax.while_loop(cond, step, state)

    def _decode_step(
        self,
        params: statelib.State,
        state: _SamplingState,
        eos_ids: jax.Array,
    ) -> _SamplingState:
        """Single autoregressive decode step."""
        batch_size = state.token_buffer.shape[0]
        step = state.decoding_step

        last_token = state.token_buffer[:, step].reshape(batch_size, 1)  # [B, 1]
        positions = state.next_position[:, None]  # [B, 1]

        model = nnx.merge(self._model_graphdef, params)
        logits, new_cache = self._forward(
            model,
            last_token,
            positions,
            None,  # pixel_values — vision tokens already injected during prefill
            None,  # pixel_position_ids
            state.cache,
            None,  # attention_mask — model builds decode mask from cache end_index
        )

        next_token, new_seed = self._sample_token(
            logits,
            state.sampling_mode,
            state.temperature,
            state.top_p,
            state.top_k,
            state.forbidden_token_ids,
            state.seed,
            step=step,
        )

        new_token_buffer = state.token_buffer.at[:, step + 1].set(next_token)
        new_done = state.done | jnp.isin(next_token, eos_ids)

        new_logits_buffer = state.logits_buffer
        if state.logits_buffer is not None:
            new_logits_buffer = state.logits_buffer.at[:, step].set(
                logits[:, 0, :].astype(jnp.float32)
            )

        return _SamplingState(
            decoding_step=step + 1,
            token_buffer=new_token_buffer,
            next_position=state.next_position + 1,
            cache=new_cache,
            done=new_done,
            logits_buffer=new_logits_buffer,
            seed=new_seed,
            total_sampling_steps=state.total_sampling_steps,
            num_input_tokens=state.num_input_tokens,
            sampling_mode=state.sampling_mode,
            temperature=state.temperature,
            top_p=state.top_p,
            top_k=state.top_k,
            forbidden_token_ids=state.forbidden_token_ids,
        )

    # ------------------------------------------------------------------
    # Token sampling helpers
    # ------------------------------------------------------------------

    def _sample_token(
        self,
        logits: jnp.ndarray,  # [B, 1, V]
        mode: str,
        temperature: float,
        top_p: float,
        top_k: int | None,
        forbidden_token_ids: tuple[int, ...] | None,
        seed: jax.Array,
        step: int | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """Returns ``(next_token [B], new_seed)``."""
        logits_1d = logits[:, -1, :]  # [B, V]
        if forbidden_token_ids:
            logits_1d = logits_1d.at[:, list(forbidden_token_ids)].set(-jnp.inf)

        if mode == "greedy":
            return jnp.argmax(logits_1d, axis=-1), seed

        key = jax.random.fold_in(seed, step)
        new_seed = jax.random.fold_in(seed, step + 1)
        probs = jax.nn.softmax(
            logits_1d.astype(jnp.float32) / temperature, axis=-1
        )
        k = probs.shape[-1] if top_k is None else top_k
        probs_sorted, indices = jax.lax.top_k(probs, k=k)
        cumsum = jnp.cumsum(probs_sorted, axis=-1)
        mask = cumsum - probs_sorted > top_p
        probs_sorted = jnp.where(mask, 0.0, probs_sorted)
        probs_sorted = probs_sorted / jnp.sum(
            probs_sorted, axis=-1, keepdims=True
        )
        sampled = jax.random.categorical(key, jnp.log(probs_sorted + 1e-10))
        next_token = jnp.take_along_axis(indices, sampled[:, None], axis=-1)[:, 0]
        return next_token, new_seed

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompts: str | Sequence[str],
        max_new_tokens: int = 100,
        images=None,  # PIL Image, list of PIL Images (one per prompt), or None
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_tokens: Sequence[int] | None = None,
        forbidden_tokens: Sequence[int] | None = None,
        seed: int = 0,
        echo: bool = False,
    ) -> list[str]:
        """Generate completions.

        Args:
          prompts: A single prompt string or a list of prompt strings.  Each
            string should already be formatted (e.g. via
            ``processor.apply_chat_template``).  If a prompt contains an image
            placeholder, pass the corresponding PIL image via ``images``.
          max_new_tokens: Maximum number of tokens to generate per prompt.
          images: Optional PIL Image(s).  Pass a list of one image per prompt,
            or a single image that is broadcast to all prompts.  ``None`` for
            text-only generation.
          temperature: Sampling temperature (used when ``top_p`` is set).
          top_p: Nucleus-sampling threshold.  If ``None``, greedy decoding.
          top_k: Limits the top-k candidates for top-p sampling.
          eos_tokens: Token IDs that signal end-of-sequence.  Defaults to the
            tokenizer's ``eos_token_id``.
          forbidden_tokens: Token IDs that may never be sampled.
          seed: Integer seed for reproducible top-p sampling.
          echo: If ``True``, include the prompt tokens in the returned strings.

        Returns:
          A list of generated strings, one per prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        sampling_mode = "top_p" if top_p is not None else "greedy"
        forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None
        rng = jax.random.PRNGKey(seed)

        eos_id = self._tokenizer.eos_token_id
        eos_ids = jnp.array(eos_tokens if eos_tokens else [eos_id])

        # Normalise image argument to a list-of-lists matching prompts.
        if images is None:
            image_lists = [[] for _ in prompts]
        elif not isinstance(images, (list, tuple)):
            # Single image broadcast to all prompts.
            image_lists = [[images]] * batch_size
        else:
            image_lists = [[img] if img is not None else [] for img in images]

        batch = encode_batch(
            self._processor,
            list(prompts),
            image_lists,
            vcfg=self._config.vision_config,
            max_length=self._cache_size,
            truncation=False,
            pad_to_multiple_of=128,
            padding_side="left",
        )

        seq_len = batch.input_tokens.shape[1]
        total_sampling_steps = seq_len + max_new_tokens
        if total_sampling_steps > self._cache_size:
            raise ValueError(
                f"seq_len ({seq_len}) + max_new_tokens ({max_new_tokens}) = "
                f"{total_sampling_steps} exceeds cache_size {self._cache_size}."
            )

        # --- Prefill ---
        state = self._prefill(
            input_ids=batch.input_tokens,
            attention_mask=batch.input_mask,
            pixel_values=batch.pixel_values,
            pixel_position_ids=batch.pixel_position_ids,
            total_sampling_steps=total_sampling_steps,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k,
            forbidden_token_ids=forbidden_token_ids,
            seed=rng,
            include_logits=False,
        )

        # --- Decode ---
        state = self._compiled_decode_fn(
            self._flattened_model_state, state, eos_ids
        )

        # --- Decode tokens to strings ---
        pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else eos_id
        )
        eos_set = set(np.array(eos_ids).tolist())
        outputs = []
        for i, token_buffer in enumerate(np.array(state.token_buffer)):
            start = 0 if echo else seq_len
            gen_tokens = token_buffer[seq_len:]
            end = seq_len + len(gen_tokens)  # default: no EOS found
            for j, tok in enumerate(gen_tokens):
                if int(tok) in eos_set:
                    end = seq_len + j
                    break

            out_tokens = token_buffer[start:end]
            out_tokens = out_tokens[out_tokens != pad_id]
            outputs.append(
                self._tokenizer.decode(
                    out_tokens.tolist(), skip_special_tokens=True
                )
            )

        return outputs


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_sampler(
    model_id_or_dir: str,
    cache_size: int = 512,
    dtype: str = "bfloat16",
    config: model_lib.ModelConfig | None = None,
) -> Gemma4Sampler:
    """Load a Gemma 4 model and return a ready-to-use ``Gemma4Sampler``.

    Args:
      model_id_or_dir: HuggingFace repo ID or local checkpoint directory.
      cache_size: KV-cache capacity in tokens.
      dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
      config: Override the model configuration.  Defaults to
        ``ModelConfig.gemma4_e4b_it()``.

    Returns:
      A ``Gemma4Sampler`` instance.
    """
    import dataclasses as _dc

    jax_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    model_dir = resolve_model_dir(model_id_or_dir)

    if config is None:
        config = model_lib.ModelConfig.gemma4_e4b_it()

    cfg = _dc.replace(config, dtype=jax_dtype, param_dtype=jax_dtype)

    # Offload the large embedding tables to CPU RAM when running on a GPU.
    # jax.pure_callback in the Embedder methods provides JIT-safe access.
    _on_gpu = jax.devices()[0].platform != "cpu"
    with jax.default_device(jax.devices()[0]):
        model = params_lib.create_model_from_safe_tensors(
            model_dir, cfg, mesh=None, dtype=jax_dtype, cpu_embed=_on_gpu
        )

    processor = load_processor(model_dir)
    return Gemma4Sampler(model, processor, cache_size=cache_size)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    model_id_or_dir: str = "google/gemma-4-e4b-it",
    prompt: str | None = None,
    image_path: str | None = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float | None = None,
    cache_size: int = 512,
    dtype: str = "bfloat16",
) -> str:
    """Run the sampler and return the generated text.

    Can be called directly from a Python console::

        from fabrique.models.gemma4.sampler import main, load_sampler

        # One-shot (reloads model each call):
        print(main('google/gemma-4-e4b-it', 'What is the capital of France?'))

        # Efficient (reuse loaded sampler):
        sampler = load_sampler('google/gemma-4-e4b-it')
        print(sampler(['Hello!', 'Tell me a joke.'], max_new_tokens=80))
    """
    model_dir = resolve_model_dir(model_id_or_dir)

    image = None
    if image_path is not None:
        from PIL import Image as PILImage

        if os.path.isfile(image_path):
            image = PILImage.open(image_path).convert("RGB")
        else:
            import requests

            image = PILImage.open(
                io.BytesIO(requests.get(image_path, timeout=30).content)
            ).convert("RGB")

    # Build a default prompt via apply_chat_template if not supplied.
    processor = load_processor(model_dir)
    if prompt is None:
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
        else:
            messages = [
                {"role": "user", "content": "What is the capital of France?"}
            ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif image is not None:
        # If prompt is given with an image, build it with the chat template so
        # the image placeholder is inserted correctly.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    sampler = load_sampler(
        model_dir, cache_size=cache_size, dtype=dtype
    )
    results = sampler(
        [prompt],
        max_new_tokens=max_new_tokens,
        images=[image] if image is not None else None,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"Prompt : {prompt!r}")
    print(f"Output : {results[0]}")
    return results[0]


if __name__ == "__main__" and "__file__" in globals():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_id_or_dir",
        default="google/gemma-4-e4b-it",
        help="HuggingFace repo ID or local checkpoint directory.",
    )
    parser.add_argument("--prompt", default=None, help="Text prompt.")
    parser.add_argument(
        "--image_path",
        default=None,
        help="Path or URL to an image (enables vision-language mode).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --top_p).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p threshold.  Omit for greedy decoding.",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=512,
        help="KV-cache size in tokens.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Compute dtype.",
    )
    _args = parser.parse_args()
    main(
        model_id_or_dir=_args.model_id_or_dir,
        prompt=_args.prompt,
        image_path=_args.image_path,
        max_new_tokens=_args.max_new_tokens,
        temperature=_args.temperature,
        top_p=_args.top_p,
        cache_size=_args.cache_size,
        dtype=_args.dtype,
    )
