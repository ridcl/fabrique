"""Parity test for the Qwen3-VL incremental decode path.

``consistency_test.py`` compares the JAX model against HuggingFace, but only for
a *full forward pass* — it never exercises the KV cache.  That let a decode bug
through: the decode step built its attention mask as ``cache_pos <= end_index``,
attending to every filled cache slot including the left-padding written during
prefill.  Generation stayed correct for ~12-25 tokens and then degraded, which
is easy to mistake for a weak model (see doc_vqa_vllm_crosscheck.py).

The check here needs no checkpoint, no GPU and no PyTorch: under greedy decoding
an incremental decode must produce exactly the tokens a single teacher-forced
full forward pass would, because both compute the same conditional. Any
cache/mask/position bug breaks that identity.

    pytest tests/qwen3vl_decode_test.py
    python tests/qwen3vl_decode_test.py     # verbose, standalone
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tiny model; keep off the GPUs

import jax.numpy as jnp
import numpy as np
from flax import nnx

from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl.vision import VisionModelConfig

CACHE_SIZE = 32
N_DECODE = 12


def _tiny_model(seed: int = 0) -> model_lib.Qwen3VL:
    config = model_lib.ModelConfig(
        num_layers=2,
        vocab_size=64,
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        head_dim=8,
        num_kv_heads=2,
        rope_theta=10_000,
        norm_eps=1e-6,
        param_dtype=jnp.float32,  # exact comparison, no bf16 noise
        # A vision config is attached only so that mRoPE is configured the way
        # it is for real checkpoints (apply_rope needs mrope_section); no image
        # is ever passed, so the tower is never built or run.
        vision_config=VisionModelConfig(
            hidden_size=8,
            out_hidden_size=32,
            depth=1,
            num_heads=1,
            intermediate_size=8,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            window_size=16,
            in_channels=3,
            num_position_embeddings=4,
            deepstack_visual_indexes=(),
            mrope_section=(2, 1, 1),  # sums to head_dim/2 = 4
            image_pad_id=63,
        ),
    )
    return model_lib.Qwen3VL(config, rngs=nnx.Rngs(seed))


def _positions(pad_mask: jnp.ndarray) -> jnp.ndarray:
    """3D M-RoPE positions for a text-only, left-padded batch.

    Real tokens are numbered from 0; padding slots collapse to 0 and are excluded
    by the padding mask.  All three M-RoPE axes coincide for text.
    """
    pos = jnp.maximum(jnp.cumsum(pad_mask, axis=-1) - 1, 0)
    return jnp.stack([pos, pos, pos])  # [3, B, L]


def _left_padded_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two sequences with *different* amounts of left padding.

    Unequal padding is what makes the bug observable: with none, there are no
    junk cache slots to leak into attention.
    """
    tokens = jnp.array(
        [[0, 0, 5, 9, 3, 7, 12, 4], [0, 0, 0, 0, 0, 11, 4, 20]], dtype=jnp.int32
    )
    pad_mask = jnp.array(
        [[0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1]], dtype=jnp.int32
    )
    return tokens, pad_mask


def _greedy_decode(model, tokens, pad_mask, n_steps):
    """Prefill + incremental decode; returns the generated token ids [B, n]."""
    batch, _ = tokens.shape
    positions = _positions(pad_mask)
    cache = model.init_cache(batch, CACHE_SIZE, jnp.float32)
    logits, cache = model(
        tokens, positions, None, None, cache, pad_mask.astype(jnp.bool_)
    )
    next_pos = jnp.max(positions[0], axis=-1) + 1  # [B]

    generated = []
    for _ in range(n_steps):
        token = jnp.argmax(logits[:, -1, :], axis=-1)  # [B]
        generated.append(token)
        pos = next_pos[:, None]
        logits, cache = model(
            token[:, None], jnp.stack([pos, pos, pos]), None, None, cache, None
        )
        next_pos = next_pos + 1
    return jnp.stack(generated, axis=1)  # [B, n_steps]


def _teacher_forced_logits(model, tokens, pad_mask, generated):
    """One full forward pass (cache=None) over prompt + generated tokens."""
    batch, n_gen = generated.shape
    full_tokens = jnp.concatenate([tokens, generated], axis=1)
    full_pad = jnp.concatenate(
        [pad_mask, jnp.ones((batch, n_gen), dtype=pad_mask.dtype)], axis=1
    )
    logits, _ = model(
        full_tokens,
        _positions(full_pad),
        None,
        None,
        None,
        full_pad.astype(jnp.bool_),
    )
    return logits


def test_decode_matches_full_forward():
    """Greedy incremental decode must equal teacher-forced full forward."""
    model = _tiny_model()
    tokens, pad_mask = _left_padded_batch()
    prompt_len = tokens.shape[1]

    generated = _greedy_decode(model, tokens, pad_mask, N_DECODE)
    logits = _teacher_forced_logits(model, tokens, pad_mask, generated)

    # Position prompt_len-1+i predicts the i-th generated token.
    expected = jnp.argmax(
        logits[:, prompt_len - 1 : prompt_len - 1 + N_DECODE, :], axis=-1
    )
    mismatch = np.asarray(generated != expected)
    assert not mismatch.any(), (
        "incremental decode diverged from full forward at step(s) "
        f"{sorted(set(np.nonzero(mismatch)[1].tolist()))}\n"
        f"  decode:       {np.asarray(generated).tolist()}\n"
        f"  full forward: {np.asarray(expected).tolist()}"
    )


def test_padding_slots_do_not_leak_into_decode():
    """Decode output must not depend on what sits in the padding slots.

    Poisoning the padded token ids changes the junk KV written during prefill.
    A correct mask ignores those slots, so the generated tokens are unchanged.
    """
    model = _tiny_model()
    tokens, pad_mask = _left_padded_batch()

    poisoned = jnp.where(pad_mask == 0, jnp.int32(63), tokens)
    clean_out = _greedy_decode(model, tokens, pad_mask, N_DECODE)
    poisoned_out = _greedy_decode(model, poisoned, pad_mask, N_DECODE)

    assert np.array_equal(np.asarray(clean_out), np.asarray(poisoned_out)), (
        "padding-slot contents changed the decoded tokens — the decode mask is "
        "attending to padding\n"
        f"  clean:    {np.asarray(clean_out).tolist()}\n"
        f"  poisoned: {np.asarray(poisoned_out).tolist()}"
    )


def test_unpadded_decode_matches_full_forward():
    """Control: with no padding the two paths must also agree."""
    model = _tiny_model()
    tokens = jnp.array([[5, 9, 3, 7, 12, 4]], dtype=jnp.int32)
    pad_mask = jnp.ones_like(tokens)

    generated = _greedy_decode(model, tokens, pad_mask, N_DECODE)
    logits = _teacher_forced_logits(model, tokens, pad_mask, generated)
    expected = jnp.argmax(
        logits[:, tokens.shape[1] - 1 : tokens.shape[1] - 1 + N_DECODE, :], axis=-1
    )
    assert np.array_equal(np.asarray(generated), np.asarray(expected))


if __name__ == "__main__":
    for fn in (
        test_unpadded_decode_matches_full_forward,
        test_decode_matches_full_forward,
        test_padding_slots_do_not_leak_into_decode,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}\n      {exc}")
