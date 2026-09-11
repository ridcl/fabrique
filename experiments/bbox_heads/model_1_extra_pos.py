"""Read Qwen3-VL at the position where it would emit a box (1 box).

Naming follows the others: the ``1`` is the number of predicted boxes.

The idea, after the discussion
-----------------------------
The previous three models all read the backbone somewhere it was never trained
to be informative -- a pooled vector, or (in the first draft of this file) a
random learned vector at an appended position, which corresponds to no token the
model has ever seen.  Frozen attention has no learned behaviour for such an
input, so a frozen control over it would measure the wrong thing.

Qwen3-VL was pretrained on grounding data using its own markup::

    <|object_ref_start|> the cat <|object_ref_end|> <|box_start|> coords <|box_end|>

So the prompt here ends with the model's own grounding prefix, and the box is
read from the hidden state at ``<|box_start|>`` -- the exact position whose next
token, in pretraining, is a coordinate.  Nothing foreign is injected: the
"slot" is a zero-initialised *offset* added to a real token's embedding, so at
step 0 this is precisely the frozen model read at its own box position, and
training can only move it away from there.

That makes the frozen run a real question -- "does Qwen3-VL already encode the
answer where it would emit it?" -- rather than "can a random probe vector learn
something", which the earlier probes already answered with no.

Readout is a LayerNorm and one Linear: a projection, not a head.  The
transformer does the work.

For M queries later, repeat the markup per query and read each
``<|box_start|>``; sequence position alone binds each readout to its query.

Cost
----
Gradients reach ``slot_delta`` through all 36 layers, so the backbone forward
belongs inside the training loop and features cannot be precomputed.  Measured
at 1.23 s/example for forward+backward with ``RematConfig.BLOCK``, against
0.36 s/example for a frozen forward pass done once.
"""

from __future__ import annotations

import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from flax import nnx

from experiments.bbox_heads import model_1
from fabrique.models.qwen3vl.model import Qwen3VL
from fabrique.models.qwen3vl.vision import VisionGridData

# Qwen3-VL's own grounding markup, straight from its tokenizer.
OBJECT_REF_START_ID = 151646
OBJECT_REF_END_ID = 151647
BOX_START_ID = 151648

# Median ground-truth box over the dataset (width 0.118, height 0.011 -- a
# single line of text).  Biases the size output so training starts at a
# plausibly shaped box rather than the half-page box a zero bias gives.
MEDIAN_BOX_SIZE = (0.118, 0.011)

# Everything outside `.backbone`: the slot offset and the readout projection.
TRAINABLE = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("backbone")))

# ...plus the LoRA adapters, which live *inside* `.backbone` and so are excluded
# by the filter above.  qwix gives them their own variable type, which is a
# cleaner handle than matching parameter names (they are called `w_lora_a` and
# `kernel_lora_a`, so a "lora_a" path filter matches nothing -- nnx path filters
# match whole segments).
TRAINABLE_WITH_LORA = nnx.Any(TRAINABLE, nnx.LoRAParam)

# Language-model projections only.  The vision tower names its projections
# `qkv_proj` and `out_proj`, neither of which matches these patterns, so the
# encoder stays frozen: the question is whether the *language* side can learn to
# route the box out of visual features it already has.
LORA_TARGETS = ".*q_proj|.*k_proj|.*gate_proj|.*up_proj|.*down_proj"


def add_lora(backbone: Qwen3VL, *, rank: int, alpha: float, rngs: nnx.Rngs) -> Qwen3VL:
    """Wrap the backbone's LM projections with LoRA adapters.

    Apply this *before* constructing :class:`SlotBoxModel`: qwix traces the
    module it is given, and `get_model_input()` describes the backbone's
    signature, not the wrapper's.

    ``lora_b`` is zero-initialised, so the adapters start as the identity and
    step 0 is still exactly the frozen model read at its own ``<|box_start|>``.
    ``rngs`` is not optional in practice -- omitting it makes qwix warn and skip
    initialising ``lora_a``.
    """
    provider = qwix.LoraProvider(module_path=LORA_TARGETS, rank=rank, alpha=alpha)
    # qwix is typed for both linen and nnx, so its return is a union; here it is
    # the same nnx module it was handed, with adapters spliced in.
    return cast(
        Qwen3VL,
        qwix.apply_lora_to_model(
            backbone, provider, rngs=rngs, **backbone.get_model_input()
        ),
    )


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def grounding_suffix(query: str) -> str:
    """The model's own grounding prefix, to append after the generation prompt.

    Ends at ``<|box_start|>`` so the final position is the one whose next token
    would be a coordinate.  Tokenises to 5 tokens for a two-word query.
    """
    return f"<|object_ref_start|>{query}<|object_ref_end|><|box_start|>"


def ensure_even_length(
    input_tokens: np.ndarray,  # [B, L]
    input_mask: np.ndarray,  # [B, L]
    positions: np.ndarray,  # [3, B, L]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad to an even sequence length, masked out.

    cuDNN's flash-attention kernel refuses an odd sequence length when training
    with a mask (``check_is_flash_attention``: "Unsupported sequence length
    Q 785, KV 785").  None of the frozen-feature models hit this because none of
    them differentiated through attention.  The filler is padding, so it changes
    nothing except the shape.
    """
    batch, length = input_tokens.shape
    if length % 2 == 0:
        return input_tokens, input_mask, positions
    pad = np.zeros((batch, 1), input_tokens.dtype)
    return (
        np.concatenate([input_tokens, pad], axis=1),
        np.concatenate([input_mask, np.zeros((batch, 1), input_mask.dtype)], axis=1),
        np.concatenate([positions, np.zeros((3, batch, 1), positions.dtype)], axis=2),
    )


def check_slot_tokens(input_tokens: np.ndarray) -> None:
    """Fail loudly if any row does not carry exactly one ``<|box_start|>``.

    The readout takes ``argmax`` over the match, which silently returns index 0
    -- the start of the image -- when there is no match at all.  Truncation or a
    prompt-format change would produce plausible-looking garbage otherwise.
    """
    counts = (input_tokens == BOX_START_ID).sum(axis=1)
    if counts.min() != 1 or counts.max() != 1:
        raise ValueError(
            f"expected exactly one <|box_start|> per example, got counts "
            f"{np.unique(counts).tolist()}; check the prompt or --max-seq-len"
        )


class SlotBoxModel(nnx.Module):
    """Qwen3-VL read at ``<|box_start|>``, projected to one box.

    Restrict the optimizer to :data:`TRAINABLE` to keep the backbone frozen;
    swap that for a LoRA filter to adapt it.
    """

    def __init__(
        self,
        backbone: Qwen3VL,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        init_size: tuple[float, float] = MEDIAN_BOX_SIZE,
    ):
        self.backbone = backbone
        self.dtype = dtype
        embed_dim = backbone.config.embed_dim

        # Zero init: at step 0 the sequence is exactly what the tokenizer
        # produced, so the run starts as the frozen model read at its own box
        # position and can only move away from there deliberately.
        self.slot_delta = nnx.Param(jnp.zeros((embed_dim,), dtype))
        self.norm = nnx.LayerNorm(embed_dim, param_dtype=dtype, rngs=rngs)
        # Centre bias 0 -> sigmoid 0.5 -> page centre; size biased at the
        # dataset median.  Small kernel init so the first step starts there
        # rather than on a saturated sigmoid flank.
        box_bias = jnp.array(
            [0.0, 0.0, _logit(init_size[0]), _logit(init_size[1])], dtype=dtype
        )
        self.proj = nnx.Linear(
            embed_dim,
            4,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
            bias_init=lambda _key, _shape, dt=dtype: box_bias.astype(dt),
        )

    def slot_hidden(
        self,
        input_tokens: jax.Array,  # [B, L]
        positions: jax.Array,  # [3, B, L]
        pixel_values: jax.Array | None,
        vision_grid: VisionGridData | None,
        padding_mask: jax.Array,  # [B, L]
    ) -> jax.Array:
        """Final hidden state at the ``<|box_start|>`` position, ``[B, D]``."""
        slot_index = jnp.argmax(input_tokens == BOX_START_ID, axis=-1)  # [B]
        rows = jnp.arange(input_tokens.shape[0])

        embeds = self.backbone.embedder.encode(input_tokens)
        delta = self.slot_delta.astype(embeds.dtype)
        embeds = embeds.at[rows, slot_index].add(delta)

        out = self.backbone(
            input_tokens,
            positions,
            pixel_values,
            vision_grid,
            None,  # cache
            padding_mask,
            output_hidden_states=True,
            # The [B, L, vocab_size] projection is unused here.  Merely wasteful
            # on a frozen forward pass, but retained as an autodiff residual --
            # and enough to OOM a 24 GB card -- once gradients flow back through
            # the backbone.
            skip_lm_head=True,
            inputs_embeds=embeds,
        )
        assert out.hidden_states is not None  # requested above
        return out.hidden_states[rows, slot_index]

    def __call__(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        pixel_values: jax.Array | None,
        vision_grid: VisionGridData | None,
        padding_mask: jax.Array,
    ) -> jax.Array:
        """``[B, 4]`` boxes ``(x0, y0, x1, y1)``; see ``model_1.clip_boxes``."""
        hidden = self.slot_hidden(
            input_tokens, positions, pixel_values, vision_grid, padding_mask
        )
        raw = self.proj(self.norm(hidden.astype(self.dtype)))
        return model_1.cxcywh_to_xyxy(nnx.sigmoid(raw))
