"""Qwen3-VL with a bounding-box regression head (experiment 1).

The cheapest possible alternative to emitting coordinates as text: run the
backbone once over (image, query), pool a single hidden vector, and regress the
four box coordinates with a small MLP.  Nothing about the backbone changes --
no new tokens, no new attention pattern, no fine-tuning; its LM head is not
even used.

    BBoxHead      LayerNorm -> Linear -> GELU -> Linear -> sigmoid
    Qwen3VLBBox   Qwen3VL (frozen) + BBoxHead

Boxes are ``(x0, y0, x1, y1)`` normalised to ``[0, 1]``, matching the
convention of ``ridcl/vqa_kvp10k_synth``.  The head predicts ``(cx, cy, w, h)``
and converts to corners: putting a sigmoid on the corners directly lets the
model emit ``x1 < x0``, a box with negative area that L1 is happy with and IoU
is not.

The head runs in float32 even though the backbone is bfloat16 -- it is a few
million parameters, and bf16 has ~3 decimal digits of mantissa, which is the
same order as the coordinate precision we are trying to learn.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from fabrique.models.qwen3vl.model import Qwen3VL
from fabrique.models.qwen3vl.vision import VisionGridData

_EPS = 1e-6

# Filter selecting the trainable parameters of a Qwen3VLBBox: everything under
# `.head`, nothing under `.backbone`.  Pass as `wrt=` to nnx.Optimizer or as the
# diff filter of nnx.value_and_grad to freeze the backbone in place.
HEAD_ONLY = nnx.All(nnx.Param, nnx.PathContains("head"))


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


class BBoxHead(nnx.Module):
    """2-layer MLP mapping one pooled hidden vector to one bounding box."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 1024,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        # Qwen hidden states carry a handful of outlier channels with magnitudes
        # in the hundreds; without a norm here the first Linear spends most of
        # training undoing them.
        self.norm = nnx.LayerNorm(embed_dim, param_dtype=dtype, rngs=rngs)
        self.fc1 = nnx.Linear(embed_dim, hidden_dim, param_dtype=dtype, rngs=rngs)
        # Small init on the output layer so training starts from sigmoid(~0) --
        # a centred half-size box -- instead of somewhere on the saturated
        # flanks, where the gradient is zero and the head never recovers.
        self.fc2 = nnx.Linear(
            hidden_dim,
            4,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
        )
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        """[B, embed_dim] -> [B, 4] boxes ``(x0, y0, x1, y1)``, unclipped."""
        x = x.astype(self.dtype)
        h = nnx.gelu(self.fc1(self.norm(x)))
        return cxcywh_to_xyxy(nnx.sigmoid(self.fc2(h)))


# ---------------------------------------------------------------------------
# Backbone plumbing
# ---------------------------------------------------------------------------


def pool_hidden(
    hidden: jax.Array,  # [B, L, D]
    padding_mask: jax.Array,  # [B, L]
    mode: str = "last",
) -> jax.Array:
    """Reduce per-token hidden states to one vector per sequence, ``[B, D]``.

    ``last`` reads the final real token.  With ``add_generation_prompt=True``
    that is the ``<|im_start|>assistant`` block, i.e. the state the model is in
    when it is about to answer -- the same vector the LM head would read.
    ``mean`` averages all real tokens, which dilutes the query but keeps more of
    the image.
    """
    mask = padding_mask.astype(hidden.dtype)
    if mode == "mean":
        total = jnp.sum(hidden * mask[..., None], axis=1)
        return total / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), _EPS)
    if mode != "last":
        raise ValueError(f"unknown pooling mode: {mode!r}")
    # Works for either padding side: first True scanning from the right.
    seq_len = padding_mask.shape[-1]
    idx = (seq_len - 1) - jnp.argmax(padding_mask[:, ::-1].astype(jnp.int32), axis=-1)
    return jnp.take_along_axis(hidden, idx[:, None, None], axis=1)[:, 0, :]


class Qwen3VLBBox(nnx.Module):
    """Frozen Qwen3-VL + a bounding-box head.

    The backbone is stored as a submodule so the whole thing checkpoints and
    shards as one object, but it is never differentiated: restrict the optimizer
    to ``HEAD_ONLY``.
    """

    def __init__(
        self,
        backbone: Qwen3VL,
        *,
        rngs: nnx.Rngs,
        head_hidden_dim: int = 1024,
        pool: str = "last",
        head_dtype: jnp.dtype = jnp.float32,
    ):
        self.backbone = backbone
        self.head = BBoxHead(
            backbone.config.embed_dim,
            head_hidden_dim,
            rngs=rngs,
            dtype=head_dtype,
        )
        self.pool = pool

    def features(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        pixel_values: jax.Array | None,
        vision_grid: VisionGridData | None,
        padding_mask: jax.Array,
    ) -> jax.Array:
        """Pooled backbone features ``[B, D]``, detached from the graph."""
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
        )
        assert out.hidden_states is not None  # requested above
        return jax.lax.stop_gradient(
            pool_hidden(out.hidden_states, padding_mask, self.pool)
        )

    def __call__(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        pixel_values: jax.Array | None,
        vision_grid: VisionGridData | None,
        padding_mask: jax.Array,
    ) -> jax.Array:
        """[B, 4] predicted boxes ``(x0, y0, x1, y1)``; see ``clip_boxes``."""
        return self.head(
            self.features(
                input_tokens, positions, pixel_values, vision_grid, padding_mask
            )
        )


# ---------------------------------------------------------------------------
# Box geometry and losses
# ---------------------------------------------------------------------------


def cxcywh_to_xyxy(boxes: jax.Array) -> jax.Array:
    """[..., 4] ``(cx, cy, w, h)`` -> ``(x0, y0, x1, y1)``.

    Deliberately *not* clipped to the unit square.  A sigmoid already bounds the
    inputs, so corners land in (-0.5, 1.5), and clipping the training path zeroes
    the gradient of any corner that leaves the image -- combined with a saturated
    sigmoid that is unrecoverable, which is exactly how the first version of this
    head died.  Clip with ``clip_boxes`` when reporting or drawing instead.
    """
    cx, cy, w, h = jnp.split(boxes, 4, axis=-1)
    return jnp.concatenate([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def clip_boxes(boxes: jax.Array) -> jax.Array:
    """Clamp boxes to the unit square, for evaluation and visualisation."""
    return jnp.clip(boxes, 0.0, 1.0)


def _area(boxes: jax.Array) -> jax.Array:
    wh = jnp.clip(boxes[..., 2:] - boxes[..., :2], 0.0)
    return wh[..., 0] * wh[..., 1]


def box_iou(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Row-wise IoU of two ``[..., 4]`` xyxy arrays -> ``[...]``."""
    iou, _ = _iou_and_union(pred, target)
    return iou


def _iou_and_union(pred: jax.Array, target: jax.Array) -> tuple[jax.Array, jax.Array]:
    lt = jnp.maximum(pred[..., :2], target[..., :2])
    rb = jnp.minimum(pred[..., 2:], target[..., 2:])
    wh = jnp.clip(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    union = _area(pred) + _area(target) - inter
    return inter / jnp.maximum(union, _EPS), union


def generalized_box_iou(pred: jax.Array, target: jax.Array) -> jax.Array:
    """GIoU: IoU minus the fraction of the enclosing box that is neither box.

    Unlike IoU it stays informative when the boxes do not overlap, which is the
    entire first phase of training a randomly initialised head.
    """
    iou, union = _iou_and_union(pred, target)
    lt = jnp.minimum(pred[..., :2], target[..., :2])
    rb = jnp.maximum(pred[..., 2:], target[..., 2:])
    wh = jnp.clip(rb - lt, 0.0)
    enclosing = wh[..., 0] * wh[..., 1]
    return iou - (enclosing - union) / jnp.maximum(enclosing, _EPS)


def bbox_loss(
    pred: jax.Array,  # [B, 4] xyxy
    target: jax.Array,  # [B, 4] xyxy
    *,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> jax.Array:
    """DETR's box loss: weighted sum of L1 and GIoU, averaged over the batch.

    The weights are DETR's defaults.  L1 alone under-penalises errors on small
    boxes (most document fields are small), GIoU alone is scale-free but has a
    weak gradient once the boxes overlap; the sum covers both.
    """
    l1 = jnp.sum(jnp.abs(pred - target), axis=-1)
    giou = 1.0 - generalized_box_iou(pred, target)
    return jnp.mean(l1_weight * l1 + giou_weight * giou)
