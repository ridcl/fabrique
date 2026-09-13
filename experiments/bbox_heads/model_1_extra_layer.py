"""One learnable pseudo-token that attends over the image tokens (1 box).

The naming follows ``model_1.py``: the ``1`` is the number of predicted boxes,
not a version number.  This is the minimal version of the pseudo-token idea --
N=1 token, one cross-attention layer over the *last* layer's image tokens --
built to test one hypothesis cheaply before committing to a multi-layer stack.

Why this and not another MLP
----------------------------
``model_1`` pools the sequence into one vector and regresses from it.  Both
pooling modes failed, in opposite ways: ``last`` memorised 605 training
examples (train IoU 0.32) but transferred nothing (eval IoU 0.010 vs a 0.002
constant-box baseline), while ``mean`` could not even memorise (train IoU 0.016)
and landed *worse* than the constant box on centre distance.  Neither failure is
about head capacity, which points at the readout: a pooled vector has no
position axis, and position is the answer.

So here the head keeps the axis.  A single pseudo-token, initialised from the
query-conditioned pooled vector, cross-attends over the ~736 image tokens.  Those
sit on a known grid -- Qwen3-VL merges 2x2 patches of 16 px, so token *i* covers
a 32x32 px cell at ``(i // grid_w, i % grid_w)`` -- which makes the attention
distribution directly interpretable as "where on the page is this field", and
makes its soft-argmax a coordinate rather than something the MLP has to decode.

Readout
-------
    centre = softmax(attn) . grid_centres     + bounded delta
    size   = sigmoid(MLP), biased at the dataset's median box

The centre is therefore correct by construction whenever attention lands on the
right cell, and the delta only has to fix sub-cell error.  ``__call__`` also
returns the attention map, so a failure is visible rather than inferred: the
training script renders it over the page.

Everything except the head is frozen, and the head is a strict superset of
``model_1``'s -- the pooled vector is still part of the MLP input, so attention
that learns nothing useful degrades to the previous model rather than below it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from experiments.bbox_heads import model_1
from fabrique.models.qwen3vl.model import Qwen3VL
from fabrique.models.qwen3vl.vision import VisionGridData

# Median ground-truth box over the dataset (measured across 608 single-answer
# answers: width p50 0.118, height p50 0.011 -- a single line of text).  Used to
# bias the size output so training starts at a plausibly shaped box instead of
# the half-page box a zero bias would give.
MEDIAN_BOX_SIZE = (0.118, 0.011)

# Maximum correction applied on top of the attention centroid.  Must be >= 1.0:
# with a smaller bound the reachable centres are [centroid +- d], and a
# near-uniform attention map at init puts the centroid at the page centre, so
# d=0.25 made 68% of this dataset's targets unreachable on step 1 -- the head
# could only escape through the attention gradient, which biases it towards
# faking the centroid with a diffuse map instead of pointing at the field.
MAX_CENTRE_DELTA = 1.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def gather_image_tokens(
    hidden: jax.Array,  # [B, L, D]
    mask: jax.Array,  # [B, L] True at image-pad positions
    n_tokens: int,  # static
) -> jax.Array:
    """Pull the ``n_tokens`` image-token hidden states out of the sequence.

    ``n_tokens`` has to be a Python int -- it is the output shape, so it cannot
    depend on traced values.  Every page is resized to the same dimensions, so
    it is constant for a run; the training script asserts that.
    """

    def _gather(h, m):
        idx = jnp.where(m, size=n_tokens, fill_value=0)[0]
        return h[idx]

    return jax.vmap(_gather)(hidden, mask)


class BBoxAttnHead(nnx.Module):
    """N=1 pseudo-token, one cross-attention layer, box readout."""

    def __init__(
        self,
        embed_dim: int,
        n_image_tokens: int,
        grid_hw: tuple[int, int],
        *,
        rngs: nnx.Rngs,
        width: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 1024,
        dtype: jnp.dtype = jnp.float32,
        init_size: tuple[float, float] = MEDIAN_BOX_SIZE,
    ):
        if width % num_heads:
            raise ValueError(f"width {width} not divisible by num_heads {num_heads}")
        self.width = width
        self.num_heads = num_heads
        self.grid_hw = grid_hw
        self.n_image_tokens = n_image_tokens
        self.dtype = dtype

        # Backbone hidden states carry outlier channels in the hundreds; the
        # projections below are trained from scratch and would spend most of
        # training undoing them.
        self.q_norm = nnx.LayerNorm(embed_dim, param_dtype=dtype, rngs=rngs)
        self.kv_norm = nnx.LayerNorm(embed_dim, param_dtype=dtype, rngs=rngs)

        self.q_proj = nnx.Linear(embed_dim, width, param_dtype=dtype, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, width, param_dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, width, param_dtype=dtype, rngs=rngs)

        # Learned position code added to the keys only (DETR-style): it helps
        # the pseudo-token address cells, while values stay content-only.  The
        # centroid readout is what carries position into the output, so values
        # do not need to encode it.
        self.pos_embed = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                rngs.params(), (n_image_tokens, width), dtype
            )
        )

        # MLP input: projected pooled vector, attended context, centroid (x, y).
        self.fc1 = nnx.Linear(2 * width + 2, hidden_dim, param_dtype=dtype, rngs=rngs)
        # Small init so the centre starts *at* the attention centroid and the
        # size starts at the dataset median, rather than on a sigmoid flank
        # where the gradient is zero (how the first model_1 head died).
        self.fc_delta = nnx.Linear(
            hidden_dim,
            2,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
        )
        size_bias = jnp.array([_logit(init_size[0]), _logit(init_size[1])], dtype=dtype)
        self.fc_size = nnx.Linear(
            hidden_dim,
            2,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
            bias_init=lambda _key, _shape, dt=dtype: size_bias.astype(dt),
        )

    def grid_centres(self) -> jax.Array:
        """[n_image_tokens, 2] centre of each merged patch, normalised to [0, 1]."""
        grid_h, grid_w = self.grid_hw
        ys, xs = jnp.meshgrid(
            jnp.arange(grid_h, dtype=self.dtype),
            jnp.arange(grid_w, dtype=self.dtype),
            indexing="ij",
        )
        return jnp.stack(
            [(xs.ravel() + 0.5) / grid_w, (ys.ravel() + 0.5) / grid_h], axis=-1
        )

    def __call__(
        self,
        pooled: jax.Array,  # [B, D]  last-token hidden state (query conditioning)
        image_tokens: jax.Array,  # [B, n, D]
    ) -> tuple[jax.Array, jax.Array]:
        """Returns ``(boxes [B, 4] xyxy, attention [B, n])``."""
        pooled = pooled.astype(self.dtype)
        img = image_tokens.astype(self.dtype)
        batch, n_tokens, _ = img.shape
        heads, head_dim = self.num_heads, self.width // self.num_heads

        q = self.q_proj(self.q_norm(pooled))  # [B, W]
        kv_in = self.kv_norm(img)
        k = self.k_proj(kv_in) + self.pos_embed  # [B, n, W]
        v = self.v_proj(kv_in)  # [B, n, W]

        qh = q.reshape(batch, heads, head_dim)
        kh = k.reshape(batch, n_tokens, heads, head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(batch, n_tokens, heads, head_dim).transpose(0, 2, 1, 3)

        logits = jnp.einsum("bhd,bhnd->bhn", qh, kh) / math.sqrt(head_dim)
        attn = jax.nn.softmax(logits, axis=-1)  # [B, heads, n]
        ctx = jnp.einsum("bhn,bhnd->bhd", attn, vh).reshape(batch, self.width)

        # Averaged over heads: one interpretable map, and the distribution whose
        # soft-argmax becomes the predicted centre.
        attn_map = jnp.mean(attn, axis=1)  # [B, n]
        centroid = attn_map @ self.grid_centres()  # [B, 2]

        feat = jnp.concatenate([q, ctx, centroid], axis=-1)
        hidden = nnx.gelu(self.fc1(feat))
        centre = centroid + MAX_CENTRE_DELTA * jnp.tanh(self.fc_delta(hidden))
        size = nnx.sigmoid(self.fc_size(hidden))

        boxes = model_1.cxcywh_to_xyxy(jnp.concatenate([centre, size], axis=-1))
        return boxes, attn_map


class Qwen3VLBBoxAttn(nnx.Module):
    """Frozen Qwen3-VL + :class:`BBoxAttnHead`.

    Restrict the optimizer to ``model_1.HEAD_ONLY`` -- the filter matches this
    module's ``.head`` too.
    """

    def __init__(
        self,
        backbone: Qwen3VL,
        *,
        n_image_tokens: int,
        grid_hw: tuple[int, int],
        rngs: nnx.Rngs,
        width: int = 512,
        num_heads: int = 8,
        head_hidden_dim: int = 1024,
        head_dtype: jnp.dtype = jnp.float32,
    ):
        vcfg = backbone.config.vision_config
        if vcfg is None:
            raise ValueError("backbone has no vision config")
        self.backbone = backbone
        self.image_pad_id = vcfg.image_pad_id
        self.n_image_tokens = n_image_tokens
        self.head = BBoxAttnHead(
            backbone.config.embed_dim,
            n_image_tokens,
            grid_hw,
            rngs=rngs,
            width=width,
            num_heads=num_heads,
            hidden_dim=head_hidden_dim,
            dtype=head_dtype,
        )

    def features(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        pixel_values: jax.Array | None,
        vision_grid: VisionGridData | None,
        padding_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """``(pooled [B, D], image_tokens [B, n, D])``, detached from the graph.

        Appended pseudo-tokens could not influence these anyway -- attention is
        causal and they would sit at the end of the sequence -- which is what
        makes it valid to compute this once per example and reuse it across
        every training step.
        """
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
        pooled = model_1.pool_hidden(out.hidden_states, padding_mask, "last")
        image = gather_image_tokens(
            out.hidden_states, input_tokens == self.image_pad_id, self.n_image_tokens
        )
        return jax.lax.stop_gradient(pooled), jax.lax.stop_gradient(image)

    def __call__(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        pixel_values: jax.Array | None,
        vision_grid: VisionGridData | None,
        padding_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """``(boxes [B, 4], attention [B, n])``; see ``model_1.clip_boxes``."""
        pooled, image = self.features(
            input_tokens, positions, pixel_values, vision_grid, padding_mask
        )
        return self.head(pooled, image)
