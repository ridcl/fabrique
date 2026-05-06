"""DiT — Diffusion Transformer for pixel-space image generation."""

from fabrique.models.dit import params
from fabrique.models.dit.model import (
    MLP,
    Attention,
    DiT,
    DiTBlock,
    DiTConfig,
    PatchEmbed,
    TimestepEmbedding,
    compute_rope,
    dit_base,
    dit_qwen,
    dit_small,
    patchify,
    unpatchify,
)

__all__ = [
    "params",
    "DiT",
    "DiTBlock",
    "DiTConfig",
    "MLP",
    "Attention",
    "PatchEmbed",
    "TimestepEmbedding",
    "compute_rope",
    "dit_base",
    "dit_qwen",
    "dit_small",
    "patchify",
    "unpatchify",
]
