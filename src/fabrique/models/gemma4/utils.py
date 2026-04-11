# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoding utilities for Gemma 4: batch and conversation encoders.

Usage example::

    from fabrique.models.gemma4 import utils as gemma4_utils
    from fabrique.models.gemma4 import model as model_lib

    processor = gemma4_utils.load_processor('/path/to/checkpoint')
    cfg = model_lib.ModelConfig.gemma4_e4b_it()

    batch = gemma4_utils.encode_batch(
        processor,
        texts=['<bos>Hello, world!'],
        images=[[]],          # no images for this example
        vcfg=cfg.vision_config,
        max_length=512,
    )
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from flax import struct
from transformers import AutoProcessor

from fabrique.models.gemma4 import vision as vision_lib


def load_processor(model_dir: str) -> AutoProcessor:
    """Load a Gemma 4 processor from a local checkpoint directory.

    Args:
      model_dir: Local directory containing a Gemma 4 checkpoint.

    Returns:
      An ``AutoProcessor`` instance for Gemma 4.
    """
    return AutoProcessor.from_pretrained(model_dir)


@struct.dataclass
class EncodedBatch:
    """Output of encode_batch / encode_messages.

    B = batch size, L = max sequence length,
    N = total patch tokens per image (padded to a common size).

    Attributes:
      input_tokens:       [B, L]    int32   — token ids (right-padded)
      input_mask:         [B, L]    bool    — True at non-padding positions
      completion_mask:    [B, L]    bool    — True at tokens to include in loss
      positions:          [B, L]    int32   — 1-D token positions
      pixel_values:       [B, N, C] float32 — patchified image pixels in [0, 1]
                                    (None if batch contains no images)
      pixel_position_ids: [B, N, 2] int32   — (x, y) patch coords; -1 = padding
                                    (None if batch contains no images)
    """

    input_tokens: np.ndarray  # [B, L]
    input_mask: np.ndarray  # [B, L]
    completion_mask: np.ndarray  # [B, L]
    positions: np.ndarray  # [B, L]
    pixel_values: np.ndarray | None  # [B, N, C]
    pixel_position_ids: np.ndarray | None  # [B, N, 2]


def encode_batch(
    processor: AutoProcessor,
    texts: list[str],
    images: list[list[Any]],
    *,
    vcfg: vision_lib.VisionConfig | None = None,
    max_length: int,
    padding: bool | str = True,
    truncation: bool | str = True,
    pad_to_multiple_of: int | None = None,
    padding_side: str | None = None,
) -> EncodedBatch:
    """Encode a batch of pre-formatted texts with corresponding image lists.

    Each ``texts[i]`` is a fully-formatted prompt string (e.g. the output of
    ``processor.apply_chat_template``).  ``images[i]`` is the list of PIL images
    for that item.  If an item has no images, pass an empty list.

    The ``vcfg`` argument is accepted but currently unused; positions are derived
    directly from the processor output.

    Args:
      processor: HuggingFace AutoProcessor for Gemma 4.
      texts: List of B formatted prompt strings.
      images: List of B image lists (each inner list may be empty).
      vcfg: VisionConfig (accepted for API parity with Qwen3-VL utils; unused).
      max_length: Maximum sequence length; longer sequences are truncated.
      padding: Passed to the processor (e.g. True, "max_length").
      truncation: Passed to the processor.
      pad_to_multiple_of: If set, pad sequence length to the next multiple.
      padding_side: If set, temporarily overrides the tokenizer's padding_side.

    Returns:
      EncodedBatch
    """
    flat_images = [img for imgs in images for img in imgs]
    tok = processor.tokenizer
    _orig_padding_side = tok.padding_side
    if padding_side is not None:
        tok.padding_side = padding_side
    try:
        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,
        )
    finally:
        tok.padding_side = _orig_padding_side

    input_ids = np.array(inputs["input_ids"], dtype=np.int32)  # [B, L]
    input_mask = np.array(inputs["attention_mask"], dtype=bool)  # [B, L]

    B, L = input_ids.shape
    # Positions: cumulative count of non-padding tokens seen so far per row.
    # For left-padded sequences the positions start at 0 for the first real token.
    cum = np.cumsum(input_mask, axis=1) - 1
    positions = np.where(input_mask, cum, 0).astype(np.int32)  # [B, L]

    if flat_images:
        pixel_values = np.array(inputs["pixel_values"], dtype=np.float32)
        pixel_position_ids = np.array(inputs["image_position_ids"], dtype=np.int32)
    else:
        pixel_values = None
        pixel_position_ids = None

    return EncodedBatch(
        input_tokens=input_ids,
        input_mask=input_mask,
        completion_mask=np.zeros((B, L), dtype=bool),
        positions=positions,
        pixel_values=pixel_values,
        pixel_position_ids=pixel_position_ids,
    )


def encode_messages(
    processor: AutoProcessor,
    conversations: list[list[dict[str, Any]]],
    loss_roles: set[str],
    *,
    vcfg: vision_lib.VisionConfig | None = None,
    max_seq_len: int,
    padding: bool | str = True,
    truncation: bool | str = True,
    pad_to_multiple_of: int | None = None,
) -> EncodedBatch:
    """Encode OpenAI-format conversations with per-role loss masking.

    Args:
      processor: HuggingFace AutoProcessor for Gemma 4.
      conversations: List of B conversations, each a list of message dicts with
        keys ``role`` (str) and ``content`` (str or list of content blocks).
      loss_roles: Set of role names whose tokens are included in the loss (e.g.
        ``{"model"}``).  Tokens from all other roles are masked out.
      vcfg: VisionConfig (accepted for API parity; unused).
      max_seq_len: Maximum sequence length.
      padding: Passed to the processor.
      truncation: Passed to the processor.
      pad_to_multiple_of: If set, pad to the next multiple.

    Returns:
      EncodedBatch with completion_mask set according to loss_roles.
    """
    texts: list[str] = []
    all_images: list[list[Any]] = []

    for conv in conversations:
        images: list[Any] = []
        for msg in conv:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        img = block.get("image")
                        if img is not None:
                            images.append(img)
        texts.append(
            processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
        )
        all_images.append(images)

    batch = encode_batch(
        processor,
        texts,
        all_images,
        vcfg=vcfg,
        max_length=max_seq_len,
        padding=padding,
        truncation=truncation,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Build the completion mask by scanning the expanded input_ids for turn
    # boundaries.  Gemma 4 uses <start_of_turn> / <end_of_turn> special tokens.
    start_id = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
    end_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")

    # Map the first sub-token of each role name to its role string.
    role_first_token: dict[int, str] = {}
    for role in ["user", "model", "system", "tool"]:
        toks = processor.tokenizer.encode(role, add_special_tokens=False)
        if toks:
            role_first_token[toks[0]] = role

    B, L = batch.input_tokens.shape
    comp_masks = np.zeros((B, L), dtype=bool)
    for b in range(B):
        ids = batch.input_tokens[b]
        i = 0
        while i < L:
            if ids[i] == start_id and i + 1 < L:
                role = role_first_token.get(int(ids[i + 1]))
                # Scan forward to find the closing <end_of_turn>.
                j = i + 1
                while j < L and ids[j] != end_id:
                    j += 1
                if role in loss_roles:
                    comp_masks[b, i : j + 1] = True
                i = j + 1
            else:
                i += 1

    return EncodedBatch(
        input_tokens=batch.input_tokens,
        input_mask=batch.input_mask,
        completion_mask=comp_masks,
        positions=batch.positions,
        pixel_values=batch.pixel_values,
        pixel_position_ids=batch.pixel_position_ids,
    )
