"""Stage-by-stage parity of the JAX Qwen3-VL vision tower against HuggingFace.

`consistency_test.py` showed that the token embeddings match exactly while the
*vision* token embeddings diverge (fp32 max ~0.74, unchanged from bf16, so it is
a real defect and not rounding).  This script feeds both towers identical
pixel_values and reports where in the tower the divergence appears: patch embed,
positional embedding, each of the 24 blocks, then the merger.

Both frameworks run on CPU in float32 so the comparison is exact.  The vision
tower is small (hidden 1024, depth 24), so this is fast even on CPU.

    python tests/qwen3vl_vision_parity.py --image /tmp/page_small.png
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl import params as params_lib
from fabrique.models.qwen3vl.loading import _load_processor, resolve_model_dir


def _diff(a, b, label):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  {label:<28} SHAPE MISMATCH jax={a.shape} pt={b.shape}")
        return None
    d = np.abs(a - b)
    print(f"  {label:<28} max={d.max():<12.6f} mean={d.mean():<12.6f} shape={a.shape}")
    return d.max()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ridcl/paperwerk-vqa")
    ap.add_argument("--config", default="qwen3vl_4b")
    ap.add_argument("--image", default="/tmp/page_small.png")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model)
    config = getattr(model_lib.ModelConfig, args.config)()
    # NOTE: create_model_from_safe_tensors(dtype=...) sets only the *parameter*
    # dtype; the modules take their compute dtype from config.param_dtype, which
    # defaults to bfloat16.  Without this the comparison silently runs bf16
    # arithmetic and every diff below is just rounding.
    config.param_dtype = jnp.float32

    # --- shared inputs: use HF's processor so pixels are identical -------------
    processor = _load_processor(model_dir)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=["x"], images=[image], return_tensors=None)
    pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
    grid_thw = np.asarray(inputs["image_grid_thw"], dtype=np.int32)
    print(f"grid_thw={grid_thw.tolist()}  pixel_values={pixel_values.shape}\n")

    # --- HF tower -------------------------------------------------------------
    hf_config = AutoConfig.from_pretrained(model_dir)
    tc = getattr(hf_config, "text_config", hf_config)
    if getattr(tc, "rope_scaling", None) is None and getattr(tc, "rope_parameters", None):
        tc.rope_scaling = dict(tc.rope_parameters)
    pt_full = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, config=hf_config, dtype=torch.float32, device_map="cpu"
    )
    pt_full.eval()
    pt = pt_full.model.visual

    # --- JAX tower ------------------------------------------------------------
    jax_full = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=None, dtype=jnp.float32
    )
    jx = jax_full.visual
    grid_data = jx.compute_grid_data(grid_thw)

    pv_pt = torch.from_numpy(pixel_values)
    pv_jx = jnp.asarray(pixel_values)

    print("=== stage-by-stage ===")
    with torch.no_grad():
        # 1. patch embedding
        h_pt = pt.patch_embed(pv_pt)
        h_jx = jx.patch_embed(pv_jx)
        _diff(h_jx, h_pt, "patch_embed")

        # 2. + interpolated position embedding
        pos_pt = pt.fast_pos_embed_interpolate(torch.from_numpy(grid_thw))
        raw = jx.pos_embed.embedding[grid_data.pos_embed_idx]
        pos_jx = (raw * grid_data.pos_embed_weights[..., None]).sum(0)[
            grid_data.pos_embed_gather
        ]
        _diff(pos_jx, pos_pt, "pos_embed (interpolated)")

        h_pt = h_pt + pos_pt.to(h_pt.dtype)
        h_jx = h_jx + pos_jx.astype(h_jx.dtype)
        _diff(h_jx, h_pt, "after pos_embed add")

        # 3. rotary tables
        pos_ids_pt = pt.rot_pos_emb(torch.from_numpy(grid_thw))
        rot_pt = pos_ids_pt.reshape(h_pt.shape[0], -1)
        emb_pt = torch.cat((rot_pt, rot_pt), dim=-1)
        _diff(grid_data.cos, emb_pt.cos(), "rotary cos")
        _diff(grid_data.sin, emb_pt.sin(), "rotary sin")

        # 4. per-block
        cu_pt = torch.from_numpy(np.asarray(grid_data.cu_seqlens))
        pos_emb_pt = (emb_pt.cos(), emb_pt.sin())
        first_bad = None
        for i, (jb, pb) in enumerate(zip(jx.blocks, pt.blocks, strict=True)):
            h_pt = pb(h_pt, cu_seqlens=cu_pt, position_embeddings=pos_emb_pt)
            h_jx = jb(h_jx, grid_data.cos, grid_data.sin, grid_data.cu_seqlens)
            m = _diff(h_jx, h_pt, f"block[{i}]")
            if first_bad is None and m is not None and m > 1e-3:
                first_bad = i

        # 5. merger
        _diff(jx.merger(h_jx), pt.merger(h_pt), "merger (vision tokens)")

    if first_bad is not None:
        print(f"\n>>> divergence first exceeds 1e-3 at vision block {first_bad}")
    else:
        print("\n>>> all blocks agree to 1e-3; look at merger / deepstack")


if __name__ == "__main__":
    main()
