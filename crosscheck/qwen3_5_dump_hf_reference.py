"""Dump HuggingFace Qwen3.5 reference activations to an .npz (torch side).

Run this in an environment that has transformers >= 5.17 and torch; it writes a
golden file that ``qwen3_5_consistency.py`` then compares against in the
JAX environment.  The two are split deliberately: the project's JAX stack pins
transformers 4.57 (which has no ``qwen3_5``), and installing the newer
transformers + CUDA torch into that environment has broken it before.

    /data/hfref-venv/bin/python crosscheck/qwen3_5_dump_hf_reference.py \
        --model Qwen/Qwen3.5-0.8B --dtype float32 \
        --out /data/consistency-tests/qwen3_5_0_8b_fp32.npz

Both paths live under ``/data`` on purpose: it is the only mount that survives a
devcontainer rebuild.  Golden files written to /tmp or the session scratchpad
have been lost to a rebuild before, and regenerating one costs a fresh venv plus
a full fp32 CPU forward pass.

If ``/data/hfref-venv`` is missing, recreate it -- note the CPU torch wheel,
which matters: the CUDA wheel pulls nvidia-cudnn-cu13, whose libcudnn.so.9
overwrites the cu12 copy that this project's JAX needs.

    uv venv /data/hfref-venv --python 3.12
    uv pip install --python /data/hfref-venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        "transformers>=5.17" torch accelerate numpy
"""

import argparse
import json

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = getattr(torch, args.dtype)
    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(cfg, "text_config", cfg)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype, device_map="cpu", attn_implementation="eager"
    )
    model.eval()

    rng = np.random.default_rng(args.seed)
    vocab = int(text_cfg.vocab_size)
    # Stay clear of special/image tokens at the top of the vocabulary.
    tokens = rng.integers(0, min(vocab, 100_000), size=(args.batch, args.seq_len))
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Capture layer-0 internals too, so a mismatch can be bisected inside the
    # first block rather than only observed at its output.
    probes: dict[str, np.ndarray] = {}

    def probe(name):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            probes[name] = t.detach().float().numpy()

        return hook

    # Qwen3_5ForCausalLM.model IS the text model; the multimodal wrapper
    # nests it under .language_model instead.
    lm = getattr(model.model, "language_model", model.model)
    layer0 = lm.layers[0]
    # Hook every decoder layer explicitly.  Relying on output_hidden_states is
    # ambiguous: its last entry is the tensor *after* the final norm, not the
    # last layer's raw output, which silently misaligns the comparison.
    layer_handles = [
        layer.register_forward_hook(probe(f"layer_{i}"))
        for i, layer in enumerate(lm.layers)
    ]
    final_norm_handle = lm.norm.register_forward_hook(probe("final_norm"))
    handles = (
        layer_handles
        + [final_norm_handle]
        + [
            layer0.input_layernorm.register_forward_hook(probe("l0_input_layernorm")),
            layer0.linear_attn.register_forward_hook(probe("l0_linear_attn")),
            layer0.post_attention_layernorm.register_forward_hook(
                probe("l0_post_attention_layernorm")
            ),
            layer0.mlp.register_forward_hook(probe("l0_mlp")),
            layer0.linear_attn.norm.register_forward_hook(probe("l0_gdn_norm")),
            layer0.linear_attn.out_proj.register_forward_hook(probe("l0_gdn_out_proj")),
            layer0.linear_attn.in_proj_qkv.register_forward_hook(
                probe("l0_gdn_in_qkv")
            ),
            layer0.linear_attn.conv1d.register_forward_hook(probe("l0_gdn_conv1d")),
        ]
    )
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
        )
    for h in handles:
        h.remove()

    # hidden_states is (embeddings, layer_1, ..., layer_N); drop the embedding
    # entry so index i is the output of layer i.
    n_layers = len(lm.layers)
    hidden = [probes[f"layer_{i}"] for i in range(n_layers)]
    payload = {
        "input_ids": tokens.astype(np.int32),
        "logits": out.logits.float().numpy(),
        "n_layers": np.array(n_layers),
        "transformers_version": np.array(transformers.__version__),
        "torch_version": np.array(torch.__version__),
        "dtype": np.array(args.dtype),
        "model": np.array(args.model),
        "text_config": np.array(json.dumps(text_cfg.to_dict(), default=str)),
    }
    for i, h in enumerate(hidden):
        payload[f"hidden_{i}"] = h
    payload["embeddings"] = out.hidden_states[0].float().numpy()
    for name, arr in probes.items():
        payload[f"probe_{name}"] = arr

    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}")
    print(
        f"  model={args.model} dtype={args.dtype} transformers={transformers.__version__}"
    )
    print(
        f"  input_ids {tokens.shape}  logits {tuple(out.logits.shape)}  layers {len(hidden)}"
    )


if __name__ == "__main__":
    main()
