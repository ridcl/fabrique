import re

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
from multimethod import multimethod

###############################################################################
#                              dtype conversions                              #
###############################################################################


def pt2jax_dtype(pt_dtype: torch.dtype):
    if not isinstance(pt_dtype, torch.dtype):
        raise ValueError(
            f"The argument to to_jax_dtype() must be an instance of "
            + f"torch.dtype, but instead {type(pt_dtype)} was received"
        )
    # not using dicts because dtypes don't have stable hash
    if pt_dtype == torch.float32:
        return jnp.float32
    if pt_dtype == torch.float16:
        return jnp.float16
    if pt_dtype == torch.bfloat16:
        return jnp.bfloat16
    else:
        raise ValueError(f"Converting {pt_dtype} to a JAX type is not implemented")


def jax2pt_dtype(dtype: jnp.dtype):
    if not isinstance(dtype, jnp.dtype):
        raise ValueError(
            f"The argument to to_pytorch_dtype() must be an instance of "
            + f"jnp.dtype, but instead {type(dtype)} was received"
        )
    # not using dicts because dtypes don't have stable hash
    if dtype == jnp.float32:
        return torch.float32
    if dtype == jnp.float16:
        return torch.float16
    if dtype == jnp.bfloat16:
        return torch.bfloat16
    else:
        raise ValueError(f"Converting {dtype} to a PyTorch type is not implemented")


###############################################################################
#                              array conversions                              #
###############################################################################


def jax2pt(x: jax.Array):
    if x.dtype == jnp.bfloat16:
        # convert via fp32 because numpy doesn't support bf16
        x32 = x.astype(jnp.float32)
        return torch.from_numpy(np.asarray(x32).copy()).bfloat16()
    else:
        return torch.from_numpy(np.asarray(x).copy())


def pt2jax(pt_x: torch.Tensor):
    if pt_x.dtype == torch.bfloat16:
        # convert via fp32 because numpy doesn't support bf16
        pt_x32 = pt_x.to(torch.float32)
        return jnp.array(pt_x32.detach().numpy(), dtype=jnp.bfloat16)
    else:
        return jnp.array(pt_x.detach().numpy())


###############################################################################
#                filling PyTorch objects with JAX params                      #
###############################################################################


@multimethod
def fill_pytorch(dst: tnn.Linear, params):
    dst.weight.data = jax2pt(params["kernel"].T)
    assert "bias" not in params, "Bias is not supported yet"


@multimethod
def fill_pytorch(dst: tnn.Embedding, params):
    dst.weight.data = jax2pt(params["embedding"])


@multimethod
def fill_pytorch(dst: tnn.Parameter, val):
    dst.data = jax2pt(val)


@multimethod
def fill_pytorch(dst: tnn.Module, params):
    fields = params.keys()
    for field in fields:
        if m := re.match(r"^([a-zA-Z0-9_]+)_([0-9]+)$", field):
            # match lists, e.g. layer_0
            pt_field, index_ = m.groups()
            index = int(index_)
            new_dst = getattr(dst, pt_field)[index]
        else:
            new_dst = getattr(dst, field)
        new_params = params[field]
        fill_pytorch(new_dst, new_params)


###############################################################################
#                creating JAX params from PyTorch state_dict                  #
###############################################################################


def convert_to_nested(flat: dict):
    """
    Convert flat structure of PyTorch state dict to
    nested structure of JAX params.
    """

    def join_numbers(key_seq):
        # ["layer", "0", ...] -> ["layer_0", ...]
        assert len(key_seq) > 0 and not key_seq[0].isnumeric()
        out = []
        for key in key_seq:
            if key.isnumeric():
                out[-1] = out[-1] + "_" + key
            else:
                out.append(key)
        return out

    nested = {}
    for key, val in flat.items():
        key_seq = key.split(".")
        key_seq = join_numbers(key_seq)
        dct = nested
        for subkey in key_seq[:-1]:
            if subkey not in dct:
                dct[subkey] = {}
            dct = dct[subkey]
        dct[key_seq[-1]] = val
    return nested


def convert_linear(state: dict):
    assert (
        "bias" not in state
    ), "Convertion of bias in torch.nn.Linear is not supported yet"
    return {"kernel": pt2jax(state["weight"].T)}


def convert_attention(state: dict):
    assert len(state.keys()) == 4
    return {
        "wq": convert_linear(state["wq"]),
        "wk": convert_linear(state["wk"]),
        "wv": convert_linear(state["wv"]),
        "wo": convert_linear(state["wo"]),
    }


def convert_feed_forward(state: dict):
    assert len(state.keys()) == 3
    return {
        "w1": convert_linear(state["w1"]),
        "w2": convert_linear(state["w2"]),
        "w3": convert_linear(state["w3"]),
    }


def convert_attention_norm(state: dict):
    return {
        "weight": pt2jax(state["weight"]),
    }


def convert_ffn_norm(state: dict):
    return {
        "weight": pt2jax(state["weight"]),
    }


def convert_layer(state: dict):
    return {
        "attention": convert_attention(state["attention"]),
        "feed_forward": convert_feed_forward(state["feed_forward"]),
        "attention_norm": convert_attention_norm(state["attention_norm"]),
        "ffn_norm": convert_ffn_norm(state["ffn_norm"]),
    }


def convert_tok_embeddings(state: dict):
    return {
        "embedding": pt2jax(state["weight"]),
    }


def convert_norm(state: dict):
    return {
        "weight": pt2jax(state["weight"]),
    }


def convert_transformer(state: dict):
    layers = {
        k: convert_layer(state[k]) for k, v in state.items() if k.startswith("layers_")
    }
    return {
        "tok_embeddings": convert_tok_embeddings(state["tok_embeddings"]),
        "norm": convert_norm(state["norm"]),
        "output": convert_linear(state["output"]),
        **layers,
    }
