"""Save / load a trained ``SudokuDiffusion`` model.

A checkpoint is a directory containing:

* ``config.json`` -- the :class:`ModelConfig` (so the model can be rebuilt
  without the caller knowing its shape), plus a small ``_meta`` block.
* ``params/``     -- the model weights, written with orbax.

``save_model`` and ``load_model`` are a matched pair::

    save_model(model, "/data/sudoku_diffusion")
    model = load_model("/data/sudoku_diffusion")   # ready for sampler.solve(...)

Only the *model* is saved -- not the optimiser state -- because the intent is a
deployable denoiser, not a resumable training run.  (For resuming, checkpoint
``nnx.state(trainer.optimizer)`` alongside it, as ``flow_imagenet.py`` does.)
"""

from __future__ import annotations

import dataclasses
import json
import os

import jax.numpy as jnp
from flax import nnx

from experiments.sudoku_diffusion.model import ModelConfig, SudokuDiffusion

_CONFIG_FILE = "config.json"
_PARAMS_DIR = "params"


def _config_to_json(config: ModelConfig) -> dict:
    """Serialise a ModelConfig to JSON-safe types (dtype -> its name)."""
    d = dataclasses.asdict(config)
    d["param_dtype"] = jnp.dtype(config.param_dtype).name
    return d


def _config_from_json(d: dict) -> ModelConfig:
    d = dict(d)
    d["param_dtype"] = jnp.dtype(d["param_dtype"])
    return ModelConfig(**d)


def save_model(model: SudokuDiffusion, path: str) -> str:
    """Save ``model`` (config + weights) to directory ``path``.

    Returns the absolute checkpoint path.  An existing checkpoint at ``path`` is
    overwritten.
    """
    import orbax.checkpoint as ocp

    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, _CONFIG_FILE), "w") as f:
        json.dump(_config_to_json(model.config), f, indent=2)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.join(path, _PARAMS_DIR), nnx.state(model), force=True)
    checkpointer.wait_until_finished()
    return path


def load_model(path: str, *, rngs: nnx.Rngs | None = None) -> SudokuDiffusion:
    """Rebuild a :class:`SudokuDiffusion` from a checkpoint written by ``save_model``.

    Args:
      path: checkpoint directory.
      rngs: RNGs used only to allocate the (immediately overwritten) initial
        parameters; defaults to ``nnx.Rngs(0)``.
    """
    import orbax.checkpoint as ocp

    path = os.path.abspath(path)
    with open(os.path.join(path, _CONFIG_FILE)) as f:
        config = _config_from_json(json.load(f))

    # Build a concrete model to provide the restore target (shapes/dtypes), then
    # overwrite its parameters with the checkpointed ones.
    model = SudokuDiffusion(config, rngs=rngs or nnx.Rngs(0))
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(
        os.path.join(path, _PARAMS_DIR), target=nnx.state(model)
    )
    nnx.update(model, restored)
    return model
