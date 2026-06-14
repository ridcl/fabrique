"""A small, dense, DiffusionGemma-like model for solving Sudoku via discrete diffusion.

See ``model.py`` for the architecture and ``tokenizer.py`` for the custom
digit-only tokenizer.
"""

from experiments.sudoku_diffusion.checkpoint import load_model, save_model
from experiments.sudoku_diffusion.generator import (
    is_valid_solution,
    make_puzzle,
    random_solution,
    solved_board_batches,
)
from experiments.sudoku_diffusion.model import (
    ModelConfig,
    SudokuDiffusion,
    count_params,
)
from experiments.sudoku_diffusion.sampler import (
    SamplerConfig,
    solve,
    solve_accuracy,
    solve_iter,
)
from experiments.sudoku_diffusion.tokenizer import SudokuTokenizer
from experiments.sudoku_diffusion.trainer import (
    DiffusionConfig,
    SudokuDiffusionTrainer,
    corrupt,
    diffusion_loss,
)
from experiments.sudoku_diffusion.visualize import render_board, visualize_solving

__all__ = [
    "ModelConfig",
    "SudokuDiffusion",
    "SudokuTokenizer",
    "count_params",
    # generation
    "random_solution",
    "make_puzzle",
    "is_valid_solution",
    "solved_board_batches",
    # training
    "DiffusionConfig",
    "SudokuDiffusionTrainer",
    "corrupt",
    "diffusion_loss",
    # sampling
    "SamplerConfig",
    "solve",
    "solve_iter",
    "solve_accuracy",
    # visualization
    "render_board",
    "visualize_solving",
    # checkpointing
    "save_model",
    "load_model",
]
