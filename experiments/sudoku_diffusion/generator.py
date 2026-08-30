"""Synthetic Sudoku data generation.

The diffusion model is trained on *solved* boards: the forward (noising) process
takes a complete grid and replaces a fraction of its cells with ``<mask>``; a
Sudoku *puzzle* is simply one such partially-masked board (see ``README.md``).
So the only thing the training loop needs is an endless stream of valid, fully
filled 9x9 grids -- which is exactly what this module produces.

Generation is *constructive*, not search-based: we lay down one canonical valid
solution from the well-known modular pattern

    value(r, c) = nums[(box * (r % box) + r // box + c) % side]

and then apply the validity-preserving symmetries of Sudoku -- relabelling the
digits, permuting rows within a band, permuting columns within a stack, and
permuting the bands/stacks themselves.  Every such transform maps a solution to
another solution, so the result is valid *by construction* and we never need a
solver.  The symmetry group is large enough (well over 10^20 distinct grids) to
treat the stream as effectively non-repeating for training.

For evaluation we also expose :func:`make_puzzle`, which knocks holes in a solved
board to produce a puzzle to solve.  (Hole removal does *not* guarantee a unique
solution; that is fine here because we always score against the known original
solution rather than asserting uniqueness.)
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from experiments.sudoku_diffusion.tokenizer import SudokuTokenizer


def random_solution(rng: np.random.Generator, box: int = 3) -> np.ndarray:
    """Return a random, valid, fully-solved ``side x side`` board (``side=box^2``).

    Digit values are ``1..side`` (no zeros -- a solution has no empty cells).
    Validity is guaranteed by construction, so no solver/backtracking is run.
    """
    side = box * box

    def shuffled(seq) -> list[int]:
        s = list(seq)
        rng.shuffle(s)
        return s

    bands = range(box)
    # Row indices grouped by band, columns by stack -- shuffling within and
    # across groups keeps every row/column/box a permutation of 1..side.
    rows = [g * box + r for g in shuffled(bands) for r in shuffled(bands)]
    cols = [g * box + c for g in shuffled(bands) for c in shuffled(bands)]
    nums = shuffled(range(1, side + 1))

    def pattern(r: int, c: int) -> int:
        return (box * (r % box) + r // box + c) % side

    board = np.array(
        [[nums[pattern(r, c)] for c in cols] for r in rows], dtype=np.int32
    )
    return board


def make_puzzle(
    solution: np.ndarray, rng: np.random.Generator, num_clues: int = 30
) -> np.ndarray:
    """Knock holes in ``solution`` to produce a puzzle (emptied cells become 0).

    Args:
      solution: a solved board as returned by :func:`random_solution`.
      rng: numpy random generator.
      num_clues: how many cells to keep filled; the rest are set to 0 (empty).
    """
    puzzle = solution.copy().reshape(-1)
    n = puzzle.size
    num_clues = int(np.clip(num_clues, 0, n))
    holes = rng.permutation(n)[: n - num_clues]
    puzzle[holes] = 0
    return puzzle.reshape(solution.shape)


def is_valid_solution(board: np.ndarray, box: int = 3) -> bool:
    """True iff ``board`` is a fully-filled, rule-respecting Sudoku solution."""
    side = box * box
    board = np.asarray(board)
    if board.shape != (side, side):
        return False
    target = set(range(1, side + 1))
    # Every row and column is a permutation of 1..side.
    if any(set(board[i]) != target for i in range(side)):
        return False
    if any(set(board[:, j]) != target for j in range(side)):
        return False
    # Every box is a permutation of 1..side.
    for br in range(0, side, box):
        for bc in range(0, side, box):
            if set(board[br : br + box, bc : bc + box].reshape(-1)) != target:
                return False
    return True


def solved_board_batches(
    batch_size: int,
    rng: np.random.Generator,
    *,
    tokenizer: SudokuTokenizer | None = None,
    box: int = 3,
) -> Iterator[np.ndarray]:
    """Yield an endless stream of token-id batches ``[batch_size, side*side]``.

    Each row is a flattened solved board.  Digit values ``1..9`` coincide with
    their token ids under :class:`SudokuTokenizer` (id ``d`` == digit ``d``), so
    the boards are already valid model input; the ``tokenizer`` argument is
    accepted only to keep the id mapping in one place and is otherwise a no-op.
    """
    tok = tokenizer or SudokuTokenizer(grid_size=box * box)
    while True:
        boards = [tok.encode_grid(random_solution(rng, box)) for _ in range(batch_size)]
        yield np.stack(boards, axis=0).astype(np.int32)
