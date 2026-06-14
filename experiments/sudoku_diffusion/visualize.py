"""Terminal visualisation of the uniform-state Sudoku denoising process.

:func:`visualize_solving` animates :func:`sampler.solve_iter` for a single
puzzle, redrawing the board after each denoising step with a controllable delay.
Cells are colour-coded against the known solution:

* **blue background**  -- a given clue (fixed throughout, never guessed);
* **green background** -- a blank the model currently has *right*;
* **red background**   -- a blank the model currently has *wrong*.

Because uniform-state diffusion renoises non-committed cells, a cell can flip
red->green->red across steps -- watching that settle is the whole point.

The animation uses ANSI escape codes (background colours + cursor repositioning)
and degrades to plain numbers when ``color=False`` or output is not a TTY.

Example::

    from experiments.sudoku_diffusion import visualize_solving
    visualize_solving(model, puzzle, solution, jax.random.key(0), delay=0.15)
"""

from __future__ import annotations

import sys
import time

import numpy as np

from experiments.sudoku_diffusion.sampler import SamplerConfig, solve_iter

# ANSI escape codes.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_FG = "\033[97m"  # bright white foreground for contrast on coloured cells
_BG = {"blue": "\033[44m", "green": "\033[42m", "red": "\033[41m"}
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


def _cell(value: int, color: str | None) -> str:
    """Render one cell as a 3-wide coloured field (or plain when no colour)."""
    text = f" {value} "
    if color is None:
        return text
    return f"{_BG[color]}{_BOLD}{_FG}{text}{_RESET}"


def render_board(
    board: np.ndarray,  # [L] or [N, N] int
    *,
    given_mask: np.ndarray,  # [L] or [N, N] bool, True = clue
    solution: np.ndarray | None = None,  # [L] or [N, N] int, for green/red
    color: bool = True,
) -> str:
    """Return a colour-coded multi-line string for a single board.

    Givens are blue; non-given cells are green/red by agreement with
    ``solution`` (if given).  3x3 boxes are separated for readability.
    """
    n = int(round(board.size**0.5))
    board = np.asarray(board).reshape(n, n)
    given_mask = np.asarray(given_mask).reshape(n, n)
    sol = None if solution is None else np.asarray(solution).reshape(n, n)
    box = int(round(n**0.5))

    lines: list[str] = []
    for r in range(n):
        cells: list[str] = []
        for c in range(n):
            if not color:
                clr = None
            elif given_mask[r, c]:
                clr = "blue"
            elif sol is None:
                clr = None
            else:
                clr = "green" if board[r, c] == sol[r, c] else "red"
            cells.append(_cell(int(board[r, c]), clr))
            if box and (c + 1) % box == 0 and c + 1 < n:
                cells.append("|")
        lines.append("".join(cells))
        if box and (r + 1) % box == 0 and r + 1 < n:
            # Separator row spanning the 3-wide cells plus the column bars.
            width = n * 3 + (n // box - 1)
            lines.append("-" * width)
    return "\n".join(lines)


def visualize_solving(
    model,
    puzzle: np.ndarray,  # [L] int; 0 = blank, 1..9 = given
    solution: np.ndarray,  # [L] int, ground truth
    key,
    *,
    config: SamplerConfig | None = None,
    delay: float = 0.2,
    step_stride: int = 1,
    color: bool = True,
    stream=sys.stdout,
) -> np.ndarray:
    """Animate the denoising of ``puzzle`` and return the final board.

    Args:
      model: a (trained) ``SudokuDiffusion``.
      puzzle: ``[L]`` int with 0 for blanks, 1..9 for clues.
      solution: ``[L]`` ground-truth board, used to colour guesses green/red.
      key: PRNG key for the sampler.
      config: sampler configuration (defaults to ``SamplerConfig()``).
      delay: seconds to pause between drawn frames.
      step_stride: draw every ``step_stride``-th step (1 = every step).
      color: emit ANSI colours (auto-disabled if ``stream`` is not a TTY).
      stream: output stream (default ``sys.stdout``).

    Returns:
      The final ``[L]`` solved board.
    """
    puzzle = np.asarray(puzzle).reshape(-1)
    solution = np.asarray(solution).reshape(-1)
    given_mask = puzzle != 0
    color = color and getattr(stream, "isatty", lambda: False)()

    n = int(round(puzzle.size**0.5))
    box = int(round(n**0.5))
    # Lines per frame: 1 header + n board rows + (n/box - 1) box separators.
    n_lines = 1 + n + (n // box - 1)

    def write(text: str) -> None:
        stream.write(text)
        stream.flush()

    if color:
        write(_HIDE_CURSOR)
    last_board = puzzle
    try:
        first = True
        for step, board in enumerate(solve_iter(model, puzzle[None], key, config)):
            last_board = board[0]
            total = (config or SamplerConfig()).num_steps
            is_last = step == total
            if not is_last and step % step_stride != 0:
                continue

            blanks = ~given_mask
            n_correct = int(((last_board == solution) & blanks).sum())
            n_blank = int(blanks.sum())
            grid = render_board(
                last_board,
                given_mask=given_mask,
                solution=solution,
                color=color,
            )
            header = f"step {step:3d}/{total}   correct blanks: {n_correct}/{n_blank}"
            frame = f"{header}\n{grid}\n"

            if color and not first:
                write(f"\033[{n_lines}A")  # move cursor up to overwrite frame
            write(frame)
            first = False
            time.sleep(delay)
    finally:
        if color:
            write(_SHOW_CURSOR)

    return last_board
