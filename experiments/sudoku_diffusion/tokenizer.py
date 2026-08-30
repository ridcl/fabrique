"""Custom tokenizer for the Sudoku diffusion model.

DiffusionGemma ships with the full Gemma SentencePiece vocabulary (256k tokens).
For Sudoku we don't need any of that: the only "language" we ever read or write
is the ten decimal digits.  A board is serialized row-by-row, cells within a row
separated by a single space and rows separated by a newline, e.g.::

    5 3 0 0 7 0 0 0 0
    6 0 0 1 9 5 0 0 0
    0 9 8 0 0 0 0 6 0
    ...

The digit ``0`` denotes an *empty* (not-yet-known) cell in the human-readable
text.  Spaces and newlines are pure formatting: they are dropped on
:meth:`encode` and re-inserted on :meth:`decode`, so the model only ever sees a
flat sequence of digit tokens (81 of them for a standard 9x9 board).

Beyond the ten digits the vocabulary carries two special tokens:

* ``<pad>`` -- padding, used to fill batches to a common length.
* ``<mask>`` -- a reserved absorbing/blank token.  Note the DiffusionGemma-style
  training in ``trainer.py`` is *uniform-state*, not masked, diffusion: it
  corrupts cells into random **digits**, never into ``<mask>``.  This token is
  therefore unused by the diffusion process itself and kept only as a
  convenience (e.g. to render not-yet-known cells); ``decode`` shows it as
  ``?``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DIGITS = "0123456789"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"


class SudokuTokenizer:
    """Maps Sudoku digit text to/from token ids.

    The vocabulary layout is fixed and tiny::

        id 0..9  -> '0'..'9'
        id 10    -> <pad>
        id 11    -> <mask>

    Args:
        grid_size: side length of the (square) board.  Used only by the grid
            helpers and by :meth:`decode` to know where to insert newlines.
    """

    def __init__(self, grid_size: int = 9):
        self.grid_size = grid_size
        self._id_to_token: list[str] = list(DIGITS) + [PAD_TOKEN, MASK_TOKEN]
        self._token_to_id: dict[str, int] = {
            tok: i for i, tok in enumerate(self._id_to_token)
        }

    # -- sizes / special ids -------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def pad_id(self) -> int:
        return self._token_to_id[PAD_TOKEN]

    @property
    def mask_id(self) -> int:
        return self._token_to_id[MASK_TOKEN]

    @property
    def digit_ids(self) -> list[int]:
        """Token ids of the ten digits, in order 0..9."""
        return [self._token_to_id[d] for d in DIGITS]

    # -- core encode / decode ------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Turn Sudoku text into a flat list of digit token ids.

        Every digit character is kept (in reading order); all other characters
        (spaces, newlines, dots, etc.) are treated as formatting and dropped.
        """
        return [self._token_to_id[ch] for ch in text if ch in self._token_to_id]

    def decode(self, ids: Sequence[int] | np.ndarray, as_grid: bool = True) -> str:
        """Turn token ids back into human-readable text.

        Args:
            ids: sequence of token ids.
            as_grid: if True, lay the digits out as a ``grid_size`` x
                ``grid_size`` board (space-separated cells, newline-separated
                rows).  Otherwise return a bare space-separated string.

        ``<pad>`` tokens are skipped; ``<mask>`` tokens are rendered as ``?`` so
        partially-denoised boards stay readable.
        """
        ids = [int(i) for i in ids]
        cells: list[str] = []
        for i in ids:
            tok = self._id_to_token[i]
            if tok == PAD_TOKEN:
                continue
            cells.append("?" if tok == MASK_TOKEN else tok)
        if not as_grid:
            return " ".join(cells)
        n = self.grid_size
        rows = [" ".join(cells[r : r + n]) for r in range(0, len(cells), n)]
        return "\n".join(rows)

    # -- numpy grid helpers --------------------------------------------------

    def encode_grid(self, grid: Iterable[Iterable[int]]) -> np.ndarray:
        """Flatten a 2D iterable of digit values into an int32 id array.

        Digit values are already in ``0..9`` so they double as token ids, but we
        route them through the vocabulary so the mapping stays in one place.
        """
        arr = np.asarray(grid, dtype=np.int32).reshape(-1)
        if arr.min() < 0 or arr.max() > 9:
            raise ValueError("grid values must be digits in 0..9")
        return arr

    def decode_grid(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Reshape a flat id sequence into a ``grid_size`` x ``grid_size`` board.

        Non-digit tokens (pad/mask) are mapped to ``0`` (empty).
        """
        flat = np.asarray([int(i) if int(i) < 10 else 0 for i in ids], dtype=np.int32)
        n = self.grid_size
        if flat.size != n * n:
            raise ValueError(f"expected {n * n} cells, got {flat.size}")
        return flat.reshape(n, n)
