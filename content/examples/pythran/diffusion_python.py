"""Readable Python implementation of a two-dimensional diffusion update."""

from __future__ import annotations

import numpy as np


def initial_field(rows: int, columns: int) -> np.ndarray:
    """Return a deterministic field with a warm square at its center."""
    if rows < 5 or columns < 5:
        raise ValueError("the diffusion grid must be at least 5 by 5")

    field = np.zeros((rows, columns), dtype=np.float64)
    row_start, row_stop = rows // 3, 2 * rows // 3
    column_start, column_stop = columns // 3, 2 * columns // 3
    field[row_start:row_stop, column_start:column_stop] = 1.0
    return field


def evolve_python(field: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    """Evolve ``field`` using explicit Python loops and fixed boundaries.

    ``alpha`` is the dimensionless diffusion coefficient ``D*dt/dx**2``. For
    this two-dimensional five-point scheme, ``0 <= alpha <= 0.25`` is stable.
    The input is not modified; a new array is returned.
    """
    current = field.copy()
    following = field.copy()
    rows, columns = current.shape

    for _ in range(steps):
        for row in range(1, rows - 1):
            for column in range(1, columns - 1):
                center = current[row, column]
                following[row, column] = center + alpha * (
                    current[row - 1, column]
                    + current[row + 1, column]
                    + current[row, column - 1]
                    + current[row, column + 1]
                    - 4.0 * center
                )
        current, following = following, current

    return current

