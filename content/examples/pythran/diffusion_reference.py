"""Vectorized NumPy reference for the diffusion teaching example."""

from __future__ import annotations

import numpy as np


def evolve_numpy(field: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    """Evolve ``field`` with NumPy slicing while keeping its edges fixed."""
    current = field.copy()
    following = field.copy()

    for _ in range(steps):
        center = current[1:-1, 1:-1]
        following[1:-1, 1:-1] = center + alpha * (
            current[:-2, 1:-1]
            + current[2:, 1:-1]
            + current[1:-1, :-2]
            + current[1:-1, 2:]
            - 4.0 * center
        )
        current, following = following, current

    return current

