"""Pythran target for the two-dimensional diffusion example."""

import numpy as np


# pythran export evolve_pythran(float64[][], float64, int)
def diffusion_step(current, following, alpha):
    """Write one stencil update to ``following``."""
    rows, columns = current.shape

    # omp parallel for shared(current, following)
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


def evolve_pythran(field, alpha, steps):
    """Evolve a float64 field using loops that Pythran can compile."""
    current = field.copy()
    following = field.copy()

    for _ in range(steps):
        diffusion_step(current, following, alpha)
        temporary = current
        current = following
        following = temporary

    return current
