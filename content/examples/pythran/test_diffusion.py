"""Correctness tests shared by the diffusion implementations."""

from __future__ import annotations

import numpy as np
import pytest

if __package__:
    from .diffusion_python import evolve_python, initial_field
    from .diffusion_pythran import evolve_pythran
    from .diffusion_reference import evolve_numpy
else:
    from diffusion_python import evolve_python, initial_field
    from diffusion_pythran import evolve_pythran
    from diffusion_reference import evolve_numpy


@pytest.mark.parametrize("shape", [(7, 9), (18, 14)])
@pytest.mark.parametrize("steps", [1, 6])
def test_implementations_match(shape: tuple[int, int], steps: int) -> None:
    field = initial_field(*shape)
    expected = evolve_numpy(field, 0.2, steps)

    np.testing.assert_allclose(
        evolve_python(field, 0.2, steps), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        evolve_pythran(field, 0.2, steps), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("kernel", [evolve_python, evolve_numpy, evolve_pythran])
def test_input_and_fixed_boundaries_are_unchanged(kernel) -> None:
    field = initial_field(11, 13)
    original = field.copy()
    result = kernel(field, 0.2, 4)

    np.testing.assert_array_equal(field, original)
    np.testing.assert_array_equal(result[0, :], original[0, :])
    np.testing.assert_array_equal(result[-1, :], original[-1, :])
    np.testing.assert_array_equal(result[:, 0], original[:, 0])
    np.testing.assert_array_equal(result[:, -1], original[:, -1])


def test_initial_field_is_deterministic_and_nontrivial() -> None:
    first = initial_field(12, 15)
    second = initial_field(12, 15)

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float64
    assert 0.0 < first.mean() < 1.0


@pytest.mark.parametrize("shape", [(4, 8), (8, 4)])
def test_initial_field_rejects_tiny_grids(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least 5 by 5"):
        initial_field(*shape)
