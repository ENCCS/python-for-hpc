"""Benchmark the diffusion implementations after checking correctness."""

from __future__ import annotations

import argparse
import importlib
import os
import statistics
import time
from collections.abc import Callable

import numpy as np

from diffusion_python import evolve_python, initial_field
from diffusion_reference import evolve_numpy


Kernel = Callable[[np.ndarray, float, int], np.ndarray]


def measure(
    kernel: Kernel,
    field: np.ndarray,
    alpha: float,
    steps: int,
    repeats: int,
) -> tuple[float, float]:
    """Return median and minimum execution time, excluding setup."""
    kernel(field, alpha, 1)  # warm up imports, dispatch, and caches
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        kernel(field, alpha, steps)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def compiled_kernel() -> Kernel | None:
    """Return the compiled Pythran kernel, or ``None`` before compilation."""
    module = importlib.import_module("diffusion_pythran")
    if not hasattr(module, "__pythran__"):
        return None
    return module.evolve_pythran


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--include-python",
        action="store_true",
        help="also time the deliberately slow Python-loop baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    field = initial_field(args.size, args.size)
    expected = evolve_numpy(field, args.alpha, args.steps)

    kernels: list[tuple[str, Kernel]] = [("NumPy", evolve_numpy)]
    pythran_kernel = compiled_kernel()
    if pythran_kernel is not None:
        kernels.append(("Pythran", pythran_kernel))
    if args.include_python:
        kernels.insert(0, ("Python loops", evolve_python))

    print(
        f"grid={field.shape}, dtype={field.dtype}, steps={args.steps}, "
        f"alpha={args.alpha}, repeats={args.repeats}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}"
    )
    print(f"{'implementation':<16} {'median (s)':>12} {'minimum (s)':>12}")
    for name, kernel in kernels:
        result = kernel(field, args.alpha, args.steps)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        median, minimum = measure(
            kernel, field, args.alpha, args.steps, args.repeats
        )
        print(f"{name:<16} {median:>12.6f} {minimum:>12.6f}")

    if pythran_kernel is None:
        print("\nPythran module not compiled; run: pythran diffusion_pythran.py")


if __name__ == "__main__":
    main()
