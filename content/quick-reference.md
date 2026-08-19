# Quick Reference

## Pythran

### Workflow

```text
profile -> preserve a reference -> test -> specify -> compile
        -> test the native module -> benchmark
```

Use Pythran for a small numerical boundary with predictable types, supported
Python/NumPy operations, and enough work to justify ahead-of-time compilation.

### Export specifications

```python
# pythran export squared_distance(float64[], float64[])
# pythran export evolve_pythran(float64[][], float64, int)
```

- `float64[]`: one-dimensional NumPy array
- `float64[][]`: two-dimensional NumPy array
- `float64`: scalar double-precision value
- `int`: integer argument

Add explicit overloads when an application accepts more than one dtype or
layout. Do not assume the compiled interface is type-generic.

### Basic commands

Run these from the Pythran example directory with the lesson environment
activated:

```bash
# Test the Python source first
python -m pytest -q test_diffusion.py

# Build the native extension
pythran diffusion_pythran.py

# Confirm that the native module was imported
python -c "import diffusion_pythran; print(diffusion_pythran.__pythran__)"

# Test again with the extension beside the source
python -m pytest -q test_diffusion.py

# Benchmark after the correctness check
python benchmark_diffusion.py --size 128 --steps 20 --repeats 5
```

### Optional compiler modes

```bash
# Native CPU and xsimd-backed vectorization
pythran -DUSE_XSIMD -march=native diffusion_pythran.py

# Add OpenMP with a compatible compiler
pythran -DUSE_XSIMD -fopenmp -march=native diffusion_pythran.py

# Inspect intermediate output
pythran -P diffusion_pythran.py
pythran -E diffusion_pythran.py
pythran -e diffusion_pythran.py
```

`-march=native` reduces binary portability. An OpenMP directive has no effect
unless the module is compiled with OpenMP support.

### Runtime checks

- Compare every compiled result with the trusted reference.
- Keep imports, compilation, and input generation outside the timed region.
- Report problem size, dtype, compiler, software versions, thread count,
  repetitions, and summary statistic.
- Set `OMP_NUM_THREADS` explicitly for scaling experiments.
- Rebuild after editing source; an old native extension can shadow the `.py`
  file.

### References

- [Pythran documentation](https://pythran.readthedocs.io/)
- [Pythran user manual](https://pythran.readthedocs.io/en/latest/MANUAL.html)
- [Supported modules and functions](https://pythran.readthedocs.io/en/latest/SUPPORT.html)
