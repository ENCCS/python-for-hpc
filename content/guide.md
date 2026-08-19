# Instructor's guide

## Pythran episode

### Why we teach this lesson

Pythran provides a compact example of a general HPC-development habit: identify
a measured numerical bottleneck, preserve a trusted implementation, compile a
small boundary, and test again before accepting performance results. The lesson
uses one diffusion kernel so that compilation, correctness, benchmarking, SIMD,
and OpenMP remain connected to the same scientific problem.

### Intended learning outcomes

After the 45–60 minute core, learners should be able to:

- Recognize a numerical function that is a plausible Pythran candidate.
- Write an export specification for a typed function boundary.
- Compile and confirm that Python imported a native Pythran extension.
- Check the compiled result against a NumPy reference before benchmarking.
- Explain when NumPy, Pythran, Numba, or Cython may be the better fit.

### Timing

The environment must be prepared before the timed lesson.

| Core segment | Suggested time |
|---|---:|
| Motivation and Pythran/Cython first look | 5 min |
| Diffusion problem, baseline, and trace exercise | 10 min |
| Export-specification exercise | 10 min |
| Compile, identify the native module, and retest | 15 min |
| Benchmark interpretation | 10 min |
| Limitations, tool choice, and summary | 10 min |

The optional extensions are independent. Allow 15–20 minutes for crossover
measurements, 10–15 minutes for native/SIMD flags, or 30–45 minutes for the
OpenMP exercise. Configuration, GIL behavior, compiler inspection, and
Transonic are reference or advanced-discussion material.

### Preparing exercises

Before the session:

1. Create the documented virtual environment and build the basic Pythran
   extension on the actual teaching system.
2. Run the source and compiled correctness tests, then build the complete Sphinx
   site. Retain the output for troubleshooting.
3. Confirm that learners will start in the repository root with the environment
   activated.
4. If teaching OpenMP, test the selected compiler and thread settings on the
   allocated machine. Do not assume the default macOS compiler supports OpenMP.
5. Decide in advance which optional extension, if any, fits the schedule.

For a machine without a working compiler, learners can still write the export
specification, run the Python-source tests, inspect instructor-provided build
output, and discuss the benchmark. Treat that as a fallback, not as evidence
that their own native module compiled.

### Other practical aspects

- Pythran compilation may take long enough that pairs progress at different
  speeds. Have a discussion prompt ready while compilation runs.
- Ask learners to show the `__pythran__` metadata and passing tests, not merely
  the presence of a file ending in `.so`.
- Keep the core benchmark small. Large crossover and scaling sweeps belong in
  extensions.
- Pair or group learners for measurements: hardware-dependent results become a
  useful comparison rather than a race for the largest speedup.

### Interesting questions you might get

**Why not use NumPy everywhere?** NumPy is often the best answer, but some
algorithms are clearer as custom loops or create costly intermediate arrays.

**Does Pythran compile every Python function?** No. It supports a numerical
subset and a documented collection of modules and functions.

**Why is my speedup different?** CPU, compiler, array size, memory bandwidth,
thread count, and background activity all affect the measurement.

**Does releasing the GIL make the function parallel?** No. GIL release permits
other Python threads to run; OpenMP is a separate request for parallel work
inside the compiled kernel.

### Typical pitfalls

- The virtual environment is not active, so `python` and `pythran` come from
  different installations.
- A stale native extension is imported after the `.py` source changes.
- The input dtype or number of dimensions does not match the export
  specification.
- Compilation time is included in an execution-time comparison.
- Learners compare functions that allocate or initialize different data.
- `alpha` exceeds the documented stability range.
- OpenMP is requested from a compiler that does not support it.
- Multiple libraries or MPI ranks create more threads than the allocation.
