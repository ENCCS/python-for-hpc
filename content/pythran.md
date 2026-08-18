# Compiling numerical Python with Pythran

:::{questions}
- When is a numerical Python function a good candidate for Pythran?
- How do we compile a Python module into a native extension?
- How do we check that optimization preserves numerical correctness?
- How can SIMD and OpenMP improve a compiled numerical kernel?
:::

:::{objectives}
- Add a Pythran export specification to a numerical Python function.
- Compile and import a Pythran extension module.
- Compare Python-loop, NumPy, and Pythran implementations fairly.
- Use native CPU, SIMD, and OpenMP options with appropriate checks.
- Recognize important limitations and choose between Pythran, Numba, and Cython.
:::

:::{prereq}
- Comfortable reading Python functions and NumPy array expressions
- Able to create a virtual environment and run commands in a terminal
- No prior C++, compiler, or OpenMP knowledge is assumed
:::

This core episode is designed for approximately **45–60 minutes**, excluding
optional deep dives.

## Why Pythran?

Python loops are convenient, but every iteration normally passes through the
Python interpreter. NumPy avoids much of that overhead by running operations in
compiled libraries. Sometimes, however, an algorithm is most naturally written
as custom loops and is not already available as one optimized library call.

[Pythran](https://pythran.readthedocs.io/) is an ahead-of-time compiler for a
numerical subset of Python. It translates an annotated Python module into a
native extension that is imported like an ordinary Python module. Pythran can
also expose compiler vectorization and OpenMP parallelism.

Pythran is a promising choice when:

- Profiling identifies a numerical Python function as a bottleneck.
- The function primarily uses loops, scalar arithmetic, and supported NumPy
  operations.
- Inputs have predictable numerical types.
- Ahead-of-time compilation fits the project's build and deployment workflow.

It is not a compiler for arbitrary Python applications. Dynamic objects,
unsupported packages, classes, and interpreter-heavy code should remain in
Python or use another approach.

## First look: Pythran and Cython

Both tools can compile the same numerical loop, but they expose types in
different places. This comparison is about the source learners maintain, not a
performance claim.

::::{container} code-comparison
:::{container} code-comparison-panel
**Pythran: types at the exported boundary**

```python
# pythran export squared_distance(float64[], float64[])
def squared_distance(a, b):
    total = 0.0
    for index in range(a.shape[0]):
        difference = a[index] - b[index]
        total += difference * difference
    return total
```

The implementation remains an ordinary Python function.
:::
:::{container} code-comparison-panel
**Cython: types within the implementation**

```cython
def squared_distance(double[:] a, double[:] b):
    cdef Py_ssize_t index
    cdef double difference
    cdef double total = 0.0

    for index in range(a.shape[0]):
        difference = a[index] - b[index]
        total += difference * difference
    return total
```

The conventional `.pyx` version provides more explicit low-level control.
:::
::::

Pythran is attractive when an existing numerical function fits its supported
Python subset. Cython is often a better fit when a project needs C/C++
interoperability, extension types, or detailed control. We will use the larger
diffusion example to test correctness and measure performance responsibly.

## The teaching problem: two-dimensional diffusion

We will evolve a temperature-like field using a five-point finite-difference
stencil:

```{math}
u^{n+1}_{i,j} = u^n_{i,j} + \alpha\left(
u^n_{i-1,j} + u^n_{i+1,j} + u^n_{i,j-1} + u^n_{i,j+1}
- 4u^n_{i,j}\right).
```

Here, {math}`\alpha=D\Delta t/\Delta x^2` is a dimensionless diffusion
coefficient. For this explicit two-dimensional scheme, we use
{math}`0 \leq \alpha \leq 1/4`. Values above this stability limit can make the
numerical solution grow rather than diffuse.

The example uses fixed boundary values: the outermost rows and columns never
change. Each implementation returns a new array and leaves its input untouched.

The files are under `content/examples/pythran/`:

| File | Purpose |
|---|---|
| {download}`diffusion_python.py <examples/pythran/diffusion_python.py>` | Readable Python-loop baseline and initial field |
| {download}`diffusion_reference.py <examples/pythran/diffusion_reference.py>` | Vectorized NumPy reference |
| {download}`diffusion_pythran.py <examples/pythran/diffusion_pythran.py>` | Compilable Pythran target |
| {download}`test_diffusion.py <examples/pythran/test_diffusion.py>` | Shared correctness tests |
| {download}`benchmark_diffusion.py <examples/pythran/benchmark_diffusion.py>` | Correctness-first benchmark driver |

Download all five files into the same directory. With NumPy, Pytest, and
Pythran installed in the active environment, verify and compile them with:

```bash
cd path/to/downloaded-files
python -m pytest -q test_diffusion.py
pythran diffusion_pythran.py
python benchmark_diffusion.py
```

## Start with a correct baseline

The loop implementation keeps two buffers. Every new value is computed only
from the previous time step, then the buffers exchange roles.

```{literalinclude} examples/pythran/diffusion_python.py
:language: python
:pyobject: evolve_python
```

The NumPy implementation expresses the same update with slices:

```{literalinclude} examples/pythran/diffusion_reference.py
:language: python
:pyobject: evolve_numpy
```

The NumPy version is our trusted reference, not an automatically inferior
competitor. For some array expressions, NumPy may already be the clearest and
fastest practical implementation.

:::{exercise} Trace one update
Consider a grid whose only non-zero value is `field[2, 2] = 1.0`. With
`alpha = 0.2`, calculate the new values at the center and its four direct
neighbors after one step. Assume all five cells are interior cells.
:::

:::{solution}
The center receives four zero-valued neighbors:

```text
1.0 + 0.2 * (0 + 0 + 0 + 0 - 4*1.0) = 0.2
```

Each direct neighbor receives the old center value and three zeros:

```text
0.0 + 0.2 * (1.0 + 0 + 0 + 0 - 4*0.0) = 0.2
```

The total value remains `1.0`: five cells now contain `0.2`.
:::

### Test before optimizing

Run the shared tests from the repository root:

```bash
.venv/bin/python -m pytest -q content/examples/pythran/test_diffusion.py
```

The tests compare the loop, NumPy, and still-uncompiled Pythran source for
multiple grid shapes and time-step counts. They also verify fixed boundaries,
input immutability, deterministic initialization, and the `float64` dtype.

This order matters: a fast result is useless if it solves a different problem.

## Add a Pythran specification

Pythran needs to know which functions become part of the native module and
which input types to compile. The following comment is an export specification:

```python
# pythran export evolve_pythran(float64[][], float64, int)
```

It requests a version of `evolve_pythran` accepting:

1. A two-dimensional `float64` array
2. A `float64` scalar for `alpha`
3. An integer number of time steps

The complete target remains valid Python:

```{literalinclude} examples/pythran/diffusion_pythran.py
:language: python
```

Specifications are part of the compiled interface. Passing an array with the
wrong dtype or dimensionality will not silently select the requested version.
For applications that accept several dtypes, add another explicit overload
rather than assuming one compiled function is type-generic.

Pythran also supports specifications in a separate `.pythran` file and can
constrain array memory layout with `order(C)` or `order(F)`. Start with the
smallest interface that the application actually needs.

## Compile and import

Change to the example directory, then build the basic native module:

```bash
cd content/examples/pythran
../../../.venv/bin/pythran diffusion_pythran.py
```

The output name includes a platform-specific extension, for example:

```text
diffusion_pythran.cpython-313-x86_64-linux-gnu.so
```

Python prefers the native extension when it is next to
`diffusion_pythran.py`, so the import does not change:

```python
from diffusion_pythran import evolve_pythran
```

Pythran modules expose `__pythran__`, which can confirm that the native module
was imported:

```python
import diffusion_pythran

assert hasattr(diffusion_pythran, "__pythran__")
print(diffusion_pythran.__pythran__)
```

:::{warning}
Do not leave an old extension module beside edited source code. Python can keep
importing the stale binary even though `diffusion_pythran.py` has changed.
Recompile after edits or remove the generated extension while developing.
:::

## Verify, then benchmark

The benchmark driver computes the NumPy result first and checks every timed
implementation against it with `numpy.testing.assert_allclose`.

From `content/examples/pythran/`, run:

```bash
python benchmark_diffusion.py --size 128 --steps 20 --repeats 5 \
  --include-python
```

The script reports:

- Grid shape and dtype
- Number of time steps
- Diffusion coefficient
- Number of repetitions
- `OMP_NUM_THREADS`, when set
- Median and minimum execution time

Compilation, imports, and input generation occur outside the timed region. Each
implementation receives the same input and performs its own internal buffer
setup.

Do not copy a speedup from someone else's machine into a conclusion. Compiler,
CPU, array size, thread count, memory bandwidth, and background activity all
affect the result. Report the configuration with the measurement.

:::{admonition} Example measurement—not an expected result
:class: note

On an Apple M2 MacBook Air with 8 CPU cores and 16 GB memory, Python 3.13.8,
NumPy 2.5.2, Pythran 0.18.1, and Apple Clang 21.0.0, a 128×128 `float64` grid
evolved for 20 steps produced these medians over 11 repetitions:

| Implementation | Median | Relative to Python loops |
|---|---:|---:|
| Python loops | 194 ms | 1.0× |
| NumPy | 0.927 ms | 209× |
| Basic Pythran | 0.110 ms | 1,760× |

Compilation was excluded, and each result was checked against NumPy before it
was timed. The large ratios mostly show how expensive nested interpreted loops
are; they are not guaranteed Pythran speedups for other programs or machines.
:::

:::{exercise} Find the crossover
Run the benchmark for grid sizes 32, 64, 128, 256, and 512. At what size does
the compiled version become consistently faster than the alternatives on your
machine? Explain why tiny kernels can be dominated by call and allocation
overhead.
:::

:::{solution}
There is no universal crossover size. A valid answer includes the machine,
compiler, dtype, step count, repetitions, and observed measurements. The main
conclusion should be that overhead matters more for small problems, while the
compiled loop becomes easier to assess when each call performs enough work.
:::

## Native CPU and SIMD compilation

Pythran forwards common compiler options to its C++ compiler. A more specialized
build can be produced with:

```bash
pythran -DUSE_XSIMD -march=native diffusion_pythran.py
```

- `-DUSE_XSIMD` enables Pythran's xsimd-backed vectorization paths.
- `-march=native` lets the compiler use instructions available on the current
  CPU.

`-march=native` reduces binary portability: a module built on one CPU may not
run on an older or different CPU. Build portable teaching material without it,
then introduce it as a local optimization that must be measured.

The stencil performs relatively little arithmetic for each group of memory
loads and stores. It can therefore become limited by memory bandwidth. SIMD
may help without producing the ideal multiplication of performance suggested
by the vector width.

## Parallelize the independent rows with OpenMP

Within one time step, every interior output cell reads only from `current` and
writes to a distinct element of `following`. Rows can therefore be processed
in parallel:

```python
# omp parallel for shared(current, following)
for row in range(1, rows - 1):
    ...
```

Build with an OpenMP-capable C++ compiler:

```bash
pythran -DUSE_XSIMD -fopenmp -march=native diffusion_pythran.py
```

The OpenMP directive has no effect unless OpenMP is enabled during compilation.
Control the runtime thread count externally:

```bash
OMP_NUM_THREADS=1 python benchmark_diffusion.py --size 1024 --steps 100
OMP_NUM_THREADS=2 python benchmark_diffusion.py --size 1024 --steps 100
OMP_NUM_THREADS=4 python benchmark_diffusion.py --size 1024 --steps 100
```

:::{note}
Apple Clang does not provide OpenMP in its default configuration. On macOS, use
an OpenMP-capable compiler such as Homebrew GCC and configure `CC` and `CXX`, or
perform the OpenMP exercise on a Linux/HPC system. Compiler names vary by
installation; do not hard-code a version in shared lesson commands.
:::

:::{admonition} Example OpenMP scaling—not an expected result
:class: note

On the same machine, an OpenMP build made with Homebrew GCC 15.2.0 processed a
1024×1024 `float64` grid for 100 steps. Medians over 9 repetitions were:

| OpenMP threads | Pythran median | Speedup over 1 thread |
|---:|---:|---:|
| 1 | 156.6 ms | 1.00× |
| 2 | 82.0 ms | 1.91× |
| 4 | 47.0 ms | 3.33× |
| 8 | 41.9 ms | 3.74× |

The flattening between four and eight threads is part of the result, not a
failure: parallel overhead, heterogeneous cores, and memory bandwidth limit
scaling.
:::

:::{exercise} Measure scaling
Record Pythran execution time for 1, 2, 4, and—if available—8 threads. Compute
speedup relative to one thread. Does doubling the threads halve the time?
Identify at least two reasons why scaling may flatten.
:::

:::{solution}
The numerical values depend on the system. Common reasons for sub-linear
scaling include:

- Memory-bandwidth saturation
- OpenMP scheduling and synchronization overhead
- A problem that is too small for the thread count
- Other processes competing for cores or memory
- Simultaneous multithreading sharing execution resources

Correctness should be checked separately for every compiled configuration
before its timings are accepted.
:::

### Avoid oversubscription

OpenMP is not the only component that may create threads. NumPy libraries,
process pools, MPI ranks, and job schedulers can all affect resource use. Four
MPI ranks that each start eight OpenMP threads request 32 threads. If only eight
cores were allocated, the result is oversubscription and often worse
performance.

Record and control thread counts as part of a reproducible benchmark.

## Configure repeated builds with `.pythranrc`

Pythran reads user configuration from `$XDG_CONFIG_HOME/.pythranrc` (normally
`~/.config/.pythranrc` when `XDG_CONFIG_HOME` is set according to local
practice) or the location described by the installed Pythran version. A minimal
compiler configuration can look like:

```ini
[compiler]
CC=gcc
CXX=g++
```

Compiler executable names and library paths are system-specific. Prefer command
line flags in a short exercise because they remain visible and reproducible.
Use configuration files for stable site or developer settings, and document
them alongside benchmark results.

Pythran also honors `CC`, `CXX`, `CXXFLAGS`, and `LDFLAGS`. Environment
variables take precedence over configuration values.

## The GIL and thread safety

During execution of a generated native function, Pythran releases Python's
Global Interpreter Lock (GIL). That permits other Python threads to execute
while the numerical kernel runs. This is separate from OpenMP: releasing the
GIL enables concurrency at the Python level, whereas an OpenMP directive asks
the compiled kernel itself to use multiple native threads.

There is an important advanced caveat. Pythran's documentation notes that
non-OpenMP builds do not use thread-safe reference counting by default. Projects
that call generated code concurrently from Python threads should review the
`THREAD_SAFE_REF_COUNT` build option and test their actual ownership patterns.

## Supported features and limitations

Consult Pythran's
[supported modules and functions](https://pythran.readthedocs.io/en/latest/SUPPORT.html)
before choosing a kernel. The supported subset evolves, so treat that page as
the authority rather than relying on a fixed list in this lesson.

:::{warning}
Pythran compiles a numerical subset of Python, not arbitrary Python programs.
Check compatibility before deciding to compile a function.
:::

:::{admonition} Common patterns that need particular care
:class: warning dropdown

- Dynamic code whose values change type during execution
- Arbitrary Python objects, classes, and dynamic dispatch
- Calls into unsupported third-party packages
- Heterogeneous containers, except supported tuple patterns
- NumPy functions or signatures absent from Pythran's support matrix
- Calculations relying on Python's arbitrary-size integer semantics
- Code relying on Python object identity or mutation behavior
:::

Native compilation also adds build, packaging, and platform-compatibility
costs.

A good workflow keeps orchestration and dynamic behavior in Python while
compiling a small, well-tested numerical boundary.

## Choosing a Python optimization approach

| Tool | Compilation model | Typical code change | Good fit |
|---|---|---:|---|
| NumPy | Precompiled operations | Low | Algorithms expressible as efficient array/library calls |
| Pythran | Ahead of time | Low–moderate | Typed numerical functions and custom loops |
| Numba | Usually just in time | Low | Interactive numerical loops supported by Numba |
| Cython | Ahead of time | Moderate–high | Fine control, C/C++ interaction, and broader extension work |

No row is universally fastest. Consider deployment requirements, supported
syntax, team familiarity, startup or build cost, hardware, and measured
performance on the real workload.

## Summary

The reliable optimization loop is:

```text
correct baseline -> test -> profile -> compile -> test -> benchmark
                 -> parallelize -> test -> benchmark
```

Pythran is most useful at a deliberate numerical boundary. Export only the
functions and types that are needed, keep the source valid Python, and preserve
one reference implementation for correctness testing.

:::{keypoints}
- Pythran ahead-of-time compiles a numerical subset of Python into a native
  extension module.
- Export specifications define the compiled interface and accepted types.
- Correctness checks must precede every performance comparison.
- SIMD, native CPU flags, and OpenMP are optional optimizations that must be
  measured and may reduce portability.
- Generated functions release the GIL, but concurrent use still requires
  attention to Pythran's thread-safety guidance.
- Pythran complements NumPy, Numba, and Cython; it does not replace them in
  every workload.
:::

## Going further

The following topics are enrichment and are not required for the core lesson.

### Inspect Pythran's generated code

Pythran can expose intermediate forms for advanced investigation:

```bash
pythran -P diffusion_pythran.py
pythran -E diffusion_pythran.py
pythran -e diffusion_pythran.py
```

These options stop the compilation pipeline at different points:

- `-P` runs only the high-level optimizer, exposing optimized Python-like
  output.
- `-E` translates to C++ without compiling and includes the Python-extension
  glue.
- `-e` is similar to `-E`, but omits the Python-extension glue so the numerical
  implementation is easier to isolate.

The generated output can help discuss constant folding, dead-code elimination,
loop transformations, and expression-template fusion. It is compiler
output—not code learners should maintain manually.

### Use Transonic as an integration layer

[Transonic](https://transonic.readthedocs.io/) provides a higher-level way to
express compiled numerical Python code and can use Pythran as a backend. It is
useful when a project wants one Python-facing optimization interface or wants
to defer some backend-specific details.

Learn direct Pythran first in this episode: it makes the compiled interface,
supported subset, and build flags visible. Treat Transonic as an integration
option rather than a requirement for understanding Pythran.

## See also

- [Pythran documentation](https://pythran.readthedocs.io/)
- [Pythran user manual](https://pythran.readthedocs.io/en/latest/MANUAL.html)
- [Supported modules and functions](https://pythran.readthedocs.io/en/latest/SUPPORT.html)
- [Pythran optimization stories](https://serge-sans-paille.github.io/pythran-stories/being-more-than-a-translator.html)
- [Transonic documentation](https://transonic.readthedocs.io/)
