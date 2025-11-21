# Introduction to HPC

:::{questions}
- What is High-Performance Computing (HPC)?
- Why do we use HPC systems?
- How does parallel computing make programs faster?
:::

:::{objectives}
- Define what High-Performance Computing (HPC) is.
- Identify the main components of an HPC system.
- Describe the difference between serial and parallel computing.
- Run a simple command on a cluster using the terminal.
:::

High-Performance Computing (HPC) refers to using many computers working
together to solve complex problems faster than a single machine could.
HPC is widely used in fields such as climate science, molecular
simulation, astrophysics, and artificial intelligence.

This lesson introduces what HPC is, why it matters, and how researchers
use clusters to perform large-scale computations.

---

## What is HPC?

HPC systems, often called *supercomputers* or *clusters*, are made up of
many computers (called **nodes**) connected by a fast network. Each node
can have multiple cores which are **CPUs** (and sometimes **GPUs**) that 
run tasks in parallel.

### Typical HPC Components

| Component | Description |
|------------|--------------|
| **Login node** | Where you connect and submit jobs |
| **Compute nodes** | Machines where your program actually runs |
| **Scheduler** | Manages job submissions and allocates resources (e.g. SLURM) |
| **Storage** | Shared file system accessible to all nodes |

---
## Single core performance optimization

Pure-Python loops are slow because each iteration runs in the Python interpreter. NumPy pushes work into optimized native code (C/C++/BLAS), drastically reducing overhead. Below we compare a Python for loop with NumPy vectorized operations and discuss tips for fair, single-core measurements.

:::{exercise} Practice making Python faster on a single CPU.
Copy and paste this code 
```python
import os
# (Optional safety if you run this inside Python, must be set BEFORE importing numpy)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from time import perf_counter

def timeit(fn, *args, repeats=5, warmup=1, **kwargs):
    # warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    # timed runs
    tmin = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        fn(*args, **kwargs)
        dt = perf_counter() - t0
        tmin = min(tmin, dt)
    return tmin

# Problem size
N = 10_000_000  # 10 million elements

# Test data (contiguous, fixed dtype)
a = np.random.rand(N).astype(np.float64)
b = np.random.rand(N).astype(np.float64)

# --- 1) Pure-Python loop sum ---
def py_sum(x):
    s = 0.0
    for v in x:         # per-element Python overhead
        s += v
    return s

# --- 2) NumPy vectorized sum ---
def np_sum(x):
    return x.sum()      # dispatches to optimized C/BLAS

# --- 3) Elementwise add then sum (Python loop) ---
def py_add_sum(x, y):
    s = 0.0
    for i in range(len(x)):
        s += x[i] + y[i]
    return s

# --- 4) Elementwise add then sum (NumPy, no temporaries) ---
def np_add_sum_no_temp(x, y):
    # np.add.reduce avoids allocating x+y temporary
    return np.add.reduce([x, y])  # equivalent to sum stacks; see alt below

# Alternative that’s typically fastest and clearer:
def np_add_sum_fast(x, y):
    return (x + y).sum()  # may allocate a temporary; fast on many BLAS builds

# Time them
print("Timing on single core (best of 5 runs):")
t_py_sum   = timeit(py_sum, a)
t_np_sum   = timeit(np_sum, a)
t_py_add   = timeit(py_add_sum, a, b)
t_np_add   = timeit(np_add_sum_fast, a, b)

print(f"Python for-loop sum:          {t_py_sum:8.4f} s")
print(f"NumPy vectorized sum:         {t_np_sum:8.4f} s")
print(f"Python loop add+sum:          {t_py_add:8.4f} s")
print(f"NumPy vectorized add+sum:     {t_np_add:8.4f} s")
```
Execute it and following let us verify the effect on the following modifications:
1. Run the timing script with N = 1_000_000, 5_000_000, 20_000_000.
2. Try float32 vs float64.
3. Switch (a + b).sum() to np.add(a, b, out=a); a.sum() and compare.
:::


#### Practical tips for single-core speed
- Prefer vectorization: Use array ops (+, *, .sum(), .dot(), np.mean, np.linalg.*) rather than per-element Python loops.
- Control temporaries: Expressions like (a + b + c).sum() may create temporaries. When memory is tight, consider in-place ops (a += b) or reductions (np.add(a, b, out=a); np.add.reduce([...])).
- Use the right dtype: float64 is standard for numerics; float32 halves memory traffic and can be faster on some CPUs/GPUs (but mind precision).
- Preallocate: Avoid growing Python lists or repeatedly allocating arrays inside loops.
- Minimize Python in hot paths: Move heavy math into NumPy calls; keep Python for orchestration only.
- Benchmark correctly: Use large N, pin threads to 1 for fair single-core tests, and report the best of multiple runs after a warmup.

--

## Parallel Computing

High-Performance Computing relies on **parallel computing**, splitting a problem into smaller parts that can be executed *simultaneously* on multiple processors.

Instead of running one instruction at a time on one CPU core, parallel computing allows you to run many instructions on many cores or even multiple machines at once.

Parallelism can occur at different levels:
- **Within a single CPU** (multiple cores)
- **Across multiple CPUs** (distributed nodes)
- **On specialized accelerators** (GPUs or TPUs)

---

### Shared-Memory Parallelism

In **shared-memory** systems, multiple processor cores share the same memory space.  
Each core can directly read and write to the same variables in memory.

This is the model used in:
- Multicore laptops and workstations  
- *Single compute nodes* on a cluster  

Programs use **threads** to execute in parallel (e.g., with OpenMP in C/C++/Fortran or **multiprocessing in Python**).

:::{keypoints} 
Advantages:
- Easy communication between threads (shared variables)
- Low latency data access

Limitations:
- Limited by the number of cores on one machine
- Risk of race conditions if data access is not synchronized
::: 

:::{exercise} Practice with threaded parallelism in Python
Example:
```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as p:
        result = p.map(square, range(8))
    print(result)
```
:::

### Distributed-Memory Parallelism

In distributed-memory systems, each processor (or node) has its own local memory.
Processors communicate by passing messages over a network.

This is the model used when a computation spans multiple nodes in an HPC cluster.

Programs written with MPI (Message Passing Interface) use explicit communication.
Below is an example using the Python library `mpi4py` that implements MPI functions
in Python 
:::{exercise} Practice with a simple MPI program
```python
# hello_mpi.py
from mpi4py import MPI

# Initialize the MPI communicator
comm = MPI.COMM_WORLD

# Get the total number of processes
size = comm.Get_size()

# Get the rank (ID) of this process
rank = comm.Get_rank()

print(f"Hello from process {rank} of {size}")

# MPI is automatically finalized when the program exits,
# but you can call MPI.Finalize() explicitly if you prefer
```
:::

For now, do not worry about understanding this code, we will see 
`mpi4py` in detail later.

:::{keypoints}
Advantages:
- Scales to thousands of nodes
- Each process works independently, avoiding memory contention

Limitations:
- Requires explicit communication (send/receive)
- More complex programming model
- More latency, requires minimizing movement of data.
:::

### Hybrid Architectures: CPU, GPU, and TPU

Modern High-Performance Computing (HPC) systems rarely rely on CPUs alone.  
They are **hybrid architectures**, combining different types of processors, typically **CPUs**, **GPUs**, and increasingly **TPUs**, to achieve both flexibility and high performance.

---

#### CPU: The General-Purpose Processor

**Central Processing Units (CPUs)** are versatile processors capable of handling a wide range of tasks.  
They consist of a small number of powerful cores optimized for complex, sequential operations and control flow.

CPUs are responsible for:
- Managing input/output operations  
- Coordinating data movement and workflow  
- Executing serial portions of applications  

They excel in **task parallelism**, where different cores perform distinct tasks concurrently.

---

#### GPU: The Parallel Workhorse

**Graphics Processing Units (GPUs)** contain thousands of lightweight cores that can execute the same instruction on many data elements simultaneously.  
This makes them ideal for **data-parallel** workloads, such as numerical simulations, molecular dynamics, and deep learning.

GPUs are optimized for:
- Large-scale mathematical computations  
- Highly parallel tasks such as matrix and vector operations  

Common GPU computing frameworks include CUDA, HIP, OpenACC, and SYCL.  

GPUs provide massive computational throughput but require explicit management of data transfers between CPU and GPU memory.  
They are now a standard component of most modern supercomputers.

---

#### TPU: Specialized Processor for Tensor Operations

**Tensor Processing Units (TPUs)** are specialized hardware accelerators designed for tensor and matrix operations, the building blocks of deep learning and AI.  
Originally developed by Google, TPUs are now used in both cloud and research HPC environments.

TPUs focus on **tensor computations** and achieve very high performance and energy efficiency for machine learning workloads.  
They are less flexible than CPUs or GPUs but excel in neural network training and inference.

## Python in High-Performance Computing

Python has become one of the most widely used languages in scientific computing due to its simplicity, readability, and extensive ecosystem of numerical libraries.  
Although Python itself is interpreted and slower than compiled languages such as C or Fortran, it now provides a mature set of tools that allow code to **run efficiently on modern HPC architectures**.

These tools map directly to the three fundamental forms of parallelism introduced earlier:

| HPC Parallelism Type | Hardware Context | Python Solutions |
|----------------------|------------------|------------------|
| **Shared-memory parallelism** | Multicore CPUs within a node | NumPy, Numba, Pythran |
| **Distributed-memory parallelism** | Multiple nodes across a cluster | mpi4py |
| **Accelerator parallelism** | GPUs and TPUs | CuPy, JAX, Numba (CUDA) |

In practice, these technologies allow Python programs to scale from a single core to thousands of nodes on hybrid CPU–GPU systems.

---

### Shared-Memory Parallelism (Multicore CPUs)

Shared-memory parallelism occurs within a single compute node, where all CPU cores access the same physical memory.  
Python supports this level of performance primarily through **compiled numerical libraries** and **JIT (Just-In-Time) compilation**, which transform slow Python loops into efficient native machine code.

#### NumPy: Foundation of Scientific Computing

**NumPy** provides fast array operations implemented in C and Fortran.  
Its vectorized operations and BLAS/LAPACK backends **automatically** exploit shared-memory parallelism through optimized linear algebra kernels.  
Although users write Python, most computations occur in compiled native code.

#### Pythran: Static Compilation of Numerical Python Code

**Pythran** compiles numerical Python code — particularly code using NumPy — into optimized C++ extensions.  
It can automatically parallelize loops using **OpenMP**, enabling true multicore utilization without manual thread management.

Key strengths:
- Converts array-oriented Python functions into C++ for near-native speed  
- Supports automatic OpenMP parallelization for CPU cores  
- Integrates easily into existing Python workflows  

Pythran is well-suited for simulations or kernels that need to exploit multiple cores on a node.

#### Numba: JIT Compilation for Shared and Accelerator Architectures

**Numba** uses LLVM to JIT-compile Python functions into efficient machine code at runtime.  
On multicore CPUs, Numba can parallelize loops using OpenMP-like constructs; on GPUs, it can emit CUDA kernels (see below).

Main advantages:
- Minimal syntax changes required  
- Explicit parallel decorators for CPU threading  
- Compatible with NumPy arrays and ufuncs  

Together, NumPy, Pythran, and Numba enable Python to fully exploit shared-memory parallelism.

---

### Distributed-Memory Parallelism (Clusters and Supercomputers)

At large scale, HPC systems use **distributed memory**, where each node has its own local memory and must communicate explicitly.  
Python provides access to this level of parallelism through **mpi4py**, a direct interface to the standard MPI library.

#### mpi4py: Scalable Distributed Computing with MPI

**mpi4py** enables Python programs to exchange data between processes running on different nodes using MPI.  
It provides both point-to-point and collective communication primitives, identical in concept to those used in C or Fortran MPI applications.

Key features:
- Works seamlessly with NumPy arrays (zero-copy data transfer)  
- Supports all MPI operations (send, receive, broadcast, scatter, gather, reduce)  
- Compatible with job schedulers such as SLURM or PBS  

With `mpi4py`, Python can participate in large-scale distributed-memory simulations or data-parallel tasks across thousands of cores.

---

### Accelerator-Specific Parallelism (GPUs and TPUs)

Modern HPC nodes increasingly include **GPUs** or **TPUs** to accelerate numerical workloads.  
Python offers several mature libraries that interface directly with these accelerators, providing high-level syntax while executing low-level parallel kernels.

#### CuPy: GPU-Accelerated NumPy Replacement

**CuPy** mirrors the NumPy API but executes array operations on GPUs using CUDA (NVIDIA) or ROCm (AMD).  
Users can port existing NumPy code to GPUs with minimal changes, gaining massive speedups for large, data-parallel computations.

Highlights:
- NumPy-compatible array and linear algebra operations  
- Native support for multi-GPU and CUDA streams  
- Tight integration with deep learning and simulation frameworks  

#### JAX: Unified Array Computing for CPUs, GPUs, and TPUs

**JAX** combines automatic differentiation and XLA-based compilation to execute Python functions efficiently on CPUs, GPUs, and TPUs.  
It is particularly well-suited for scientific machine learning and differentiable simulations.

Key strengths:
- Just-In-Time (JIT) compilation via XLA  
- Transparent execution on accelerators (GPU, TPU)  
- Built-in vectorization and automatic differentiation  

JAX provides a single high-level API for heterogeneous HPC nodes, seamlessly handling hybrid CPU–GPU–TPU workflows.

---

### Summary: Python Across HPC Architectures

Python can now leverage **all layers of hybrid HPC architectures** through specialized libraries:

| Architecture | Parallelism Type | Typical Python Tools | Example Use Cases |
|---------------|------------------|----------------------|-------------------|
| **Multicore CPUs** | Shared memory | NumPy, Pythran, Numba | Numerical kernels, vectorized math |
| **Clusters** | Distributed memory | mpi4py | Large-scale simulations, domain decomposition |
| **GPUs / TPUs** | Accelerator parallelism | CuPy, JAX, Numba (CUDA) | Machine learning, dense linear algebra |

Together, these tools allow Python to serve as a *high-level orchestration language* that transparently scales from a single laptop core to full supercomputing environments — integrating shared-memory, distributed-memory, and accelerator-based parallelism in one ecosystem.

---

:::{keypoints}
- Python’s ecosystem maps naturally onto hybrid HPC architectures.  
- **NumPy, Numba, and Pythran** exploit shared-memory parallelism on multicore CPUs.  
- **mpi4py** extends Python to distributed-memory clusters.  
- **CuPy and JAX** enable acceleration on GPUs and TPUs.  
- These libraries allow researchers to combine high productivity with near-native performance across all layers of HPC systems.
:::

