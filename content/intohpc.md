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
can have multiple **CPUs** (and sometimes **GPUs**) that run tasks in
parallel.

### Typical HPC Components

| Component | Description |
|------------|--------------|
| **Login node** | Where you connect and submit jobs |
| **Compute nodes** | Machines where your program actually runs |
| **Scheduler** | Manages job submissions and allocates resources (e.g. SLURM) |
| **Storage** | Shared file system accessible to all nodes |

---

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

Advantages:
	•	Easy communication between threads (shared variables)
	•	Low latency data access

Limitations:
	•	Limited by the number of cores on one machine
	•	Risk of race conditions if data access is not synchronized

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

How many CPU cores does your machine have? Try changing the number of workers.
:::

### Distributed-Memory Parallelism

In distributed-memory systems, each processor (or node) has its own local memory.
Processors communicate by passing messages over a network.

This is the model used when a computation spans multiple nodes in an HPC cluster.

Programs written with MPI (Message Passing Interface) use explicit communication.
Below is an example using the Python library `mpi4py` that implements MPI functions
in Python 

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
For now, do not worry about understanding this code, we will see 
`mpi4py` in detail later.

Advantages:
	•	Scales to thousands of nodes
	•	Each process works independently, avoiding memory contention

Limitations:
	•	Requires explicit communication (send/receive)
	•	More complex programming model
  •	More latency, requires minimizing movement of data.


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

Python is one of the most widely used languages in scientific research due to its simplicity, readability, and rich ecosystem of numerical libraries.  
While Python itself is interpreted and not as fast as compiled languages like C or Fortran, it has developed a strong foundation for **high-performance computing (HPC)** through specialized libraries and interfaces.

These tools allow scientists and engineers to write code in Python while still achieving near-native performance on CPUs, GPUs, and distributed systems.

---

#### JAX: High-Performance Array Computing with Accelerators

**JAX** is a library developed by Google for high-performance numerical computing and automatic differentiation.  
It combines the familiar NumPy-like interface with **just-in-time (JIT)** compilation using **XLA (Accelerated Linear Algebra)**.  
This allows Python functions to be compiled and executed efficiently on both **CPUs and GPUs**.

Key features:
- Transparent acceleration on GPUs and TPUs  
- Automatic differentiation for machine learning and optimization  
- Vectorization and parallel execution across devices  

JAX is widely used in scientific machine learning, physics-informed models, and numerical optimization on large-scale clusters.

---

#### Pythran: Static Compilation of Numerical Python Code

**Pythran** is a Python-to-C++ compiler designed to speed up numerical Python code, especially code that uses **NumPy**.  
It allows scientists to write high-level Python functions and then compile them into efficient native extensions that run close to C or Fortran performance.

Unlike tools that interface with existing Fortran code, Pythran works by **analyzing and translating Python source code itself** into optimized C++ code under the hood.  
This makes it especially useful for accelerating array-oriented computations without leaving Python syntax.

Key features:
- Compiles numerical Python code (especially with NumPy) to efficient machine code  
- Supports automatic parallelization via OpenMP  
- Produces portable C++ extensions that can be imported directly into Python  
- Requires only minimal code annotations to guide type inference  

Typical use cases include:
- Scientific simulations and numerical kernels  
- Loops or array operations that are too slow in pure Python  
- HPC applications that need native performance but prefer to stay within the Python ecosystem  

Pythran complements other tools such as **Numba** and **Cython**, giving scientists a flexible pathway to accelerate Python code without rewriting it in C or Fortran.

#### Numba: Just-In-Time Compilation for CPUs and GPUs

**Numba** is a Just-In-Time (JIT) compiler that translates numerical Python code into fast machine code at runtime using LLVM.  
It requires minimal code changes and provides performance close to C or Fortran.

Main advantages:
- Speeds up array-oriented computations on CPUs  
- Enables parallel loops and multi-threading  
- Supports GPU acceleration through CUDA targets  

Numba is ideal for users who want to optimize existing Python scripts for parallel CPU or GPU execution without rewriting them in another language.

---

#### CuPy: GPU-Accelerated Array Library

**CuPy** provides a drop-in replacement for NumPy that runs on **NVIDIA GPUs** and also **AMD GPUs**.  
It mimics the NumPy API, allowing users to accelerate numerical code simply by changing the import statement, while taking full advantage of GPU parallelism.

Highlights:
- NumPy-compatible syntax for easy transition  
- GPU-backed arrays and linear algebra operations  
- Integration with deep learning and simulation workflows  

CuPy is particularly useful for applications with large data arrays or intensive numerical workloads, such as molecular modeling, image processing, and machine learning.

---

#### mpi4py: Parallel and Distributed Computing with MPI

**mpi4py** provides Python bindings to the standard **Message Passing Interface (MPI)**, allowing distributed-memory parallelism across multiple processes or nodes.  
It enables Python programs to run efficiently on supercomputers, leveraging the same communication model used in large-scale C or Fortran HPC applications.

Capabilities:
- Point-to-point and collective communication  
- Parallel I/O and reduction operations  
- Compatibility with SLURM and other HPC schedulers  

mpi4py bridges Python’s high-level usability with the scalability of traditional HPC systems, making it possible to prototype and deploy distributed applications quickly.

