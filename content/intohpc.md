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

## Connecting to a Cluster

You usually access an HPC system through a **login node** using SSH.

```bash
ssh username@hpc-login.example.org
```
Once connected, you can see what resources are available using the
scheduler’s commands (for example, SLURM):

```bash
sinfo
```

:::{exercise} Exploring the Cluster
	1.	Log in to your training cluster using ssh.
	2.	Run sinfo to see available partitions and nodes.
	3.	Try squeue to view currently running jobs.
:::

---

## Parallel Computing

High-Performance Computing relies on **parallel computing**, splitting a problem into smaller parts that can be executed *simultaneously* on multiple processors.

Instead of running one instruction at a time on one CPU core, parallel computing allows you to run many instructions on many cores or even multiple machines at once.

Parallelism can occur at different levels:
- **Within a single CPU** (multiple cores)
- **Across multiple CPUs** (distributed nodes)
- **On specialized accelerators** (GPUs)

---

### Shared-Memory Parallelism

In **shared-memory** systems, multiple processor cores share the same memory space.  
Each core can directly read and write to the same variables in memory.

This is the model used in:
- Multicore laptops and workstations  
- *Single compute nodes* on a cluster  

Programs use **threads** to execute in parallel (e.g., with OpenMP in C/C++/Fortran or multiprocessing in Python).

Example (OpenMP-style pseudocode):

```c
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    a[i] = b[i] + c[i];
}
```
Advantages:
	•	Easy communication between threads (shared variables)
	•	Low latency data access

Limitations:
	•	Limited by the number of cores on one machine
	•	Risk of race conditions if data access is not synchronized

:::{exercise} Shared-Memory Example
Try using Python’s multiprocessing module to parallelize a simple loop:
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as p:
        result = p.map(square, range(8))
    print(result)

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

---

#### Putting It All Together

Modern HPC clusters combine these processor types to create **heterogeneous nodes**.  
A single node may contain multiple CPUs and several GPUs or TPUs, interconnected through high-speed buses.

| Processor Type | Main Role | Best Suited For |
|-----------------|------------|-----------------|
| **CPU** | General-purpose control, coordination, and I/O | Serial and logic-heavy tasks |
| **GPU** | Massive parallel numerical computation | Data-parallel workloads and simulations |
| **TPU** | Specialized tensor acceleration | Machine learning and AI workloads |

This combination allows each processor type to handle the tasks for which it is best optimized, enabling both high performance and efficient energy use.

---

#### Hybrid Programming Models

To take full advantage of hybrid systems, programmers often combine multiple parallel models within one application:

- **MPI + OpenMP:** Distributed memory between nodes and shared memory within each node.  
- **MPI + CUDA or OpenACC:** Distributed communication across nodes, GPU acceleration within nodes.  
- **MPI + XLA or TPU runtime:** MPI coordination with tensor operations handled on TPUs.

This layered approach leverages:
- **Distributed memory** across cluster nodes  
- **Shared memory** and **accelerators** within each node  

Such **hybrid parallelism** is now the dominant model in large-scale scientific and industrial computing.

---

:::{keypoints}
- Modern HPC systems combine CPUs, GPUs, and TPUs in hybrid architectures.  
- CPUs handle coordination and control, while GPUs and TPUs execute highly parallel workloads.  
- GPUs excel at general numerical parallelism; TPUs specialize in tensor operations for AI.  
- Hybrid programming models (MPI with OpenMP, CUDA, or TPU backends) enable efficient use of all resources.
:::

---

### Hybrid Programming Models

Modern HPC applications are rarely limited to a single parallel programming model.  
Instead, they often combine multiple layers of parallelism to make the best use of available hardware — an approach known as **hybrid programming**.

Hybrid programming allows developers to exploit **distributed memory parallelism** across nodes and **shared memory or accelerator parallelism** within nodes.  
This enables applications to scale efficiently from a few cores to thousands of nodes on large supercomputers.

---

#### MPI + OpenMP: Hybrid Parallelism on CPU Nodes

When an HPC cluster consists of CPU-only nodes, a common hybrid model combines **MPI (Message Passing Interface)** and **OpenMP (Open Multi-Processing)**.

- **MPI** handles communication between nodes in the cluster.  
  Each MPI process runs on a separate node or subset of nodes and uses message passing to exchange data.  

- **OpenMP** provides shared-memory parallelism within each node.  
  Multiple threads inside a single MPI process can execute concurrently on the cores of that node.

This structure creates a **two-level hierarchy** of parallelism:
1. **Inter-node parallelism** — achieved through MPI (distributed memory).  
2. **Intra-node parallelism** — achieved through OpenMP (shared memory).

**Benefits of MPI + OpenMP hybridization:**
- Reduces the total number of MPI processes, decreasing communication overhead.  
- Exploits the full capacity of multi-core CPUs within each node.  
- Balances scalability (via MPI) and efficiency (via OpenMP threads).  

**Challenges:**
- Requires careful balancing between MPI processes and OpenMP threads per node.  
- Data locality and thread affinity must be managed to avoid performance loss.

This model is widely used in simulation codes such as computational fluid dynamics, finite element methods, and electronic structure calculations.

---

#### MPI + GPU: Hybrid Parallelism on Heterogeneous Nodes

When HPC nodes include GPUs or other accelerators, the hybrid model extends naturally to **MPI + GPU programming**.  
In this approach, **MPI** is still responsible for communication between nodes, while **GPU programming frameworks** (such as CUDA, HIP, or OpenACC) handle massive data-parallel computations within each node.

The workflow typically looks like this:
- Each node runs one or more MPI processes.  
- Each MPI process controls one or more GPUs.  
- Within a node, the heavy numerical work is offloaded from the CPU to the GPU.  

This model provides three complementary layers of parallelism:
1. **MPI level:** communication and coordination between nodes.  
2. **Node level:** management and scheduling by the CPU host.  
3. **Device level:** massive parallel computation executed by the GPU.  

**Benefits of MPI + GPU hybridization:**
- Enables scaling across thousands of GPUs distributed among cluster nodes.  
- Delivers enormous speedups for highly parallel workloads (e.g., molecular dynamics, deep learning, weather modeling).  
- Frees CPU resources for orchestration and non-parallel tasks.

**Challenges:**
- Requires explicit management of data transfer between CPU and GPU memory.  
- Increased complexity in software development and debugging.  
- Performance depends on matching problem size and data layout to GPU architecture.

Hybrid MPI + GPU programming has become the dominant model for modern supercomputers, including those in the **Top500** list, as nearly all of them are now GPU-accelerated.

---

#### Choosing the Right Hybrid Model

The choice between MPI + OpenMP and MPI + GPU depends on:
- **Hardware configuration:** CPU-only clusters vs. GPU-accelerated clusters.  
- **Application characteristics:** memory usage, data dependencies, and computational intensity.  
- **Scalability goals:** how far the problem needs to scale across nodes and cores.  

| Hybrid Model | Hardware | Parallelism Within Node | Inter-node Parallelism | Typical Use |
|---------------|-----------|--------------------------|------------------------|--------------|
| **MPI + OpenMP** | Multi-core CPU nodes | Shared-memory (threads) | MPI processes | General-purpose scientific computing |
| **MPI + GPU** | CPU + GPU nodes | GPU data parallelism | MPI processes | HPC with heavy numerical or ML workloads |

Both models represent essential building blocks of high-performance computing, and many large-scale codes support both configurations for flexibility across systems.

---

:::{keypoints}
- Hybrid programming combines multiple parallel models to fully utilize modern hardware.  
- **MPI + OpenMP** is the standard approach for CPU-only systems.  
- **MPI + GPU** extends hybrid parallelism to heterogeneous CPU–GPU architectures.  
- Choosing the right model depends on hardware, scalability, and the computational nature of the application.
:::

### Python in High-Performance Computing

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

#### PyTrAn: Python–Fortran Integration for Legacy HPC Codes

**PyTrAn** (Python–Fortran Translator or Interface) is designed to connect Python with legacy Fortran routines, allowing developers to reuse existing HPC codebases efficiently.  
Many scientific applications — such as weather modeling, quantum chemistry, and fluid dynamics — rely on decades of optimized Fortran code.

Through interfaces like **f2py** or PyTrAn, Python can:
- Call high-performance Fortran routines directly  
- Combine new Python modules with legacy HPC libraries  
- Simplify workflows while maintaining computational efficiency  

This integration makes Python a flexible *glue language* in hybrid HPC applications.

---

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

**CuPy** provides a drop-in replacement for NumPy that runs on **NVIDIA GPUs**.  
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

---

#### Summary: Python’s Role in Modern HPC

Python now plays an essential role in high-performance and scientific computing.  
Thanks to tools such as JAX, PyTrAn, Numba, CuPy, and mpi4py, Python can serve as both a **high-level interface** for productivity and a **high-performance backend** through compilation and parallelization.

| Library | Focus Area | Target Hardware | Key Strength |
|----------|-------------|-----------------|---------------|
| **JAX** | Automatic differentiation, numerical computing | CPU, GPU, TPU | JIT compilation and accelerator support |
| **PyTrAn** | Interfacing Python with Fortran codes | CPU | Integration with legacy HPC software |
| **Numba** | Just-in-time compilation | CPU, GPU | Native performance with minimal code changes |
| **CuPy** | GPU-accelerated array computing | GPU | Drop-in NumPy replacement for GPUs |
| **mpi4py** | Distributed parallel computing | CPU clusters | MPI-based scaling across nodes |

These libraries make Python a truly viable language for HPC workflows — from prototyping to full-scale production on modern supercomputers.

---

:::{keypoints}
- Python’s modern ecosystem enables real HPC performance through JIT compilation and hardware acceleration.  
- JAX, Numba, and CuPy provide acceleration on CPUs, GPUs, and TPUs.  
- PyTrAn bridges Python with legacy Fortran codes used in scientific computing.  
- mpi4py enables distributed-memory parallelism across clusters using MPI.  
- Together, these tools make Python a powerful and accessible option for modern HPC.
:::