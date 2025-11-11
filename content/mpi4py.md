# Introduction to MPI with Python (mpi4py)

:::{questions}
- What is MPI, and how does it enable parallel programs to communicate?
- How does Python implement MPI through the `mpi4py` library?
- What are point-to-point and collective communications?
- How does `mpi4py` integrate with NumPy for efficient data exchange?
:::

:::{objectives}
- Understand the conceptual model of MPI: processes, ranks, and communication.
- Distinguish between point-to-point and collective operations.
- Recognize how NumPy arrays act as communication buffers in `mpi4py`.
- See how `mpi4py` bridges Python and traditional HPC concepts.
:::

---

## What Is MPI?

**MPI (Message Passing Interface)** is a standard for communication among processes that run on distributed-memory systems, such as HPC clusters.  

When you run an MPI program, the system **creates multiple independent processes**, each running its *own copy* of the same program.  
Each process is assigned a unique identifier called a **rank**, and all ranks together form a **communicator** (most commonly `MPI.COMM_WORLD`).

Key ideas:
- Every process runs the same code, but can behave differently depending on its rank.  
- Processes do not share memory, they communicate by **sending and receiving messages**.  
- MPI provides a consistent way to exchange data across nodes and processors.

Conceptually, this is sometimes called the **SPMD (Single Program, Multiple Data)** model.

In order to see how this works let us try this simple code snippet:
```python
# mpi_hello.py
from mpi4py import MPI

# Initialize communicator
comm = MPI.COMM_WORLD

# Get the number of processes
size = comm.Get_size()

# Get the rank (ID) of this process
rank = comm.Get_rank()

# Print a message from each process
print(f"Hello from process {rank} out of {size}")
```

---

## Point-to-Point Communication

The most basic form of communication in MPI is **point-to-point**, meaning data is sent from one process directly to another.  

Each message involves:
- A **sender** and a **receiver**
- A **tag** identifying the message type
- A **data buffer** that holds the information being transmitted

Typical operations:
- **Send:** one process transmits data.
- **Receive:** another process waits for that data.

In `mpi4py`, each of these operations maps directly to MPI’s underlying mechanisms but with a simple Python interface.  
Conceptually, this allows one process to hand off a message to another in a fully parallel environment.

Examples of conceptual use cases:
- Distributing different chunks of data to multiple workers.
- Passing boundary conditions between neighboring domains in a simulation.

To illustrate this let us construct a minimal example:

```python
# mpi_send_recv.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Process 0 sends a message
    data = "Hello from process 0"
    comm.send(data, dest=1)  # send to process 1
    print(f"Process {rank} sent data: {data}")

elif rank == 1:
    # Process 1 receives a message
    data = comm.recv(source=0)  # receive from process 0
    print(f"Process {rank} received data: {data}")

else:
    # Other ranks do nothing
    print(f"Process {rank} is idle")
```

---

## Collective Communication

While point-to-point operations handle pairs of processes, **collective operations** involve all processes in a communicator.  
They provide coordinated data exchange and synchronization patterns that are efficient and scalable.

Common collectives include:
- **Broadcast:** One process sends data to all others.  
- **Scatter:** One process distributes distinct pieces of data to each process.  
- **Gather:** Each process sends data back to a root process.  
- **Reduce:** All processes combine results using an operation (e.g., sum, max).  

Collectives are conceptually similar to group conversations, where every participant either contributes, receives, or both.  
They are essential for algorithms that require sharing intermediate results or aggregating outputs.

Example broadcast + gather
```python
# mpi_collectives.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Broadcast example ---
if rank == 0:
    data = "Hello from the root process"
else:
    data = None

# Broadcast data from process 0 to all others
data = comm.bcast(data, root=0)
print(f"Process {rank} received: {data}")

# --- Gather example ---
# Each process creates its own message
local_msg = f"Message from process {rank}"

# Gather all messages at the root process (rank 0)
all_msgs = comm.gather(local_msg, root=0)

if rank == 0:
    print("\nGathered messages at root:")
    for msg in all_msgs:
        print(msg)
```

---

## Integration with NumPy: Buffer-Like Objects

One of the strengths of `mpi4py` is its **tight integration with NumPy**.  
MPI operations in `mpi4py` can directly send and receive **buffer-like objects**, such as NumPy arrays, without copying data back and forth between Python and C memory.

Conceptually:
- Each NumPy array exposes its underlying memory buffer.
- MPI can access this memory region directly.
- This avoids serialization overhead (no need for `pickle`) and achieves near-native C performance.

This makes it possible to:
- Efficiently distribute large numerical datasets across processes.  
- Perform collective operations directly on arrays.  
- Integrate `mpi4py` seamlessly into scientific Python workflows.

For example, collective reductions like sums or means across large NumPy arrays are routine building blocks in parallel simulations and machine learning workloads.

Example with collectives:
```python
# mpi_numpy_collectives.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total number of elements in the big array (must be divisible by size)
N = 10_000_000

# Only rank 0 creates the full array
if rank == 0:
    big_array = np.ones(N, dtype="float64")  # for simplicity, all ones
else:
    big_array = None

# Each process will receive a chunk of this size
local_N = N // size

# Allocate local buffer on each process
local_array = np.empty(local_N, dtype="float64")

# Scatter the big array from root to all processes
comm.Scatter(
    [big_array, MPI.DOUBLE],  # send buffer (only meaningful on root)
    [local_array, MPI.DOUBLE],  # receive buffer on every rank
    root=0,
)

# Each process computes a local sum
local_sum = np.sum(local_array)

# Reduce all local sums to a global sum on root
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Global sum = {global_sum}")
    print(f"Expected   = {float(N)}")
```

To really understand the power of collectives, we can have a look at the same code using p2p communication:
```python
# mpi_numpy_point_to_point.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total number of elements in the big array (must be divisible by size)
N = 10_000_000
local_N = N // size

if rank == 0:
    # Root creates the full big array
    big_array = np.ones(N, dtype="float64")  # all ones for simplicity

    # --- Send chunks to other ranks ---
    for dest in range(1, size):
        start = dest * local_N
        end = (dest + 1) * local_N
        chunk = big_array[start:end]
        comm.Send([chunk, MPI.DOUBLE], dest=dest, tag=0)

    # Root keeps its own chunk (rank 0)
    local_array = big_array[0:local_N]

else:
    # Other ranks allocate a buffer and receive their chunk
    local_array = np.empty(local_N, dtype="float64")
    comm.Recv([local_array, MPI.DOUBLE], source=0, tag=0)

# Each process computes its local sum
local_sum = np.sum(local_array)

if rank == 0:
    # Root starts with its own local sum
    global_sum = local_sum

    # --- Receive partial sums from all other ranks ---
    for source in range(1, size):
        recv_sum = comm.recv(source=source, tag=1)
        global_sum += recv_sum

    print(f"Global sum (point-to-point) = {global_sum}")
    print(f"Expected                    = {float(N)}")

else:
    # Other ranks send their local sum back to root
    comm.send(local_sum, dest=0, tag=1)
```
---

## Summary

**mpi4py** provides a simple yet powerful bridge between Python and the Message Passing Interface used in traditional HPC applications.  
Conceptually, it introduces the same communication paradigms used in compiled MPI programs but with Python’s expressiveness and interoperability.

| Concept | Description |
|----------|-------------|
| **Process** | Independent copy of the program with its own memory space |
| **Rank** | Identifier for each process within a communicator |
| **Point-to-Point** | Direct communication between pairs of processes |
| **Collective** | Group communication involving all processes |
| **NumPy Buffers** | Efficient memory sharing for large numerical data |

mpi4py allows Python users to write distributed parallel programs that scale from laptops to supercomputers, making it an invaluable tool for modern scientific computing.

---

:::{keypoints}
- MPI creates multiple independent processes running the same program.  
- Point-to-point communication exchanges data directly between two processes.  
- Collective communication coordinates data exchange across many processes.  
- mpi4py integrates tightly with NumPy for efficient, zero-copy data transfers.  
- These concepts allow Python programs to scale effectively on HPC systems.
:::