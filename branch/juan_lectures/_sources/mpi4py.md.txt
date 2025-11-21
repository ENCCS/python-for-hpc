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

**MPI (Message Passing Interface)** is a standardized programming model for communication among processes that run on **distributed-memory systems**, such as HPC clusters.

In a distributed-memory system, each compute node (or process) has its **own local memory**.  
Unlike shared-memory systems, where threads can directly read and write to a common address space, distributed processes **cannot directly access each other’s memory**.  
To collaborate, they must explicitly **send and receive messages** containing the data they need to share.

### Independent Processes and the SPMD Model

When you run an MPI program, the system launches **multiple independent processes**, each running its **own copy** of the same program.  
This design is fundamental: because each process owns its own memory space, it must contain its own copy of the code to execute its portion of the computation.

Each process:
- Runs the same code but operates on a different subset of the data.  
- Is identified by a unique number called its **rank**.  
- Belongs to a **communicator**, a group of processes that can exchange messages (most commonly `MPI.COMM_WORLD`).

This model is known as **SPMD: Single Program, Multiple Data**:  
a single source program runs simultaneously on many processes, each working on different data.

### Why Copies of the Program Are Needed?

Because processes in distributed memory do not share variables or memory addresses, each process must have:
- Its **own copy of the executable code**, and  
- Its **own private workspace (variables, arrays, etc.)**.

This independence is crucial for scalability:
- Each process can execute independently without memory contention.  
- The program can scale to thousands of nodes, since no shared memory bottleneck exists.  
- Data movement becomes explicit and controllable, ensuring predictable performance on large clusters.

### Sharing Data Between Processes

Although memory is not shared, processes can **cooperate** by exchanging information through **message passing**.  
MPI defines two main communication mechanisms:

1. **Point-to-point communication**: Data moves **directly** between two processes.  
2. **Collective communication**: Data is exchanged among **all processes** in a communicator in a coordinated way.  

:::{keypoints}
- **Process:** Each MPI program runs as multiple independent processes, not threads.  
- **Rank:** Every process has a unique identifier (its *rank*) within a communicator, used to distinguish and coordinate them.  
- **Communication:** Processes exchange data explicitly through message passing, either **point-to-point** (between pairs) or **collective** (among groups).  

Together, these three ideas form the foundation of MPI’s model for parallel computing.
:::

---

## mpi4py

`mpi4py` is the standard Python interface to the **Message Passing Interface (MPI)**, the same API used by C, C++, and Fortran codes for distributed-memory parallelism.  
It allows Python programs to run on many processes, each with its own memory space, communicating through explicit messages.

### Communicators and Initialization

In MPI, all communication occurs through a **communicator**, an object that defines which processes can talk to each other.  
When a program starts, each process automatically becomes part of a predefined communicator called **`MPI.COMM_WORLD`**.

This object represents *all processes* that were launched together by `mpirun` or `srun`.

A typical initialization pattern looks like this:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD      # Initialize communicator
size = comm.Get_size()     # Total number of processes
rank = comm.Get_rank()     # Rank (ID) of this process

print(f"I am rank {rank} of {size}")
```
Every process executes the same code, but rank and size allow them to behave differently.

:::{exercise} Hello world MPI
Copy and paste this code and execute it using `mpirun -n N mpi_hello.py` where `N` is the number of tasks. \
**Note:** Do not put more tasks than the number of cores that your computer has.

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
print(f"Hello world")
```
This code snippet illustrates how independent processes run copies of the program. \
To practice further try the following:
1. Use the `rank` variable to print the square of `rank` in each rank.
2. Make the program print only in rank 0, hint: `if rank == 0:`
:::
:::{solution}

*Solution 1:* Print the square of each rank
```python
# mpi_hello_square.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Each process prints its rank and the square of its rank
print(f"Process {rank} of {size} has value {rank**2}")
```
*Solution 2:* Print only one process (rank 0)
```python
# mpi_hello_rank0.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print(f"Hello world from the root process (rank {rank}) out of {size} total processes")
```
:::

### Method Naming Convention

In `mpi4py`, most MPI operations exist in **two versions**, a *lowercase* and an *uppercase* form, that differ in how they handle data.

| Convention | Example | Description |
|-------------|----------|-------------|
| **Lowercase methods** | `send()`, `recv()`, `bcast()`, `gather()` | High-level, Pythonic methods that can send and receive arbitrary Python objects. Data is automatically serialized (pickled). Simpler to use but slower for large data. |
| **Uppercase methods** | `Send()`, `Recv()`, `Bcast()`, `Gather()` | Low-level, performance-oriented methods that operate on **buffer-like objects** such as NumPy arrays. Data is transferred directly from memory without serialization, achieving near-C speed. |

**Rule of thumb:**  
Use *lowercase* methods for small control messages or Python objects,  
and *uppercase* methods for numerical data stored in arrays when performance matters.

#### Syntax differences
**Lowercase (Python objects):**
```python
comm.send(obj, dest=1)
data = comm.recv(source=0)
```
- The message (obj) can be any Python object.
- MPI automatically serializes and deserializes it internally.
- Fewer arguments: simple but less efficient for large data.

**Uppercase (buffer-like objects, e.g., NumPy arrays):**
```python
comm.Send([array, MPI.DOUBLE], dest=1)
comm.Recv([array, MPI.DOUBLE], source=0)
```
- Requires explicit definition of the data buffer and its MPI datatype. (same syntax as C++)
- Works directly with the memory address of the array (no serialization).
- Achieves maximum throughput for numerical and scientific workloads.

---

## Point-to-Point Communication

The most basic form of communication in MPI is **point-to-point**, meaning data is sent from one process directly to another.  

Each message involves:
- A **sender** and a **receiver**
- A **tag** identifying the message type
- A **data buffer** that holds the information being transmitted

These operations are methods of the class `MPI.COMM_WORLD`. This means that one needs to initialize it

Typical operations:
- **Send:** one process transmits data. 
- **Receive:** another process waits for that data.

In `mpi4py`, each of these operations maps directly to MPI’s underlying mechanisms but with a simple Python interface.  
Conceptually, this allows one process to hand off a message to another in a fully parallel environment.

Examples of conceptual use cases:
- Distributing different chunks of data to multiple workers.
- Passing boundary conditions between neighboring domains in a simulation.

:::{exercise} Point-to-Point Communication
Copy and paste the code below into a file called `mpi_send_recv.py`.
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
Run the program using:
`mpirun -n 3 python mpi_send_recv.py`
You should see output indicating that process 0 sent data and process 1 received it, while all others remained idle.
Now try:
1.	Change the roles:
Make process 1 send a reply back to process 0 after receiving the message.
Use `comm.send()` and `comm.recv()` in both directions.
2.	Blocking communication:
Notice that `comm.send()` and `comm.recv()` are blocking operations.
- Add a short delay using `time.sleep(rank)` before sending or receiving.
- Observe how process 0 must wait until process 1 calls `recv()` before it can continue, and vice versa.
- Try swapping the order of the calls (e.g., both processes call `send()` first), what happens?
- You may notice the program hangs or deadlocks, because both processes are waiting for a `recv()` that never starts.
:::

:::{solution}
*Solution 1:* Change the roles (reply back):
```python
# mpi_send_recv_reply.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data_out = "Hello from process 0"
    comm.send(data_out, dest=1)
    print(f"Process {rank} sent: {data_out}")

    data_in = comm.recv(source=1)
    print(f"Process {rank} received: {data_in}")

elif rank == 1:
    data_in = comm.recv(source=0)
    print(f"Process {rank} received: {data_in}")

    data_out = "Reply from process 1"
    comm.send(data_out, dest=0)
    print(f"Process {rank} sent: {data_out}")

else:
    print(f"Process {rank} is idle")
```
*Solution 2:* Blocking communication behavior:
1.	Add a delay (e.g., time.sleep(rank)) before send/recv and observe that each blocking call waits for its partner. Example:
```python
# mpi_blocking_delay.py
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

time.sleep(rank)  # stagger arrival

if rank == 0:
    comm.send("msg", dest=1)
    print("0 -> sent")
    print("0 -> got:", comm.recv(source=1))

elif rank == 1:
    print("1 -> got:", comm.recv(source=0))
    comm.send("ack", dest=0)
    print("1 -> sent")
```
2.	Deadlock demonstration:
```python
# mpi_deadlock.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank in (0, 1):
    # Both ranks call send() first -> potential deadlock
    comm.send(f"from {rank}", dest=1-rank)
    # This recv may never be reached if partner is also stuck in send()
    print("received:", comm.recv(source=1-rank))
```
:::
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

:::{exercise} Collectives
Let us run this code to see the collectives `bcast` and `gather` in action:

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
Now try the following: 
1. Change the root process: In the broadcast section, change the root process from `rank 0` to `rank 1`.
2. How would be the same done with point to point communication?
:::
:::{solution}
*Solution 1:* Change the root process:
The root process is the one that handles the behaviour of the collectives. So we just need to change the root
of the collective **broadcast**.
```python
# --- Broadcast example ---
if rank == 1:
    data = "Hello from process 1 (new root)"
else:
    data = None

# Broadcast data from process 1 to all others
data = comm.bcast(data, root=1)
print(f"Process {rank} received: {data}")
```
*Solution 2:* Manual broadcasting the previous code:
To reproduce a broadcast manually using only send() and recv(), one could write:
```python
# Manual broadcast using point-to-point
if rank == 1:
    data = "Hello from process 1 (manual broadcast)"
    # Send to all other processes
    for dest in range(size):
        if dest != rank:
            comm.send(data, dest=dest)
else:
    data = comm.recv(source=1)

print(f"Process {rank} received: {data}")
```
:::
---

## Integration with NumPy: Buffer-Like Objects

A major strength of `mpi4py` is its **direct integration with NumPy arrays**.  
MPI operations can send and receive **buffer-like objects**, such as NumPy arrays, without copying data between Python and C memory.

:::{keypoints} Important
Remember that **buffer-like objects** can be used with the **uppercase methods** for avoiding serialization and its time overhead.
:::
Because NumPy arrays expose their internal memory buffer, MPI can access this data directly.  
This eliminates the need for serialization (no `pickle` step) and allows **near-native C performance** for communication and collective operations.

Conceptually:
- Each NumPy array acts as a **contiguous memory buffer**.  
- MPI transfers data directly from this buffer to another process’s memory.  
- This mechanism is ideal for large numerical datasets, enabling efficient data movement in parallel programs.

This integration makes it possible to:
- Distribute large datasets across processes using **collectives** like `Scatter` and `Gather`.  
- Combine results efficiently with operations like `Reduce` or `Allreduce`.  
- Seamlessly integrate parallelism into scientific Python workflows.

---

:::{exercise} Collective Operations on NumPy Arrays
In this example, you will see how collective MPI operations distribute and combine large arrays across multiple processes using **buffer-based communication**.

Save the following code as `mpi_numpy_collectives.py` and run it with multiple processes:

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
    [big_array, MPI.DOUBLE],       # send buffer (only valid on root)
    [local_array, MPI.DOUBLE],     # receive buffer on all processes
    root=0,
)

# Each process computes a local sum
local_sum = np.sum(local_array)

# Reduce all local sums to a global sum on the root process
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Global sum = {global_sum}")
    print(f"Expected   = {float(N)}")
```
Run the program using
```bash
mpirun -n 4 python mpi_numpy_collectives.py
```
Questions:
1.	Which MPI calls distribute and collect data in this program?
2.	Why is it necessary to preallocate local_array on every process?
3.	What would happen if you used lowercase methods (scatter, reduce) instead of Scatter, Reduce?
:::
:::{solution}

*Solution 1:* The MPI calls that distribute and collect data in this program are comm.Scatter() and comm.reduce().
Scatter divides the large NumPy array on the root process and sends chunks to all ranks, while Reduce collects the locally computed results and combines them (using MPI.SUM) into a single global result on the root process.

*Solution 2:* It is necessary to preallocate local_array on every process because the uppercase MPI methods (Scatter, Gather, Reduce, etc.) work directly with memory buffers.
Each process must provide a fixed, correctly sized buffer so that MPI can write received data directly into it without additional memory allocation or copying.

*Solution 3:* If lowercase methods (scatter, reduce) were used instead, MPI would serialize and deserialize the Python objects being communicated (using pickle).
This would make the program simpler but significantly slower for large numerical arrays, since it adds extra copying and memory overhead.
Using the uppercase buffer-based methods avoids this cost and achieves near-native C performance.
:::
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