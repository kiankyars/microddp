## Syllabus

### [Monolith](https://github.com/kiankyars/microddp/blob/main/src/monolith.py)

- **Concept:** Establish a baseline on a single CPU/GPU.
- **Lab:** Just `nn.Sequential` with 16 layers and a simple training loop.

### Motivation for DP

- **The Data Wall:** Why we need to process more data faster.
  - **Batch Size:** 1 Million samples.
  - **Training Time:** Days on a single GPU.
  - **Solution:** Data Parallelism (replicate model, split data).

### [Manual](https://github.com/kiankyars/microddp/blob/main/src/manual.py)

Manually split the batch across two "GPUs" and average gradients.

- **The Exercise:** Try to train by manually averaging gradients from two model copies.
- **The Lesson:** You have to manually synchronize gradients; autograd computes them locally.

### Distributed Basics

- **Concept:** Rank, World Size, and Process Group.
  - **The Process Group:** Imagine a conference call. Before anyone can talk, they must dial in. `init_process_group` is dialing in.
  - **World Size:** The total number of people on the call (e.g., 4 GPUs).
  - **Rank:** Your unique ID badge (0, 1, 2, 3).
  - **Rank 0** is the "Boss" (usually handles logging, saving checkpoints, and data loading).
- **Concept:** What torchrun does.
  - **Process Isolation:** `torchrun` spawns completely separate Python interpreter instances. Each has its own memory space.
  - **True Parallelism:** Because these are separate processes (not threads), the OS schedules them across different physical CPU cores or GPUs.
  - **The Network Bridge:** When you call `dist.all_reduce`, the data is synchronized across all processes.

### Data Parallelism

#### [Naive DP](./naive.md)

- **Concept:** Manual gradient averaging after backward pass.
- **Lab:** Implement Naive Data Parallel. Forward on local chunk, backward, then all-reduce gradients.

#### [DDP](./ddp.md)

- **Concept:** Gradient hooks for automatic all-reduce during backward.
- **Lab:** DDP with gradient hooks (more efficient than manual averaging).

---

## Library

**1. `comms.py` (The Glue)**

- **Initialization:** A wrapper around `dist.init_process_group()`.
- **All-Reduce:** `all_reduce(tensor, op)` and `all_reduce_mean(tensor)` for gradient synchronization.

**2. `model.py` (The Subject)**

- **FullMLP:** A class `FullMLP(nn.Module)` with 16 `Linear` layers. Replicated on each rank.

**3. `schedule.py` (The Engine)**

- **Naive DP:** Manual gradient averaging after backward.
- **DDP:** Gradient hooks for automatic synchronization during backward.

