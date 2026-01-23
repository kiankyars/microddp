## Distributed Basics

- **Concept:** Rank, World Size, and Process Group.
  - **The Process Group:** Imagine a conference call. Before anyone can talk, they must dial in. `init_process_group` is dialing in.
  - **World Size:** The total number of people on the call (e.g., 4 GPUs).
  - **Rank:** Your unique ID badge (0, 1, 2, 3).
  - **Rank 0** is the "Boss" (usually handles logging, saving checkpoints, and data loading).
- **Concept:** What torchrun does.
  - **Process Isolation:** `torchrun` spawns completely separate Python interpreter instances. Each has its own memory space.
  - **True Parallelism:** Because these are separate processes (not threads), the OS schedules them across different physical CPU cores or GPUs.
  - **The Network Bridge:** When you call `dist.all_reduce`, the data is synchronized across all processes.