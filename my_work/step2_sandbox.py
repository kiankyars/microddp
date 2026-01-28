"""
Step 2: Sandbox for learning dist.all_reduce

This file is for experimenting with PyTorch's distributed all_reduce.
Run with: torchrun --nproc-per-node=4 my_work/step2_sandbox.py
"""

import torch
import torch.distributed as dist

from src.comms import init_distributed, cleanup

rank, world_size, device = init_distributed()

# Each rank starts with different data
# Rank 0: [1.0, 2.0, 3.0]
# Rank 1: [4.0, 5.0, 6.0]
# Rank 2: [7.0, 8.0, 9.0]
# etc.

initial_value = rank * 3 + 1
tensor = torch.tensor([float(initial_value), float(initial_value + 1), float(initial_value + 2)], 
                      device=device)

if rank == 0:
    print(f"=== All-Reduce Sandbox (World Size: {world_size}) ===\n")
    print("Initial tensors:")

dist.barrier()
print(f"Rank {rank}: {tensor.cpu().tolist()}")
dist.barrier()

# All-reduce with SUM (default operation)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

if rank == 0:
    print("\nAfter dist.all_reduce (SUM):")

dist.barrier()
print(f"Rank {rank}: {tensor.cpu().tolist()}")
dist.barrier()

# To get MEAN, divide by world_size after SUM
tensor_mean = tensor / world_size

if rank == 0:
    print("\nAfter dividing by world_size (MEAN):")

dist.barrier()
print(f"Rank {rank}: {tensor_mean.cpu().tolist()}")
dist.barrier()

cleanup()
