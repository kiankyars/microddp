"""
Step 2: Sandbox for dist.all_reduce

This file is for experimenting with PyTorch's distributed primitives.
Run with: torchrun --nproc-per-node=4 my_work/step2_sandbox.py
"""

import torch
import torch.distributed as dist

from src.comms import init_distributed, cleanup

rank, world_size, device = init_distributed()

'''
=============================================================================
Exercise 1: Basic all_reduce with SUM&AVG
=============================================================================
'''
# Each rank starts with different data
# Rank 0: [1.0, 2.0, 3.0], Rank 1: [4.0, 5.0, 6.0], etc.

initial_value = rank * 3 + 1
tensor = torch.tensor([float(initial_value + i) for i in range(3)], 
                      device=device)

if rank == 0:
    print(f"=== Exercise 1: all_reduce (World Size: {world_size}) ===")
    print("Initial tensors:")

print(f"Rank {rank}: {tensor.tolist()}")

# TODO: all_reduce

if rank == 0:
    print("After all_reduce:")
print(f"Rank {rank}: {tensor.tolist()}")

'''
=============================================================================
Exercise 2: broadcast - one rank sends to all
=============================================================================
'''

tensor2 = torch.tensor([42.0] if rank == 0 else [0.0], device=device)

if rank == 0:
    print(f"\n=== Exercise 2: broadcast ===")
    print("Before broadcast:")
print(f"Rank {rank}: {tensor2.tolist()}")

# TODO: broadcast from rank 0 to all ranks

if rank == 0:
    print("After broadcast from rank 0:")
print(f"Rank {rank}: {tensor2.tolist()}")  # All ranks should have [42.0]

'''
=============================================================================
Exercise 3: reduce - all ranks send to one
=============================================================================
'''

tensor3 = torch.tensor([float(rank + 1)], device=device)

if rank == 0:
    print(f"\n=== Exercise 3: reduce to rank 0 ===")
    print("Before reduce:")
print(f"Rank {rank}: {tensor3.tolist()}")

# TODO: reduce (SUM) to rank 0

if rank == 0:
    print("After reduce to rank 0:")
print(f"Rank {rank}: {tensor3.tolist()}")  # Only rank 0 has the sum

cleanup()
