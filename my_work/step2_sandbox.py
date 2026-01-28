"""
Step 2: Sandbox for learning dist.all_reduce

This file is for experimenting with PyTorch's distributed primitives.
Run with: PYTHONPATH=. torchrun --nproc-per-node=4 my_work/step2_sandbox.py
"""

import torch
import torch.distributed as dist

from src.comms import init_distributed, cleanup

rank, world_size, device = init_distributed()

# =============================================================================
# Exercise 1: Basic all_reduce with SUM
# =============================================================================
# Each rank starts with different data
# Rank 0: [1.0, 2.0, 3.0], Rank 1: [4.0, 5.0, 6.0], etc.

initial_value = rank * 3 + 1
tensor = torch.tensor([float(initial_value), float(initial_value + 1), float(initial_value + 2)], 
                      device=device)

if rank == 0:
    print(f"=== Exercise 1: all_reduce SUM (World Size: {world_size}) ===")
    print("Initial tensors:")

dist.barrier()
print(f"  Rank {rank}: {tensor.cpu().tolist()}")
dist.barrier()

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

if rank == 0:
    print("After all_reduce SUM:")
dist.barrier()
print(f"  Rank {rank}: {tensor.cpu().tolist()}")
dist.barrier()

# =============================================================================
# Exercise 2: Implement MEAN using SUM
# =============================================================================
# TODO: Create a new tensor and compute the mean across all ranks
# Hint: MEAN = SUM / world_size

tensor2 = torch.tensor([float(rank + 1)], device=device)  # Rank 0: [1], Rank 1: [2], etc.

if rank == 0:
    print(f"\n=== Exercise 2: Compute MEAN ===")
    print("Initial tensors:")
dist.barrier()
print(f"  Rank {rank}: {tensor2.cpu().tolist()}")
dist.barrier()

# TODO: all_reduce and divide to get mean
dist.all_reduce(tensor2, op=dist.ReduceOp.SUM)
tensor2 = tensor2 / world_size

if rank == 0:
    print("After MEAN:")
dist.barrier()
print(f"  Rank {rank}: {tensor2.cpu().tolist()}")  # Should be 2.5 for 4 ranks
dist.barrier()

# =============================================================================
# Exercise 3: broadcast - one rank sends to all
# =============================================================================
# Only rank 0 has the "secret" value, broadcast it to everyone

tensor3 = torch.tensor([42.0] if rank == 0 else [0.0], device=device)

if rank == 0:
    print(f"\n=== Exercise 3: broadcast ===")
    print("Before broadcast:")
dist.barrier()
print(f"  Rank {rank}: {tensor3.cpu().tolist()}")
dist.barrier()

# TODO: broadcast from rank 0 to all ranks
dist.broadcast(tensor3, src=0)

if rank == 0:
    print("After broadcast from rank 0:")
dist.barrier()
print(f"  Rank {rank}: {tensor3.cpu().tolist()}")  # All ranks should have [42.0]
dist.barrier()

# =============================================================================
# Exercise 4: reduce - all ranks send to one
# =============================================================================
# All ranks contribute, only rank 0 gets the result

tensor4 = torch.tensor([float(rank + 1)], device=device)

if rank == 0:
    print(f"\n=== Exercise 4: reduce to rank 0 ===")
    print("Before reduce:")
dist.barrier()
print(f"  Rank {rank}: {tensor4.cpu().tolist()}")
dist.barrier()

# TODO: reduce (SUM) to rank 0
dist.reduce(tensor4, dst=0, op=dist.ReduceOp.SUM)

if rank == 0:
    print("After reduce to rank 0:")
dist.barrier()
print(f"  Rank {rank}: {tensor4.cpu().tolist()}")  # Only rank 0 has the sum
dist.barrier()

cleanup()
