import os
import torch
import torch.distributed as dist

from src.comms import DataParallelComms, init_distributed

# TODO: Initialize distributed environment
rank, world_size, device = None, None, None  # TODO: init_distributed()
comms = None  # TODO: DataParallelComms(rank, world_size)

# Each rank starts with different data
# Rank 0: [1.0, 2.0, 3.0]
# Rank 1: [4.0, 5.0, 6.0]
# Rank 2: [7.0, 8.0, 9.0]
# After all-reduce_mean: all ranks should have the average

initial_value = rank * 3 + 1 if rank is not None else 0
tensor = torch.tensor([float(initial_value), float(initial_value + 1), float(initial_value + 2)], 
                     device=device if device else "cpu")

if rank == 0:
    print(f"=== All-Reduce Test (World Size: {world_size}) ===\n")
    print("Initial tensors:")

# TODO: Print initial tensor for each rank
# dist.barrier()
# print(f"Rank {rank}: {tensor.cpu().tolist()}")
# dist.barrier()

# TODO: Use comms.all_reduce_mean() to average the tensor across all ranks
# comms.all_reduce_mean(tensor)

if rank == 0:
    print("\nAfter all-reduce_mean:")

# TODO: Print result and verify all ranks have the same averaged tensor
# dist.barrier()
# print(f"Rank {rank}: {tensor.cpu().tolist()}")
# dist.barrier()

# TODO: Clean up
# dist.destroy_process_group()

