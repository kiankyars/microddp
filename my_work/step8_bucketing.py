"""
Step 8: Gradient Bucketing

This exercise demonstrates how bucketing improves communication efficiency.
Run with: torchrun --nproc-per-node=4 my_work/step8_bucketing.py
"""

import torch
import torch.optim as optim
import torch.distributed as dist

from src.comms import DataParallelComms, init_distributed
from src.model import FullMLP
from src.bucketing import compare_bucketed_vs_unbucketed

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16

# TODO: Initialize distributed environment
rank, world_size, device = None, None, None  # TODO: init_distributed()
comms = None  # TODO: DataParallelComms(rank, world_size)

if rank == 0:
    print(f"=== Gradient Bucketing Comparison (World Size: {world_size}) ===\n")

# TODO: Initialize model
model = None  # TODO: FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device)

# TODO: Split batch across ranks
full_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
full_target = torch.randint(0, 2, (BATCH_SIZE,))

chunk_size = BATCH_SIZE // world_size if world_size else BATCH_SIZE
start_idx = (rank * chunk_size) if rank is not None else 0
end_idx = start_idx + chunk_size

input_chunk = None  # TODO: full_input[start_idx:end_idx].to(device)
target_chunk = None  # TODO: full_target[start_idx:end_idx].to(device)

# TODO: Call compare function to see bucketed vs unbucketed performance
# compare_bucketed_vs_unbucketed(model, comms, input_chunk, target_chunk, device)

# TODO: Clean up
# dist.destroy_process_group()
