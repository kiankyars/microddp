"""
Demo: Gradient Bucketing Comparison

Run with: torchrun --nproc-per-node=4 examples/demo_bucketing.py

This demonstrates the performance improvement from gradient bucketing.
"""

import torch
import torch.optim as optim
from src.comms import DataParallelComms, init_distributed
from src.model import FullMLP
from src.bucketing import compare_bucketed_vs_unbucketed

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16

# Initialize distributed environment
rank, world_size, device = init_distributed()
comms = DataParallelComms(rank, world_size)

if rank == 0:
    print(f"=== Gradient Bucketing Demo (World Size: {world_size}) ===\n")

# Initialize model
model = FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device)

# Split batch across ranks
chunk_size = BATCH_SIZE // world_size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size

full_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
full_target = torch.randint(0, 2, (BATCH_SIZE,))

input_chunk = full_input[start_idx:end_idx].to(device)
target_chunk = full_target[start_idx:end_idx].to(device)

# Compare bucketed vs unbucketed
unbucketed_time, bucketed_time = compare_bucketed_vs_unbucketed(
    model, comms, input_chunk, target_chunk, device
)

if rank == 0:
    print(f"\n=== Bucketing Performance Comparison ===")
    print(f"Unbucketed time: {unbucketed_time*1000:.2f} ms")
    print(f"Bucketed time: {bucketed_time*1000:.2f} ms")
    print(f"Speedup: {unbucketed_time/bucketed_time:.2f}x")

torch.distributed.destroy_process_group()
