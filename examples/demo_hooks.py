"""
Demo: Gradient Hooks Comparison

Run with: torchrun --nproc-per-node=4 examples/demo_hooks.py

This demonstrates the performance difference between hook-based and manual all-reduce.
"""

import torch
import torch.optim as optim
from src.comms import DataParallelComms, init_distributed
from src.model import FullMLP
from examples.hooks import compare_hook_vs_manual_timing

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16

# Initialize distributed environment
rank, world_size, device = init_distributed()
comms = DataParallelComms(rank, world_size)

if rank == 0:
    print(f"=== Gradient Hooks Demo (World Size: {world_size}) ===\n")

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

# Compare hook vs manual timing
compare_hook_vs_manual_timing(model, comms, input_chunk, target_chunk, device)

torch.distributed.destroy_process_group()
