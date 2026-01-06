"""
torchrun --nproc-per-node=4 src/main.py

In Data Parallelism:
- Each rank has a complete copy of the model
- Each rank processes a different chunk of the batch
- Gradients are averaged across all ranks after backward pass
- All ranks update with the same averaged gradients
"""

import time

import torch
import torch.optim as optim

# Import our modules
from comms import DataParallelComms, init_distributed
from model import FullMLP
from schedule import naive_data_parallel_step, register_ddp_hooks

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# 1. Setup Distributed Environment
rank, world_size, device = init_distributed()
comms = DataParallelComms(rank, world_size)

# Set base seed, then offset by rank for different data chunks
torch.manual_seed(42)
# Each rank will get a different chunk of the batch

if rank == 0:
    print(f"--- Starting Micro DDP on {world_size} Processes ({device}) ---")

# 2. Initialize the Full Model (replicated on each rank)
model = FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device)

# 3. Setup Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Data Generation
# Each rank gets a different chunk of the batch
full_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
full_target = torch.randint(0, 2, (BATCH_SIZE,))

# Split batch across ranks
chunk_size = BATCH_SIZE // world_size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size

input_chunk = full_input[start_idx:end_idx].to(device)
target_chunk = full_target[start_idx:end_idx].to(device)

# 5. Training Loop
start_time = time.time()
model.train()
for step in range(STEPS):
    optimizer.zero_grad()

    # Data parallel step: forward, backward, all-reduce gradients
    loss = naive_data_parallel_step(model, comms, input_chunk, target_chunk, device)

    # Optimizer step (all ranks have same averaged gradients)
    optimizer.step()

    # Logging
    if rank == 0 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

# Clean up
if rank == 0:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")

torch.distributed.destroy_process_group()

