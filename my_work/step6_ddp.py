import time

import torch
import torch.optim as optim

from src.comms import DataParallelComms, init_distributed
from src.model import FullMLP
from src.schedule import register_ddp_hooks

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# Setup distributed environment
rank, world_size, device = init_distributed()
comms = DataParallelComms(rank, world_size)

if rank == 0:
    print(f"--- Starting Micro DDP with Hooks on {world_size} Processes ({device}) ---")

# Initialize model and optimizer
model = FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TODO: Register gradient hooks for automatic all-reduce during backward
# register_ddp_hooks(model, comms)

# Split batch across ranks
chunk_size = BATCH_SIZE // world_size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size

full_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
full_target = torch.randint(0, 2, (BATCH_SIZE,))

input_chunk = full_input[start_idx:end_idx].to(device)
target_chunk = full_target[start_idx:end_idx].to(device)

# Training Loop
start_time = time.time()
model.train()
for step in range(STEPS):
    optimizer.zero_grad()

    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass (gradients automatically all-reduced via hooks)
    loss.backward()

    # Optimizer step
    optimizer.step()

    if rank == 0 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

if rank == 0:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")

torch.distributed.destroy_process_group()
