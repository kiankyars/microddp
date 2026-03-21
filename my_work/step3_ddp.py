import time

import torch
import torch.optim as optim

from src.comms import init_distributed, cleanup
from src.model import FullMLP
from src.optimisations import register_hooks

# 1. Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# 2. Setup distributed environment
rank, world_size, device = init_distributed()
torch.manual_seed(42)

if rank == 0:
    print(f"--- Starting Micro DDP with Hooks on {world_size} Processes ({device}) ---")

# 3. Initialize model and optimizer
model = FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TODO: Register gradient hooks for automatic all-reduce during backward
# register_hooks(model)

# 4. Split batch across ranks
full_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
full_target = torch.randint(0, 2, (BATCH_SIZE,))

chunk_size = None  # TODO
start_idx = None  # TODO
end_idx = None    # TODO

input_chunk = None    # TODO
target_chunk = None    # TODO

# 5. Training Loop
start_time = time.time()
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    loss = model(input_chunk, target_chunk)
    loss.backward()
    optimizer.step()
    if rank == 0 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

if rank == 0:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")

cleanup()
