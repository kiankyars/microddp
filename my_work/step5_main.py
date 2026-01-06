import time

import torch
import torch.optim as optim

from src.comms import DataParallelComms, init_distributed
from src.model import FullMLP

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# TODO: Setup distributed environment
rank, world_size, device = None, None, None  # TODO: init_distributed()
comms = None  # TODO: DataParallelComms(rank, world_size)

if rank == 0:
    print(f"--- Starting Micro DDP on {world_size} Processes ({device}) ---")

# TODO: Initialize model and optimizer
model = None  # TODO
optimizer = None  # TODO

# TODO: Split batch across ranks
# Each rank should get a different chunk of the batch
full_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
full_target = torch.randint(0, 2, (BATCH_SIZE,))

input_chunk = None  # TODO
target_chunk = None  # TODO

# Training Loop
# TODO: Uncomment and fix once you've implemented the above
# start_time = time.time()
# model.train()
# for step in range(STEPS):
#     optimizer.zero_grad()
#
#     # TODO: Forward pass
#     loss = model(input_chunk, target_chunk)
#
#     # TODO: Backward pass
#     loss.backward()
#
#     # TODO: All-reduce gradients (average across all ranks)
#     for param in model.parameters():
#         if param.grad is not None:
#             comms.all_reduce_mean(param.grad)
#
#     # TODO: Optimizer step
#     optimizer.step()
#
#     if rank == 0 and step % 5 == 0:
#         print(f"Step {step:02d} | Loss: {loss.item():.6f}")
#
# if rank == 0:
#     print("--- Training Complete ---")
#     duration = time.time() - start_time
#     print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")
#
# torch.distributed.destroy_process_group()

