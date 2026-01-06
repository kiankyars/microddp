import time

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50


class FullMLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        logits = self.net(x)
        return self.loss_fn(logits, targets)


# 2. Setup
torch.manual_seed(42)

# TODO: Simulate 2 GPUs manually
device1 = None  # TODO
device2 = None  # TODO

model1 = FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device1)
model2 = FullMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device2)

optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

# TODO: Split batch across "GPUs"
fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

input1 = None  # TODO
target1 = None  # TODO

input2 = None  # TODO
target2 = None  # TODO

# 3. Training Loop
print("--- Training Manual Data Parallel (Bridge to Distributed) ---")
start_time = time.time()
model1.train()
model2.train()

for step in range(STEPS):
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    # Forward pass on both "GPUs"
    loss1 = model1(input1, target1)
    loss2 = model2(input2, target2)

    # Backward pass
    loss1.backward()
    loss2.backward()

    # TODO: Manual gradient averaging
    # Average gradients across models

    optimizer1.step()
    optimizer2.step()

    # TODO: Sync model weights (optional, but ensures consistency)

    if step % 5 == 0:
        avg_loss = (loss1.item() + loss2.item()) / 2.0
        print(f"Step {step:02d} | Loss: {avg_loss:.6f}")

duration = time.time() - start_time
final_loss = (loss1.item() + loss2.item()) / 2.0
print(f"Final Loss: {final_loss:.6f} | Time: {duration:.3f}s")

