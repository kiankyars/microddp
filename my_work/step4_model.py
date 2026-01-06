import torch.nn as nn

# TODO: Implement FullMLP
# In Data Parallelism, every GPU has a complete copy of the model
# This is different from Pipeline Parallelism where each GPU has a slice


class FullMLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        # TODO: Build the full model
        # - depth layers of Linear(dim, dim) + ReLU
        # - Final Linear(dim, 2) for classification
        # - CrossEntropyLoss
        pass

    def forward(self, x, targets=None):
        # TODO: Forward pass
        # Return loss if targets provided, otherwise return logits
        pass

