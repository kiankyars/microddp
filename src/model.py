import torch.nn as nn


class FullMLP(nn.Module):
    """
    Full MLP model replicated on each rank.
    In Data Parallelism, every GPU has a complete copy of the model.
    """

    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        logits = self.net(x)
        if targets is not None:
            return self.loss_fn(logits, targets)
        return logits

