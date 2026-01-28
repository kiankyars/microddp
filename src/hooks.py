"""
Gradient hooks for distributed training.

Hooks allow all-reduce to happen during backward pass rather than after,
enabling overlap of communication and computation.
"""

import torch
import torch.distributed as dist


def register_allreduce_hooks(model, comms):
    """
    Register gradient hooks on model parameters that all-reduce gradients
    as they become available during backward pass.
    """
    for param in model.parameters():
        if param.requires_grad:
            def make_hook():
                def hook(grad):
                    if grad is not None:
                        comms.allreducemean(grad)
                    return grad
                return hook
            param.register_hook(make_hook())
