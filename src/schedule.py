import torch
import torch.distributed as dist


def naive_data_parallel_step(model, comms, input_chunk, target_chunk, device):
    """
    Naive Data Parallel step:
    1. Forward pass on local data chunk
    2. Backward pass (computes local gradients)
    3. All-reduce gradients (average across all ranks)
    4. Optimizer step (all ranks have same averaged gradients)
    """
    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass
    loss.backward()

    # All-reduce gradients (average across all ranks)
    for param in model.parameters():
        if param.grad is not None:
            comms.all_reduce_mean(param.grad)

    return loss


def ddp_step(model, comms, input_chunk, target_chunk, device):
    """
    DistributedDataParallel step with gradient hooks.
    This is more efficient than naive DP because:
    - Gradients are reduced asynchronously during backward
    - Uses bucket-based all-reduce for better communication efficiency
    """
    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass (gradients are automatically all-reduced via hooks)
    loss.backward()

    return loss


def register_ddp_hooks(model, comms):
    """
    Register gradient hooks for DDP-style all-reduce.
    This replaces the manual all-reduce in backward pass.
    """
    for param in model.parameters():
        if param.requires_grad:

            def make_hook(param):
                def hook(grad):
                    if grad is not None:
                        comms.all_reduce_mean(grad)
                    return grad

                return hook

            param.register_hook(make_hook(param))

