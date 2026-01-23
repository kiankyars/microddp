import torch
import torch.distributed as dist
from bucketing import BucketedDDPHooks


def naive_data_parallel_step(model, comms, input_chunk, target_chunk, device):
    """
    Naive Data Parallel step:
    1. Forward pass on local data chunk
    2. Backward pass (computes local gradients)
    3. All-reduce gradients (average across all ranks)
    4. Optimizer step (all ranks have same averaged gradients)
    
    This is the simplest form of data parallelism but inefficient because:
    - Communication happens sequentially after all gradients are computed
    - No overlap between computation and communication
    - Each gradient is all-reduced separately (many small messages)
    """
    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass
    loss.backward()

    # All-reduce gradients (average across all ranks)
    # This happens AFTER all gradients are computed (no overlap)
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
    - Enables computation/communication overlap
    """
    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass (gradients are automatically all-reduced via hooks)
    # Hooks are called as soon as each gradient is ready, enabling overlap
    loss.backward()

    return loss


def register_ddp_hooks(model, comms, use_bucketing=True, bucket_size_mb=25.0):
    """
    Register gradient hooks for DDP-style all-reduce.
    This replaces the manual all-reduce in backward pass.
    
    Args:
        model: Model to register hooks on
        comms: Communication primitives
        use_bucketing: Whether to use gradient bucketing (default: True)
        bucket_size_mb: Bucket size in MB (default: 25.0)
    
    Returns:
        BucketedDDPHooks instance if use_bucketing=True, None otherwise
    """
    if use_bucketing:
        # Use bucketed hooks for better efficiency
        return BucketedDDPHooks(model, comms, bucket_size_mb=bucket_size_mb)
    else:
        # Simple hooks: one all-reduce per parameter
        for param in model.parameters():
            if param.requires_grad:

                def make_hook(param):
                    def hook(grad):
                        if grad is not None:
                            comms.all_reduce_mean(grad)
                        return grad

                    return hook

                param.register_hook(make_hook(param))
        return None

