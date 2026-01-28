"""
Gradient hooks for distributed training.

Hooks allow all-reduce to happen during backward pass rather than after,
enabling overlap of communication and computation.
"""

from src.bucketing import BucketedDDPHooks


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
                            comms.allreducemean(grad)
                        return grad

                    return hook

                param.register_hook(make_hook(param))
        return None