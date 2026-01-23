"""
Gradient Bucketing for DDP

Bucketing groups small gradients into larger messages for more efficient
communication. This reduces the number of all-reduce calls and improves
bandwidth utilization.
"""

import torch
import torch.distributed as dist
from collections import defaultdict
from typing import List, Callable


class GradientBucket:
    """
    A bucket that groups multiple parameter gradients together.
    """
    
    def __init__(self, params: List[torch.nn.Parameter], bucket_size: int):
        self.params = params
        self.bucket_size = bucket_size
        self.gradients = []
        self.ready = False
        
    def add_gradient(self, grad: torch.Tensor):
        """Add a gradient to this bucket."""
        self.gradients.append(grad)
        if len(self.gradients) >= len(self.params):
            self.ready = True
    
    def all_reduce(self, comms, op=dist.ReduceOp.SUM):
        """
        Perform all-reduce on the concatenated gradients in this bucket.
        """
        if not self.gradients:
            return
        
        # Concatenate all gradients in the bucket
        flat_grads = torch.cat([g.flatten() for g in self.gradients])
        
        # All-reduce the concatenated tensor
        comms.all_reduce(flat_grads, op=op)
        
        # Split back and update original gradients
        offset = 0
        for i, grad in enumerate(self.gradients):
            grad_size = grad.numel()
            grad.copy_(flat_grads[offset:offset+grad_size].reshape(grad.shape))
            offset += grad_size


class BucketedDDPHooks:
    """
    DDP-style gradient hooks with bucketing for efficient communication.
    
    Key idea: Instead of all-reducing each gradient separately, group them
    into buckets and all-reduce the buckets. This:
    1. Reduces communication overhead (fewer all-reduce calls)
    2. Improves bandwidth utilization (larger messages)
    3. Enables overlap (can start reducing bucket while computing next gradients)
    """
    
    def __init__(self, model: torch.nn.Module, comms, bucket_size_mb: float = 25.0):
        self.model = model
        self.comms = comms
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets: List[GradientBucket] = []
        self.param_to_bucket = {}
        self._create_buckets()
        self._register_hooks()
    
    def _create_buckets(self):
        """
        Create buckets by grouping parameters based on their size.
        Parameters are grouped in reverse order (last layer first) to enable
        overlap: we can start reducing early layers while later layers are still
        computing gradients.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Sort parameters in reverse order (for overlap optimization)
        params = list(reversed(params))
        
        current_bucket_params = []
        current_bucket_size = 0
        
        for param in params:
            param_size_bytes = param.numel() * param.element_size()
            
            # If adding this param would exceed bucket size, finalize current bucket
            if current_bucket_params and current_bucket_size + param_size_bytes > self.bucket_size_bytes:
                bucket = GradientBucket(current_bucket_params, current_bucket_size)
                self.buckets.append(bucket)
                for p in current_bucket_params:
                    self.param_to_bucket[p] = bucket
                current_bucket_params = []
                current_bucket_size = 0
            
            current_bucket_params.append(param)
            current_bucket_size += param_size_bytes
        
        # Add final bucket
        if current_bucket_params:
            bucket = GradientBucket(current_bucket_params, current_bucket_size)
            self.buckets.append(bucket)
            for p in current_bucket_params:
                self.param_to_bucket[p] = bucket
    
    def _register_hooks(self):
        """
        Register hooks on each parameter that will all-reduce gradients
        when they become available.
        """
        self.pending_buckets = set()
        
        def make_hook(param):
            def hook(grad):
                if grad is None:
                    return grad
                
                bucket = self.param_to_bucket.get(param)
                if bucket is None:
                    # Fallback: all-reduce immediately if not in a bucket
                    self.comms.all_reduce_mean(grad)
                    return grad
                
                # Add gradient to bucket
                bucket.add_gradient(grad)
                
                # If bucket is ready, all-reduce it
                if bucket.ready and bucket not in self.pending_buckets:
                    self.pending_buckets.add(bucket)
                    bucket.all_reduce(self.comms, op=dist.ReduceOp.SUM)
                    # Average the gradients
                    for grad in bucket.gradients:
                        grad.div_(self.comms.world_size)
                    self.pending_buckets.remove(bucket)
                
                return grad
            
            return hook
        
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(make_hook(param))
    
    def get_bucket_info(self):
        """Return information about buckets for debugging/analysis."""
        return {
            'num_buckets': len(self.buckets),
            'bucket_sizes': [len(b.params) for b in self.buckets],
            'bucket_sizes_mb': [b.bucket_size / (1024 * 1024) for b in self.buckets]
        }


def compare_bucketed_vs_unbucketed(model, comms, input_chunk, target_chunk, device):
    """
    Compare performance of bucketed vs unbucketed gradient synchronization.
    """
    import time
    
    # Test unbucketed (one all-reduce per parameter)
    model_copy1 = type(model)(model.net[0].in_features, len(model.net) // 2).to(device)
    optimizer1 = torch.optim.Adam(model_copy1.parameters())
    
    start = time.time()
    for _ in range(10):
        optimizer1.zero_grad()
        loss = model_copy1(input_chunk, target_chunk)
        loss.backward()
        # Unbucketed: all-reduce each gradient separately
        for param in model_copy1.parameters():
            if param.grad is not None:
                comms.all_reduce_mean(param.grad)
        optimizer1.step()
    unbucketed_time = time.time() - start
    
    # Test bucketed
    model_copy2 = type(model)(model.net[0].in_features, len(model.net) // 2).to(device)
    optimizer2 = torch.optim.Adam(model_copy2.parameters())
    bucketed_hooks = BucketedDDPHooks(model_copy2, comms, bucket_size_mb=25.0)
    
    start = time.time()
    for _ in range(10):
        optimizer2.zero_grad()
        loss = model_copy2(input_chunk, target_chunk)
        loss.backward()
        optimizer2.step()
    bucketed_time = time.time() - start
    
    return unbucketed_time, bucketed_time
