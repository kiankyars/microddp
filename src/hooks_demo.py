"""
Demonstration of Gradient Hook Execution Order

This module shows how gradient hooks work in DDP and why they enable
overlap between computation and communication.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List
from comms import DataParallelComms


class HookTracker:
    """
    Tracks when gradient hooks are called during backward pass.
    """
    
    def __init__(self):
        self.hook_calls = []
        self.param_order = []
    
    def make_hook(self, param_name: str, param_idx: int):
        """Create a hook that tracks when it's called."""
        def hook(grad):
            import time
            call_time = time.time()
            self.hook_calls.append({
                'param': param_name,
                'idx': param_idx,
                'time': call_time,
                'grad_shape': grad.shape if grad is not None else None
            })
            return grad
        return hook


def demonstrate_hook_order(model: nn.Module, comms: DataParallelComms):
    """
    Demonstrate the order in which gradient hooks are called.
    
    Key insight: Hooks are called in REVERSE order of forward pass.
    This enables overlap: we can start reducing early-layer gradients
    while later layers are still computing.
    """
    tracker = HookTracker()
    
    # Register hooks on all parameters
    param_names = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_names.append(name)
            hook = tracker.make_hook(name, i)
            param.register_hook(hook)
    
    # Create dummy input
    input_tensor = torch.randn(4, 128)
    target = torch.randint(0, 2, (4,))
    
    # Forward pass
    loss = model(input_tensor, target)
    
    # Backward pass - hooks will be called here
    loss.backward()
    
    # Print hook execution order
    print("=== Gradient Hook Execution Order ===\n")
    print("Hooks are called in REVERSE order of forward pass:")
    print("(Last layer first, first layer last)\n")
    
    for i, call in enumerate(tracker.hook_calls):
        print(f"{i+1}. {call['param']} (shape: {call['grad_shape']})")
    
    print("\nWhy this matters:")
    print("- We can start all-reducing early layers while later layers compute")
    print("- This enables computation/communication overlap")
    print("- Bucketing groups these hooks for efficiency")


def demonstrate_overlap(model: nn.Module, comms: DataParallelComms, 
                        input_chunk, target_chunk, device):
    """
    Demonstrate how hooks enable overlap between computation and communication.
    """
    import time
    import threading
    
    # Track when gradients are computed vs when they're communicated
    grad_ready_times = {}
    comm_start_times = {}
    comm_end_times = {}
    
    def make_hook_with_timing(param_name):
        def hook(grad):
            if grad is not None:
                import time
                grad_ready_times[param_name] = time.time()
                
                # Simulate communication (all-reduce)
                comm_start_times[param_name] = time.time()
                comms.all_reduce_mean(grad)
                comm_end_times[param_name] = time.time()
            
            return grad
        
        return hook
    
    # Register timing hooks
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook_with_timing(name))
    
    import torch.distributed as dist
    
    # Forward pass
    dist.barrier()
    start_forward = time.time()
    loss = model(input_chunk, target_chunk)
    end_forward = time.time()
    dist.barrier()
    
    # Backward pass (hooks called here)
    dist.barrier()
    start_backward = time.time()
    loss.backward()
    end_backward = time.time()
    dist.barrier()
    
    # Analyze overlap
    print("=== Computation/Communication Overlap Analysis ===\n")
    print(f"Forward time: {(end_forward - start_forward)*1000:.2f} ms")
    print(f"Backward time: {(end_backward - start_backward)*1000:.2f} ms")
    print()
    
    if grad_ready_times:
        first_grad_time = min(grad_ready_times.values())
        last_grad_time = max(grad_ready_times.values())
        last_comm_end = max(comm_end_times.values()) if comm_end_times else last_grad_time
        
        computation_window = last_grad_time - first_grad_time
        total_window = last_comm_end - first_grad_time
        
        overlap_ratio = computation_window / total_window if total_window > 0 else 0
        
        print(f"First gradient ready: {first_grad_time - start_backward:.4f}s after backward start")
        print(f"Last gradient ready: {last_grad_time - start_backward:.4f}s after backward start")
        print(f"Last communication done: {last_comm_end - start_backward:.4f}s after backward start")
        print(f"Overlap ratio: {overlap_ratio:.2%}")
        print()
        print("With bucketing, we can start communicating early gradients")
        print("while later gradients are still being computed!")


def compare_hook_vs_manual_timing(model, comms, input_chunk, target_chunk, device):
    """
    Compare timing of hook-based vs manual all-reduce.
    """
    import time
    
    import torch.distributed as dist
    
    # Method 1: Manual all-reduce (after backward)
    model1 = type(model)(model.net[0].in_features, len(model.net) // 2).to(device)
    optimizer1 = torch.optim.Adam(model1.parameters())
    
    dist.barrier()
    start1 = time.time()
    
    loss1 = model1(input_chunk, target_chunk)
    loss1.backward()
    
    # Manual all-reduce (sequential, no overlap)
    for param in model1.parameters():
        if param.grad is not None:
            comms.all_reduce_mean(param.grad)
    
    optimizer1.step()
    dist.barrier()
    manual_time = time.time() - start1
    
    # Method 2: Hook-based (potential overlap)
    model2 = type(model)(model.net[0].in_features, len(model.net) // 2).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters())
    
    # Register hooks
    for param in model2.parameters():
        if param.requires_grad:
            def make_hook():
                def hook(grad):
                    if grad is not None:
                        comms.all_reduce_mean(grad)
                    return grad
                return hook
            param.register_hook(make_hook())
    
    dist.barrier()
    start2 = time.time()
    
    loss2 = model2(input_chunk, target_chunk)
    loss2.backward()  # Hooks called during backward
    optimizer2.step()
    
    dist.barrier()
    hook_time = time.time() - start2
    
    if comms.rank == 0:
        print("=== Hook vs Manual All-Reduce Timing ===\n")
        print(f"Manual (sequential): {manual_time*1000:.2f} ms")
        print(f"Hook-based: {hook_time*1000:.2f} ms")
        print(f"Speedup: {manual_time/hook_time:.2f}x")
        print()
        print("Note: Actual speedup depends on:")
        print("- Model architecture (layer sizes)")
        print("- Network bandwidth")
        print("- Whether communication overlaps with computation")
