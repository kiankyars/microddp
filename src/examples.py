"""
Examples and performance comparisons for MicroDDP components.

Run with: torchrun --nproc-per-node=4 src/examples.py
"""

import time
import torch
import torch.distributed as dist

from src.comms import init_distributed, cleanup
from src.model import FullMLP
from src.allreduce import allreduce1, allreduce2, allreduce3, allreduce4
from src.optimisations import register_bucketed_hooks


def allreducemean(tensor):
    """
    In-place all-reduce (SUM) followed by division by world_size.
    """
    world_size = dist.get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)


def compare_allreduce_algorithms(rank, world_size, device, tensor_size=1000, num_iterations=10):
    """
    Compare performance of different all-reduce algorithms.
    """
    tensor = torch.randn(tensor_size, device=device)
    
    # Warmup
    for _ in range(3):
        t = tensor.clone()
        allreduce1(t)
        dist.barrier()
    
    # Benchmark allreduce1 (reduce + broadcast)
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        t = tensor.clone()
        allreduce1(t)
        dist.barrier()
    allreduce1_time = (time.time() - start) / num_iterations
    
    # Benchmark allreduce2 (manual send/recv)
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        _ = allreduce2(rank, tensor.clone())
        dist.barrier()
    allreduce2_time = (time.time() - start) / num_iterations
    
    # Benchmark allreduce3 (reduce_scatter + all_gather)
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        _ = allreduce3(tensor.clone())
        dist.barrier()
    allreduce3_time = (time.time() - start) / num_iterations
    
    # Benchmark allreduce4 (ring)
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        send = tensor.clone()
        recv = torch.zeros_like(tensor)
        allreduce4(send, recv)
        dist.barrier()
    allreduce4_time = (time.time() - start) / num_iterations
    
    # Benchmark PyTorch's optimized all-reduce
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        result_pytorch = tensor.clone()
        dist.all_reduce(result_pytorch)
        dist.barrier()
    pytorch_time = (time.time() - start) / num_iterations
    
    if rank == 0:
        print(f"\n=== All-Reduce Performance Comparison ===")
        print(f"Tensor size: {tensor_size}")
        print(f"World size: {world_size}")
        print(f"allreduce1 (reduce+broadcast): {allreduce1_time*1000:.2f} ms")
        print(f"allreduce2 (manual send/recv): {allreduce2_time*1000:.2f} ms")
        print(f"allreduce3 (scatter+gather):   {allreduce3_time*1000:.2f} ms")
        print(f"allreduce4 (ring):             {allreduce4_time*1000:.2f} ms")
        print(f"PyTorch All-Reduce:            {pytorch_time*1000:.2f} ms")


def compare_hook_vs_manual_timing(model, input_chunk, target_chunk, device):
    """
    Compare timing of hook-based vs manual all-reduce.
    """
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
            allreducemean(param.grad)
    
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
                        allreducemean(grad)
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
    
    if dist.get_rank() == 0:
        print("\n=== Hook vs Manual All-Reduce Timing ===")
        print(f"Manual (sequential): {manual_time*1000:.2f} ms")
        print(f"Hook-based: {hook_time*1000:.2f} ms")
        print(f"Speedup: {manual_time/hook_time:.2f}x")


def compare_bucketed_vs_unbucketed(model, input_chunk, target_chunk, device):
    """
    Compare performance of bucketed vs unbucketed gradient synchronization.
    """
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
                allreducemean(param.grad)
        optimizer1.step()
    unbucketed_time = time.time() - start
    
    # Test bucketed
    model_copy2 = type(model)(model.net[0].in_features, len(model.net) // 2).to(device)
    optimizer2 = torch.optim.Adam(model_copy2.parameters())
    register_bucketed_hooks(model_copy2, bucket_size_mb=25.0)
    
    start = time.time()
    for _ in range(10):
        optimizer2.zero_grad()
        loss = model_copy2(input_chunk, target_chunk)
        loss.backward()
        optimizer2.step()
    bucketed_time = time.time() - start
    
    if dist.get_rank() == 0:
        print(f"\n=== Bucketing Performance Comparison ===")
        print(f"Unbucketed time: {unbucketed_time*1000:.2f} ms")
        print(f"Bucketed time: {bucketed_time*1000:.2f} ms")
        print(f"Speedup: {unbucketed_time/bucketed_time:.2f}x")


def main():
    """
    Run all performance comparisons.
    """
    rank, world_size, device = init_distributed()
    
    if rank == 0:
        print(f"=== Examples ===")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    # All-reduce algorithm comparison
    compare_allreduce_algorithms(rank, world_size, device, tensor_size=100000, num_iterations=20)
    
    # Hook vs manual comparison
    model = FullMLP(128, 16).to(device)
    chunk_size = 32 // world_size
    input_chunk = torch.randn(chunk_size, 128, device=device)
    target_chunk = torch.randint(0, 2, (chunk_size,), device=device)
    
    compare_hook_vs_manual_timing(model, input_chunk, target_chunk, device)
    
    # Bucketed vs unbucketed comparison
    compare_bucketed_vs_unbucketed(model, input_chunk, target_chunk, device)
    
    cleanup()


if __name__ == "__main__":
    main()
