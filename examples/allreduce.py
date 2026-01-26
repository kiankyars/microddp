"""
All-Reduce Algorithms from First Principles

This module implements all-reduce algorithms without using PyTorch's
distributed primitives, to understand the underlying communication patterns.

Key Concepts:
- Naive All-Reduce: O(n²) communication complexity
- Ring All-Reduce: O(n) communication complexity, optimal bandwidth usage
"""

import torch
import torch.distributed as dist
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AllReduceAlgorithms:
    """
    Educational implementations of all-reduce algorithms.
    These demonstrate the communication patterns used in DDP.
    """

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def naive_all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        """
        Naive All-Reduce: O(n²) communication complexity
        
        Algorithm:
        1. Each rank sends its tensor to rank 0
        2. Rank 0 sums all tensors
        3. Rank 0 broadcasts result to all ranks
        
        Communication steps: 2*(n-1) = O(n)
        But each step involves all ranks, so total messages: O(n²)
        """
        result = tensor.clone()
        
        # Phase 1: Gather to rank 0
        if self.rank == 0:
            # Rank 0 receives from all other ranks
            for src_rank in range(1, self.world_size):
                recv_tensor = torch.zeros_like(tensor)
                dist.recv(recv_tensor, src=src_rank)
                if op == dist.ReduceOp.SUM:
                    result += recv_tensor
        else:
            # Other ranks send to rank 0
            dist.send(tensor, dst=0)
        
        # Phase 2: Broadcast from rank 0
        dist.broadcast(result, src=0)
        
        return result

    def ring_all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        """
        Simplified Ring All-Reduce for educational purposes.
        Works on the full tensor (not chunked) to make the pattern clearer.
        
        This is less bandwidth-efficient but easier to understand.
        
        Algorithm: For full-tensor, we need to ensure each rank receives
        the original data from each other rank exactly once. We do this by
        having each rank send its original data (not accumulated) in each step.
        """
        result = tensor.clone()
        original = tensor.clone()  # Keep original to send in each step
        
        # Phase 1: Scatter-Reduce
        # Each rank sends its original data around the ring
        # After (world_size - 1) steps, each rank will have received data from all other ranks
        for step in range(self.world_size - 1):
            send_to = (self.rank + 1) % self.world_size
            recv_from = (self.rank - 1) % self.world_size
            
            # Use blocking operations with proper ordering to avoid deadlock
            # Even ranks send first, odd ranks receive first
            if self.rank % 2 == 0:
                # Even ranks: send then receive
                dist.send(original, dst=send_to)  # Send original, not accumulated
                recv_tensor = torch.zeros_like(result)
                dist.recv(recv_tensor, src=recv_from)
            else:
                # Odd ranks: receive then send
                recv_tensor = torch.zeros_like(result)
                dist.recv(recv_tensor, src=recv_from)
                dist.send(original, dst=send_to)  # Send original, not accumulated
            
            # Accumulate received data (SUM operation)
            result += recv_tensor
            
            # Rotate: the data we received becomes the original we'll send next
            # (This simulates the ring movement)
            original = recv_tensor.clone()
        
        return result


def compare_all_reduce_algorithms(rank, world_size, tensor_size=1000, num_iterations=10):
    """
    Compare performance of different all-reduce algorithms.
    """
    import time
    
    algorithms = AllReduceAlgorithms(rank, world_size)
    tensor = torch.randn(tensor_size, device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Warmup
    for _ in range(3):
        _ = algorithms.naive_all_reduce(tensor.clone())
        _ = algorithms.ring_all_reduce(tensor.clone())
        dist.barrier()
    
    # Benchmark naive all-reduce
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        _ = algorithms.naive_all_reduce(tensor.clone())
        dist.barrier()
    naive_time = (time.time() - start) / num_iterations
    
    # Benchmark ring all-reduce
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        _ = algorithms.ring_all_reduce(tensor.clone())
        dist.barrier()
    ring_time = (time.time() - start) / num_iterations
    
    # Benchmark PyTorch's optimized all-reduce
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        result_pytorch = tensor.clone()
        dist.all_reduce(result_pytorch, op=dist.ReduceOp.SUM)
        dist.barrier()
    pytorch_time = (time.time() - start) / num_iterations
    
    if rank == 0:
        print(f"\n=== All-Reduce Performance Comparison ===")
        print(f"Tensor size: {tensor_size}")
        print(f"World size: {world_size}")
        print(f"Naive All-Reduce: {naive_time*1000:.2f} ms")
        print(f"Ring All-Reduce (simple): {ring_time*1000:.2f} ms")
        print(f"PyTorch All-Reduce: {pytorch_time*1000:.2f} ms")
        print(f"\nSpeedups:")
        print(f"  Ring vs Naive: {naive_time/ring_time:.2f}x")
        print(f"  PyTorch vs Naive: {naive_time/pytorch_time:.2f}x")
        print(f"  PyTorch vs Ring: {ring_time/pytorch_time:.2f}x")
    
    return naive_time, ring_time, pytorch_time


def main():
    """
    Example: Comparing All-Reduce Algorithms
    
    This script demonstrates the difference between naive and ring all-reduce.
    Run with: torchrun --nproc-per-node=4 examples/allreduce.py
    """
    import os
    from src.comms import init_distributed, DataParallelComms
    
    # Initialize distributed environment
    rank, world_size, device = init_distributed()
    comms = DataParallelComms(rank, world_size)
    
    if rank == 0:
        print(f"=== All-Reduce Algorithm Comparison ===\n")
        print(f"World size: {world_size}")
        print(f"Device: {device}\n")
    
    # Create test tensor
    tensor_size = 1000
    tensor = torch.randn(tensor_size, device=device)
    
    # Initialize algorithms
    algorithms = AllReduceAlgorithms(rank, world_size)
    
    # Test all three methods and verify they produce the same result
    dist.barrier()
    
    # 1. Naive all-reduce (SUM)
    result_naive = algorithms.naive_all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
    dist.barrier()
    
    # 2. Ring all-reduce (SUM)
    result_ring = algorithms.ring_all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
    dist.barrier()
    
    # 3. PyTorch's optimized all-reduce (SUM)
    result_pytorch = tensor.clone()
    dist.all_reduce(result_pytorch, op=dist.ReduceOp.SUM)
    dist.barrier()
    
    # Verify all three produce the same SUM result
    if rank == 0:
        diff_naive_ring = (result_naive - result_ring).abs().max()
        diff_ring_pytorch = (result_ring - result_pytorch).abs().max()
        diff_naive_pytorch = (result_naive - result_pytorch).abs().max()
        
        print("=== Verification: All methods should produce the same SUM ===")
        print(f"Naive vs Ring difference: {diff_naive_ring.item():.6f}")
        print(f"Ring vs PyTorch difference: {diff_ring_pytorch.item():.6f}")
        print(f"Naive vs PyTorch difference: {diff_naive_pytorch.item():.6f}")
        
        # Use 1e-5 threshold for floating point precision
        all_match = (diff_naive_ring.item() < 1e-5 and 
                    diff_ring_pytorch.item() < 1e-5 and 
                    diff_naive_pytorch.item() < 1e-5)
        print(f"All results match: {all_match}\n")
    
    # Performance comparison
    if rank == 0:
        print("Running performance comparison...")
    
    compare_all_reduce_algorithms(rank, world_size, tensor_size=10000, num_iterations=20)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
