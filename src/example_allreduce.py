"""
Example: Comparing All-Reduce Algorithms

This script demonstrates the difference between naive and ring all-reduce.
Run with: torchrun --nproc-per-node=4 src/example_allreduce.py
"""

import torch
import torch.distributed as dist
from comms import init_distributed, DataParallelComms
from allreduce import AllReduceAlgorithms, compare_all_reduce_algorithms


def main():
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
    
    # Test ring all-reduce
    comms.barrier()
    result_ring = algorithms.ring_all_reduce_simple(tensor.clone())
    comms.barrier()
    
    # Test PyTorch's optimized all-reduce
    result_pytorch = tensor.clone()
    comms.barrier()
    dist.all_reduce(result_pytorch, op=dist.ReduceOp.SUM)
    result_pytorch.div_(world_size)  # Average
    comms.barrier()
    
    # Verify results are similar
    if rank == 0:
        diff = (result_ring - result_pytorch).abs().max()
        print(f"Max difference between implementations: {diff.item():.6f}")
        print(f"Results match: {diff.item() < 1e-5}\n")
    
    # Performance comparison
    if rank == 0:
        print("Running performance comparison...")
    
    compare_all_reduce_algorithms(rank, world_size, tensor_size=10000, num_iterations=20)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
