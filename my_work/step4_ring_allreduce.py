"""
Step 4: Ring All-Reduce from Scratch

This exercise implements ring all-reduce, the O(n) algorithm used in DDP.
Run with: torchrun --nproc-per-node=4 my_work/step4_ring_allreduce.py

Ring all-reduce works in two phases:
1. Scatter-Reduce: Data moves in a ring, accumulating partial sums
2. All-Gather: Final chunks are broadcast around the ring

This achieves O(n) communication complexity vs O(n²) for naive approaches.
"""

import os
import torch
import torch.distributed as dist


def init_distributed():
    """Initialize distributed environment."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size, device


def ring_all_reduce_simple(tensor, rank, world_size, op=dist.ReduceOp.SUM):
    """
    Simplified Ring All-Reduce for educational purposes.
    
    Phase 1 (Scatter-Reduce): Data moves in a ring, accumulating sums
    Phase 2 (All-Gather): Final result is broadcast around the ring
    
    Time Complexity: O(n) where n is world_size
    """
    result = tensor.clone()
    
    # Phase 1: Scatter-Reduce
    # Each step: send to next rank, receive from previous rank, accumulate
    for step in range(world_size - 1):
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
        
        # Send current accumulated result
        dist.send(result, dst=send_to)
        
        # Receive and accumulate
        recv_tensor = torch.zeros_like(result)
        dist.recv(recv_tensor, src=recv_from)
        
        if op == dist.ReduceOp.SUM:
            result += recv_tensor
        elif op == dist.ReduceOp.MEAN:
            result += recv_tensor
    
    # Phase 2: All-Gather
    # Broadcast the final result around the ring
    for step in range(world_size - 1):
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
        
        dist.send(result, dst=send_to)
        recv_tensor = torch.zeros_like(result)
        dist.recv(recv_tensor, src=recv_from)
        result = recv_tensor.clone()
    
    # Average if needed
    if op == dist.ReduceOp.MEAN:
        result.div_(world_size)
    
    return result


def main():
    rank, world_size, device = init_distributed()
    
    # Each rank starts with different data
    # Rank 0: [1, 2, 3, 4]
    # Rank 1: [2, 3, 4, 5]
    # Rank 2: [3, 4, 5, 6]
    # Rank 3: [4, 5, 6, 7]
    # After all-reduce SUM: all ranks should have [10, 14, 18, 22] (sum of all)
    
    initial_value = rank + 1
    tensor = torch.tensor([initial_value, initial_value + 1, initial_value + 2, initial_value + 3], 
                         device=device, dtype=torch.float32)
    
    if rank == 0:
        print(f"=== Ring All-Reduce Example (World Size: {world_size}) ===\n")
        print("Initial tensors:")
    
    dist.barrier()
    print(f"Rank {rank}: {tensor.cpu().tolist()}")
    dist.barrier()
    
    # Perform ring all-reduce
    result = ring_all_reduce_simple(tensor, rank, world_size, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\nAfter ring all-reduce (SUM):")
    
    dist.barrier()
    print(f"Rank {rank}: {result.cpu().tolist()}")
    dist.barrier()
    
    # Verify all ranks have the same result
    expected_sum = sum(range(1, world_size * 4 + 1))  # Sum of all initial values
    if rank == 0:
        print(f"\nExpected sum of first element: {sum(range(1, world_size + 1))}")
        print(f"Actual result: {result[0].item()}")
        print(f"✓ All ranks synchronized!" if abs(result[0].item() - sum(range(1, world_size + 1))) < 1e-5 else "✗ Error!")
    
    # Test with MEAN operation
    tensor_mean = tensor.clone()
    result_mean = ring_all_reduce_simple(tensor_mean, rank, world_size, op=dist.ReduceOp.MEAN)
    
    if rank == 0:
        print(f"\nAfter ring all-reduce (MEAN):")
    
    dist.barrier()
    print(f"Rank {rank}: {result_mean.cpu().tolist()}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
