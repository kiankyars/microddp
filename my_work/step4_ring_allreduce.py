"""
Step 4: Ring All-Reduce from Scratch

This exercise implements ring all-reduce, the O(n) algorithm used in DDP.
Run with: torchrun --nproc-per-node=4 my_work/step4_ring_allreduce.py

Ring all-reduce works in two phases:
1. Scatter-Reduce: Data moves in a ring, accumulating partial sums
2. All-Gather: Final result is broadcast around the ring

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
    # TODO: Implement scatter-reduce phase
    # For each step: send to (rank + 1) % world_size, receive from (rank - 1) % world_size, accumulate
    for step in range(world_size - 1):
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
        
        # TODO: Send result, receive, and accumulate
        pass
    
    # Phase 2: All-Gather
    # TODO: Implement all-gather phase
    # For each step: send to (rank + 1) % world_size, receive from (rank - 1) % world_size
    for step in range(world_size - 1):
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
        
        # TODO: Send result, receive, and update
        pass
    
    # TODO: If op is MEAN, divide result by world_size
    
    return result


def main():
    rank, world_size, device = init_distributed()
    
    # Each rank starts with different data
    initial_value = rank + 1
    tensor = torch.tensor([initial_value, initial_value + 1, initial_value + 2, initial_value + 3], 
                         device=device, dtype=torch.float32)
    
    if rank == 0:
        print(f"=== Ring All-Reduce Example (World Size: {world_size}) ===\n")
        print("Initial tensors:")
    
    dist.barrier()
    print(f"Rank {rank}: {tensor.cpu().tolist()}")
    dist.barrier()
    
    # TODO: Perform ring all-reduce and verify all ranks get the same result
    result = ring_all_reduce_simple(tensor, rank, world_size, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\nAfter ring all-reduce (SUM):")
    
    dist.barrier()
    print(f"Rank {rank}: {result.cpu().tolist()}")
    dist.barrier()
    
    if rank == 0:
        expected_sum = sum(range(1, world_size + 1))
        print(f"\nExpected sum of first element: {expected_sum}")
        print(f"Actual result: {result[0].item()}")
        print(f"✓ All ranks synchronized!" if abs(result[0].item() - expected_sum) < 1e-5 else "✗ Error!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
