"""
Step 3: All-Reduce from Scratch

This exercise implements ring all-reduce, the O(n) algorithm used in DDP.
Run with: PYTHONPATH=. torchrun --nproc-per-node=4 my_work/step3_allreduce.py

Ring all-reduce works in two phases:
1. Scatter-Reduce: Data moves in a ring, accumulating partial sums
2. All-Gather: Final result is broadcast around the ring

This achieves O(n) communication complexity vs O(n²) for naive approaches.
"""

import torch
import torch.distributed as dist

from src.comms import init_distributed


def ring_all_reduce(tensor, rank, world_size):
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
    result = ring_all_reduce(tensor, rank, world_size)
    
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
