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
            result.zero_()  # Will be overwritten by broadcast
        
        # Phase 2: Broadcast from rank 0
        dist.broadcast(result, src=0)
        
        return result

    def ring_all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        """
        Ring All-Reduce: O(n) communication complexity, optimal bandwidth
        
        Algorithm (2 phases):
        Phase 1 - Scatter-Reduce:
        - Data moves in a ring: rank i sends to rank (i+1) mod n
        - Each rank accumulates partial sums
        - After n-1 steps, each rank has a chunk of the final sum
        
        Phase 2 - All-Gather:
        - Same ring pattern, but now broadcasting the final chunks
        - After n-1 steps, all ranks have the complete result
        
        Total steps: 2*(n-1) = O(n)
        Each step: one send, one receive per rank
        Total messages: 2*(n-1) = O(n) (optimal!)
        """
        chunk_size = tensor.numel() // self.world_size
        if chunk_size == 0:
            chunk_size = tensor.numel()
        
        # Reshape tensor into chunks for ring communication
        tensor_flat = tensor.flatten()
        chunks = []
        for i in range(self.world_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, tensor.numel())
            if i < self.world_size - 1:
                chunks.append(tensor_flat[start_idx:end_idx].clone())
            else:
                # Last chunk gets remainder
                chunks.append(tensor_flat[start_idx:].clone())
        
        # Pad chunks to same size if needed
        max_chunk_size = max(c.numel() for c in chunks)
        for i, chunk in enumerate(chunks):
            if chunk.numel() < max_chunk_size:
                padding = torch.zeros(max_chunk_size - chunk.numel(), 
                                     dtype=chunk.dtype, device=chunk.device)
                chunks[i] = torch.cat([chunk, padding])
        
        # Phase 1: Scatter-Reduce
        # Each rank starts with its own chunk
        my_chunk_idx = self.rank
        accumulated = chunks[my_chunk_idx].clone()
        
        for step in range(self.world_size - 1):
            # Send to next rank in ring
            send_to = (self.rank + 1) % self.world_size
            # Receive from previous rank in ring
            recv_from = (self.rank - 1) % self.world_size
            
            # Calculate which chunk we're working on in this step
            chunk_idx = (self.rank - step) % self.world_size
            
            # Send our current accumulated chunk
            send_tensor = accumulated.clone()
            dist.send(send_tensor, dst=send_to)
            
            # Receive and accumulate
            recv_tensor = torch.zeros_like(accumulated)
            dist.recv(recv_tensor, src=recv_from)
            
            if op == dist.ReduceOp.SUM:
                accumulated += recv_tensor
            elif op == dist.ReduceOp.MEAN:
                accumulated += recv_tensor
        
        # Phase 2: All-Gather
        # Now we broadcast the final chunks around the ring
        result_chunks = [None] * self.world_size
        result_chunks[my_chunk_idx] = accumulated.clone()
        
        for step in range(self.world_size - 1):
            send_to = (self.rank + 1) % self.world_size
            recv_from = (self.rank - 1) % self.world_size
            
            # Calculate which chunk we're broadcasting in this step
            chunk_idx = (self.rank - step - 1) % self.world_size
            
            # Send the chunk we have
            if result_chunks[chunk_idx] is not None:
                dist.send(result_chunks[chunk_idx], dst=send_to)
            
            # Receive and store
            recv_tensor = torch.zeros_like(accumulated)
            dist.recv(recv_tensor, src=recv_from)
            result_chunks[(chunk_idx - 1) % self.world_size] = recv_tensor.clone()
        
        # Reconstruct the full tensor from chunks
        result = torch.cat([chunk[:chunks[i].numel()] 
                           for i, chunk in enumerate(result_chunks)], dim=0)
        
        # Reshape to original shape
        return result.reshape(tensor.shape)

    def ring_all_reduce_simple(self, tensor, op=dist.ReduceOp.SUM):
        """
        Simplified Ring All-Reduce for educational purposes.
        Works on the full tensor (not chunked) to make the pattern clearer.
        
        This is less bandwidth-efficient but easier to understand.
        """
        result = tensor.clone()
        
        # Phase 1: Scatter-Reduce
        for step in range(self.world_size - 1):
            send_to = (self.rank + 1) % self.world_size
            recv_from = (self.rank - 1) % self.world_size
            
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
        for step in range(self.world_size - 1):
            send_to = (self.rank + 1) % self.world_size
            recv_from = (self.rank - 1) % self.world_size
            
            dist.send(result, dst=send_to)
            recv_tensor = torch.zeros_like(result)
            dist.recv(recv_tensor, src=recv_from)
            result = recv_tensor.clone()
        
        # Average if needed
        if op == dist.ReduceOp.MEAN:
            result.div_(self.world_size)
        
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
        _ = algorithms.ring_all_reduce_simple(tensor.clone())
        dist.barrier()
    
    # Benchmark ring all-reduce
    dist.barrier()
    start = time.time()
    for _ in range(num_iterations):
        result_ring = algorithms.ring_all_reduce_simple(tensor.clone())
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
        print(f"Ring All-Reduce (simple): {ring_time*1000:.2f} ms")
        print(f"PyTorch All-Reduce: {pytorch_time*1000:.2f} ms")
        print(f"Speedup: {ring_time/pytorch_time:.2f}x")
    
    return ring_time, pytorch_time
