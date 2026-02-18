"""
All-reduce implementations for distributed training.

Four different implementations demonstrating various approaches:
1. allreduce1: reduce + broadcast
2. allreduce2: manual send/recv
3. allreduce3: reduce_scatter + all_gather
4. allreduce4: ring all-reduce
"""

import torch
import torch.distributed as dist


def allreduce1(tensor: torch.Tensor):
    """
    Simplest all-reduce: reduce to rank 0, then broadcast.
    """
    dist.reduce(tensor, dst=0)
    dist.broadcast(tensor, src=0)


def allreduce2(rank, tensor):
    """
    Manual all-reduce using explicit send/recv operations.
    """
    result = tensor.clone()
    size = dist.get_world_size()
    
    # Phase 1: Reduce to rank 0
    if rank == 0:
        for src_rank in range(1, dist.get_world_size()):
            recv_tensor = torch.zeros_like(tensor)
            dist.recv(recv_tensor, src=src_rank)
            result += recv_tensor
    else:
        dist.send(tensor, dst=0)
    
    # Phase 2: Broadcast from rank 0
    if rank == 0:
        for dst_rank in range(1, size):
            dist.send(result, dst=dst_rank)
    else:
        dist.recv(result, src=0)


def allreduce3(tensor):
    """
    All-reduce using reduce_scatter + all_gather.
    More bandwidth efficient as it reduces data movement.
    """
    size = dist.get_world_size()
    input_list = list(torch.chunk(tensor, size))
    scattered_chunk = torch.zeros_like(input_list[0])
    dist.reduce_scatter(output=scattered_chunk, input_list=input_list)
    # After reduce_scatter, each scattered_chunk is the sum of that chunk across all ranks
    scattered_chunks = [torch.zeros_like(scattered_chunk) for _ in range(size)]
    dist.all_gather(tensor_list=scattered_chunks, tensor=scattered_chunk)
    return torch.cat(scattered_chunks)


def allreduce4(send, recv):
    """
    Ring all-reduce implementation.
    Optimal for large tensors as it achieves full bandwidth utilization.
    """
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = send.clone()
    recv_buff = send.clone()
    accum = send.clone()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv_buff[:]
        else:
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]
