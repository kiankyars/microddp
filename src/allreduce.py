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

    tensor.copy_(result)


def allreduce3(tensor):
    """
    All-reduce using reduce_scatter + all_gather.
    More bandwidth efficient as it reduces data movement.
    """
    size = dist.get_world_size()
    chunks = list(torch.chunk(tensor, size))
    scattered = torch.zeros_like(chunks[0])
    dist.reduce_scatter(output=scattered, input_list=chunks)
    dist.all_gather(tensor_list=chunks, tensor=scattered)


def allreduce4(tensor):
    """
    Chunked ring all-reduce: the bandwidth-optimal algorithm used in real DDP.

    Phase 1 (Scatter-Reduce): each rank accumulates one chunk's sum.
    Phase 2 (All-Gather): each rank propagates its reduced chunk to all others.

    Each step transfers 1/N of the tensor, so all links are busy simultaneously.
    Total communication: 2 * (N-1)/N * tensor_size (approaches 2x for large N).
    """
    rank = dist.get_rank()
    size = dist.get_world_size()
    left, right = (rank - 1) % size, (rank + 1) % size
    chunks = list(torch.chunk(tensor, size))

    # Phase 1: Scatter-Reduce — after this, rank i holds the full sum of chunk i
    for step in range(size - 1):
        send_idx = (rank - step) % size
        recv_idx = (rank - step - 1) % size
        buf = torch.empty_like(chunks[recv_idx])
        req = dist.isend(chunks[send_idx].contiguous(), right)
        dist.recv(buf, left)
        chunks[recv_idx] = chunks[recv_idx] + buf
        req.wait()

    # Phase 2: All-Gather — propagate each reduced chunk around the ring
    for step in range(size - 1):
        send_idx = (rank - step + 1) % size
        recv_idx = (rank - step) % size
        buf = torch.empty_like(chunks[recv_idx])
        req = dist.isend(chunks[send_idx].contiguous(), right)
        dist.recv(buf, left)
        chunks[recv_idx] = buf
        req.wait()

    tensor.copy_(torch.cat(chunks))