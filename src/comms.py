import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2. Set Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    elif torch.cpu.is_available():
        device = torch.device("cpu")
    else:
        exit()

    # 3. Initialize Group
    if torch.cuda.is_available():
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size, device


class DataParallelComms:
    """
    Communication primitives for Data Parallelism.
    Main operation: all-reduce for gradient synchronization.
    """

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        """
        All-reduce operation: sum gradients across all ranks.
        Result is stored in-place in the input tensor.
        """
        dist.all_reduce(tensor, op=op)

    def all_reduce_mean(self, tensor):
        """
        All-reduce with averaging (sum then divide by world_size).
        This is what we need for gradient averaging in DP.
        """
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(self.world_size)

