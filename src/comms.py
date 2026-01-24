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
    Implements all-reduce from first principles using reduce + broadcast.
    """

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def all_reduce_mean(self, tensor):
        """
        All-reduce with averaging implemented from first principles.
        
        Algorithm:
        1. Reduce: Sum all tensors to rank 0 (using reduce)
        2. Broadcast: Distribute the sum from rank 0 to all ranks (using broadcast)
        3. Average: Divide by world_size to get the mean
        
        This demonstrates that all-reduce = reduce + broadcast.
        """
        # Step 1: Reduce all tensors to rank 0 (sum operation)
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        
        # Step 2: Broadcast the sum from rank 0 to all ranks
        dist.broadcast(tensor, src=0)
        
        # Step 3: Average by dividing by world_size
        tensor.div_(self.world_size)

