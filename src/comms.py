import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return rank, world_size, acc


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

