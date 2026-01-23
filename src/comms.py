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
    
    Additional primitives:
    - Barriers: Synchronize all ranks at a point
    - Broadcast: Distribute data from one rank to all others
    - Scatter/Gather: Distribute/collect data across ranks
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

    def barrier(self):
        """
        Barrier synchronization: all ranks wait here until everyone arrives.
        
        Use cases:
        - Ensure all ranks finish computation before proceeding
        - Synchronize before timing measurements
        - Coordinate checkpoint saving/loading
        """
        dist.barrier()

    def broadcast(self, tensor, src=0):
        """
        Broadcast: send tensor from src rank to all other ranks.
        
        Use cases:
        - Initialize model weights from rank 0
        - Distribute hyperparameters
        - Share random seeds for reproducibility
        
        Args:
            tensor: Tensor to broadcast (in-place operation)
            src: Source rank (default: 0)
        """
        dist.broadcast(tensor, src=src)

    def scatter(self, scatter_list, src=0):
        """
        Scatter: distribute a list of tensors from src to all ranks.
        
        Each rank receives scatter_list[rank] from the source.
        
        Use cases:
        - Distribute different data chunks to each rank
        - Split a large dataset across ranks
        
        Args:
            scatter_list: List of tensors (only used on src rank)
            src: Source rank (default: 0)
        
        Returns:
            Tensor received by this rank
        """
        if self.rank == src:
            assert len(scatter_list) == self.world_size, \
                f"scatter_list must have {self.world_size} tensors"
            dist.scatter(scatter_list[self.rank], scatter_list, src=src)
            return scatter_list[self.rank]
        else:
            output = torch.zeros_like(scatter_list[0]) if src == 0 else None
            dist.scatter(output, src=src)
            return output

    def gather(self, tensor, gather_list=None, dst=0):
        """
        Gather: collect tensors from all ranks to dst rank.
        
        Use cases:
        - Collect results from all ranks to rank 0 for logging
        - Aggregate metrics across ranks
        
        Args:
            tensor: Tensor to send from this rank
            gather_list: List to store gathered tensors (only used on dst)
            dst: Destination rank (default: 0)
        
        Returns:
            gather_list if rank == dst, None otherwise
        """
        if self.rank == dst:
            if gather_list is None:
                gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gather_list, dst=dst)
            return gather_list
        else:
            dist.gather(tensor, dst=dst)
            return None

    def all_gather(self, tensor):
        """
        All-Gather: collect tensors from all ranks to all ranks.
        
        Unlike gather(), all ranks receive the result.
        
        Use cases:
        - Collecting metrics that all ranks need
        - Sharing local batch statistics
        
        Args:
            tensor: Tensor to send from this rank
        
        Returns:
            List of tensors from all ranks
        """
        gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, tensor)
        return gather_list

