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

def cleanup():
    dist.destroy_process_group()

