import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initialize the distributed process group from torchrun environment variables.

    Returns:
        (rank, world_size, device)
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.accelerator.current_accelerator()
    if device == torch.device("mps"):
        device = "cpu"
    backend = torch.distributed.get_default_backend_for_device(device)

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return rank, world_size, device


def cleanup():
    dist.destroy_process_group()

