import os

import torch
import torch.distributed as dist

# TODO: Implement init_distributed()
# Read environment variables: RANK, WORLD_SIZE, LOCAL_RANK
# Set device based on CUDA availability
# Initialize process group with appropriate backend


def init_distributed():
    # TODO
    pass


# TODO: Implement DataParallelComms class
# It should have:
# - __init__(rank, world_size)
# - all_reduce(tensor, op) - basic all-reduce
# - all_reduce_mean(tensor) - all-reduce with averaging


class DataParallelComms:
    def __init__(self, rank, world_size):
        # TODO
        pass

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        # TODO
        pass

    def all_reduce_mean(self, tensor):
        # TODO
        pass

