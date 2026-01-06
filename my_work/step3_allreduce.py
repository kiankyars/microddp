import torch
import torch.distributed as dist

# TODO: Test all-reduce operation
# This exercise helps you understand how all-reduce works

# Simulate all-reduce manually (without actual distributed setup)
# Create tensors on different "ranks" and manually compute the result

# Rank 0 tensor
tensor0 = torch.tensor([1.0, 2.0, 3.0])

# Rank 1 tensor
tensor1 = torch.tensor([4.0, 5.0, 6.0])

# Rank 2 tensor
tensor2 = torch.tensor([7.0, 8.0, 9.0])

# TODO: What is the result of all-reduce SUM?
# result = ?

# TODO: What is the result of all-reduce MEAN (sum then divide by world_size)?
# result_mean = ?

print("All-reduce SUM result:", None)  # TODO
print("All-reduce MEAN result:", None)  # TODO

