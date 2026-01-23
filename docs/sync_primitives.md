# Synchronization Primitives in Distributed Training

## Overview

Beyond all-reduce, distributed training needs other communication primitives for coordination.

## Barrier

**What it does:** All ranks wait until everyone arrives.

```python
comms.barrier()
# All ranks wait here until everyone reaches this point
```

**Use cases:**
- Synchronize before timing measurements
- Ensure all ranks finish before proceeding
- Coordinate checkpoint saving/loading

**Example:**
```python
# Measure time accurately across all ranks
comms.barrier()
start = time.time()
# ... do work ...
comms.barrier()
end = time.time()
```

## Broadcast

**What it does:** Send tensor from one rank to all others.

```python
# Rank 0 has the tensor, all others get a copy
if rank == 0:
    tensor = torch.randn(10, 10)
else:
    tensor = torch.zeros(10, 10)

comms.broadcast(tensor, src=0)
# Now all ranks have the same tensor (from rank 0)
```

**Use cases:**
- Initialize model weights from rank 0
- Distribute hyperparameters
- Share random seeds for reproducibility

**Example:**
```python
# Initialize model on rank 0, broadcast to all
if rank == 0:
    model_state = model.state_dict()
else:
    model_state = None

# Broadcast state dict (simplified - actual implementation more complex)
comms.broadcast(model_state, src=0)
```

## Scatter

**What it does:** Distribute different chunks of data to each rank.

```python
# Rank 0 has a list of tensors, each rank gets one
if rank == 0:
    scatter_list = [torch.randn(10) for _ in range(world_size)]
else:
    scatter_list = None

my_chunk = comms.scatter(scatter_list, src=0)
# Rank i gets scatter_list[i]
```

**Use cases:**
- Distribute different data chunks to each rank
- Split a large dataset across ranks
- Distribute work items

**Example:**
```python
# Distribute batch indices to each rank
if rank == 0:
    all_indices = list(range(batch_size))
    chunks = [all_indices[i::world_size] for i in range(world_size)]
else:
    chunks = None

my_indices = comms.scatter(chunks, src=0)
```

## Gather

**What it does:** Collect tensors from all ranks to one rank.

```python
# Each rank has a tensor, rank 0 collects all
my_tensor = torch.randn(10)

if rank == 0:
    gather_list = [torch.zeros(10) for _ in range(world_size)]
    comms.gather(my_tensor, gather_list, dst=0)
    # gather_list[i] now has tensor from rank i
else:
    comms.gather(my_tensor, dst=0)
```

**Use cases:**
- Collect results from all ranks to rank 0 for logging
- Aggregate metrics across ranks
- Collect validation results

**Example:**
```python
# Collect losses from all ranks
local_loss = loss.item()

if rank == 0:
    all_losses = [0.0] * world_size
    comms.gather(torch.tensor([local_loss]), 
                 [torch.tensor([l]) for l in all_losses], dst=0)
    avg_loss = sum(all_losses) / world_size
    print(f"Average loss: {avg_loss}")
else:
    comms.gather(torch.tensor([local_loss]), dst=0)
```

## All-Gather

**What it does:** Collect tensors from all ranks to all ranks.

```python
# Each rank has a tensor, all ranks get all tensors
my_tensor = torch.randn(10)

all_tensors = comms.all_gather(my_tensor)
# all_tensors[i] is the tensor from rank i
# All ranks have the same all_tensors list
```

**Use cases:**
- Collecting metrics that all ranks need
- Sharing local batch statistics
- Implementing custom reduction operations

**Example:**
```python
# All ranks need to know batch statistics from all ranks
local_mean = input_chunk.mean()

all_means = comms.all_gather(torch.tensor([local_mean]))
global_mean = torch.stack(all_means).mean()
```

## Communication Patterns Summary

| Primitive | Source → Dest | Use Case |
|-----------|---------------|----------|
| All-Reduce | All → All (sum) | Gradient synchronization |
| Broadcast | One → All | Model initialization |
| Scatter | One → All (different) | Data distribution |
| Gather | All → One | Result collection |
| All-Gather | All → All (collect) | Shared statistics |
| Barrier | All → All (sync) | Synchronization |

## Implementation

See `src/comms.py` for implementations of all primitives:
- `barrier()`: Synchronization
- `broadcast()`: One-to-all
- `scatter()`: One-to-all (different data)
- `gather()`: All-to-one
- `all_gather()`: All-to-all (collection)

## Best Practices

1. **Use barriers for timing:** Ensures accurate measurements
2. **Broadcast for initialization:** Ensures all ranks start with same state
3. **Gather for logging:** Only rank 0 needs to print/log
4. **All-gather sparingly:** More expensive than gather

## Further Reading

- [MPI Communication Primitives](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node109.htm)
- [PyTorch Distributed Communication](https://pytorch.org/docs/stable/distributed.html#collective-communication)
