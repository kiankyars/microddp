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

### How Broadcast Works: Divide-and-Conquer (O(log n))

Broadcast uses a divide-and-conquer strategy to achieve O(log n) time complexity:

**Algorithm (Binary Tree):**
1. Rank 0 (root) sends to rank 1
2. Ranks 0-1 send to ranks 2-3
3. Ranks 0-3 send to ranks 4-7
4. Continue until all ranks have the data

**Example with 8 ranks:**
```
Step 1: 0 → 1
Step 2: 0 → 2, 1 → 3
Step 3: 0 → 4, 1 → 5, 2 → 6, 3 → 7
```

**Time Complexity:** O(log n) steps where n is world size
- Naive approach (0 sends to each rank): O(n) steps
- Divide-and-conquer: O(log n) steps

**Why it's efficient:** Each step doubles the number of ranks with the data, so we need log₂(n) steps to reach all n ranks.

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

## Reduce

**What it does:** Combine tensors from all ranks using an operation (sum, max, min) and store result on one rank.

```python
# Each rank has a tensor, rank 0 gets the sum
my_tensor = torch.randn(10)

if rank == 0:
    result = torch.zeros(10)
    comms.reduce(my_tensor, dst=0, op=dist.ReduceOp.SUM)
    # result now contains sum of all tensors
else:
    comms.reduce(my_tensor, dst=0, op=dist.ReduceOp.SUM)
```

**Use cases:**
- Sum metrics across all ranks
- Find maximum/minimum values across ranks
- Aggregate before all-reduce (reduce is one-way, all-reduce is two-way)

### How Reduce Works: Divide-and-Conquer (O(log n))

Reduce uses the same divide-and-conquer strategy as broadcast, but in reverse:

**Algorithm (Binary Tree, bottom-up):**
1. Ranks 0-1 reduce to rank 0
2. Ranks 2-3 reduce to rank 2, then rank 2 sends to rank 0
3. Ranks 4-7 reduce to rank 4, then rank 4 sends to rank 0
4. Continue until all data is reduced to rank 0

**Example with 8 ranks (sum operation):**
```
Step 1: 1 → 0 (sum), 3 → 2 (sum), 5 → 4 (sum), 7 → 6 (sum)
Step 2: 2 → 0 (sum), 6 → 4 (sum)
Step 3: 4 → 0 (sum)
```

**Time Complexity:** O(log n) steps where n is world size
- Naive approach (each rank sends to 0): O(n) steps
- Divide-and-conquer: O(log n) steps

**Why it's efficient:** Each step halves the number of ranks that need to send data, so we need log₂(n) steps to reduce everything to rank 0.

**Note:** All-reduce is essentially reduce + broadcast, so it also uses divide-and-conquer and achieves O(log n) complexity (though ring all-reduce is often preferred for bandwidth efficiency).

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
| Reduce | All → One (sum/max/min) | Metric aggregation |
| Scatter | One → All (different) | Data distribution |
| Gather | All → One | Result collection |
| All-Gather | All → All (collect) | Shared statistics |
| Barrier | All → All (sync) | Synchronization |

## Implementation

See `src/comms.py` for implementations of all primitives:
- `barrier()`: Synchronization
- `broadcast()`: One-to-all
- `reduce()`: All-to-one (with reduction operation)
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
