# Naive Data Parallelism

## Concept

In Naive Data Parallelism:

1. Each rank has a **complete copy** of the model
2. Each rank processes a **different chunk** of the batch
3. After backward pass, **gradients are averaged** across all ranks using all-reduce
4. All ranks update with the **same averaged gradients**

## Algorithm

```
For each training step:
  1. Split batch across ranks
  2. Forward pass on local chunk (each rank)
  3. Backward pass (computes local gradients)
  4. All-reduce gradients (average across all ranks)
  5. Optimizer step (all ranks have same averaged gradients)
```

## Two Naive Approaches

There are two naive ways to all-reduce gradients, both with problems:

### Approach 1: All-Reduce Each Gradient Separately

Each parameter's gradient is all-reduced individually in a loop:

```python
for param in model.parameters():
    if param.grad is not None:
        all_reduce_mean(param.grad)
```

**Problems:**
- **Many small messages:** N parameters = N separate all-reduce operations
- **Poor bandwidth utilization:** Small messages have high communication overhead
- **No overlap:** Communication happens sequentially after all gradients are computed

### Approach 2: Concatenate and All-Reduce Once

Flatten all gradients into one tensor and all-reduce it:

```python
# Flatten all gradients
all_grads = torch.cat([p.grad.flatten() for p in model.parameters()])
all_reduce_mean(all_grads)
# Unflatten back to parameters
offset = 0
for param in model.parameters():
    grad_size = param.grad.numel()
    param.grad.copy_(all_grads[offset:offset+grad_size].reshape(param.grad.shape))
    offset += grad_size
```

**Problems:**
- **No overlap:** Still must wait for all gradients before communicating
- **Memory overhead:** Need to concatenate all gradients (extra memory copy)
- **Less flexible:** Can't start communicating early gradients while later ones compute

## Why Both Are "Naive"

Both approaches share the fundamental problem:

- **No computation/communication overlap:** Must complete entire forward+backward pass before any communication
- **Sequential execution:** Computation and communication happen in separate phases

This is the baseline that DDP improves upon with gradient hooks (which enable overlap) and bucketing (which optimizes communication).

## Key Difference from Pipeline Parallelism

- **Pipeline Parallelism:** Model is split across ranks (each rank has a slice)
- **Data Parallelism:** Model is replicated across ranks (each rank has full model)

## Implementation

See `src/schedule.py::naive_data_parallel_step`

