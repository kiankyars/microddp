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

## Key Difference from Pipeline Parallelism

- **Pipeline Parallelism:** Model is split across ranks (each rank has a slice)
- **Data Parallelism:** Model is replicated across ranks (each rank has full model)

## Implementation

See `src/schedule.py::naive_data_parallel_step`

