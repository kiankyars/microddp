# DistributedDataParallel (DDP)

## Concept

DDP improves upon Naive DP by:

1. **Gradient Hooks:** Automatically all-reduce gradients during backward pass
2. **Bucket-based All-Reduce:** Groups small gradients into buckets for efficiency
3. **Overlap Computation/Communication:** Can overlap gradient computation with communication

## Algorithm

```
For each training step:
  1. Split batch across ranks
  2. Forward pass on local chunk
  3. Backward pass (gradients automatically all-reduced via hooks)
  4. Optimizer step
```

## Gradient Hooks

Gradient hooks are registered on each parameter. When `loss.backward()` is called:

1. Gradients are computed locally
2. Hooks automatically trigger all-reduce
3. All ranks end up with averaged gradients

## Implementation

See `src/schedule.py::ddp_step` and `register_ddp_hooks`

