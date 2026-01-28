# DistributedDataParallel (DDP)

## Naive Approach: All-Reduce After Backward

```python
loss = model(input)
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        all_reduce_mean(param.grad)
optim.step()
```

- **No computation/communication overlap:** Must complete entire forward+backward pass before any communication

## Concept

DDP improves upon Naive DP by:

1. **Gradient Hooks:** Automatically all-reduce gradients during backward pass
2. **Bucket-based All-Reduce:** Groups gradients into buckets for efficiency