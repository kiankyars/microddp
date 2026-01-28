# Naive Approach: All-Reduce After Backward

```python
loss = model(input)
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        all_reduce_mean(param.grad)
optim.step()
```

- **No computation/communication overlap:** Must complete entire forward+backward pass before any communication