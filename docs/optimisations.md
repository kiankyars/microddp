# Optimisations

## Naive Approach: All-Reduce After Backward

```python
loss = model(input)
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        allreducemean(param.grad)
optim.step()
```

- **No computation/communication overlap:** Must complete entire forward+backward pass before any communication

## Hooks

- Last layers compute gradients first, hooks all-reduce them once they're calculated
- Enables **computation/communication overlap**

## How DDP Uses Hooks

### Without Hooks

```python
loss.backward()
# All gradients computed, THEN we communicate
for param in model.parameters():
    allreducemean(param.grad)
```

### With Hooks

```python
# Hooks registered before backward
register_ddp_hooks(model, comms)

loss.backward()
# Hooks called as each gradient is ready
```

**Timeline:**
```
[Compute grad3] → [All-reduce grad3] ┐
[Compute grad2] → [All-reduce grad2] ├─ Overlap!
[Compute grad1] → [All-reduce grad1] ┘
[Optimizer step]
```

## Hook Implementation

### Simple Hook

```python
def make_hook():
    def hook(grad):
        if grad is not None:
            allreducemean(grad)  # Synchronize immediately
        return grad
    return hook

param.register_hook(make_hook())
```

## Demo

```bash
torchrun --nproc-per-node=4 src/hooks.py
```

## Pitfalls

### Closure Issues

```python
# WRONG - all hooks use the last param!
for param in model.parameters():
    def hook(grad):
        allreducemean(param.grad)
    param.register_hook(hook)
```

### Timing Issues

- The above hooks do not work with gradient accumulation

## Bucketing

- Bucket gradients into and all-reduce the buckets.

### 1. Create Buckets

```python
# Bucket 1: Layers 16-13
# Bucket 2: Layers 12-9
# Bucket 3: Layers 8-5
# Bucket 4: Layers 4-1
```

### 2. Register Hooks

Each parameter gets a hook that:
1. Adds its gradient to the appropriate bucket
2. When bucket is full, triggers all-reduce on the bucket

## Bucket Size

- Bucket size should be large enough to amortize communication overhead, but small enough to enable overlap.
- **Typical bucket size:** 25-100 MB

## Demo

```bash
torchrun --nproc-per-node=4 src/bucketing.py
```

## Further Reading

- [PyTorch Autograd Hooks](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function.register_hook)
- [DDP Hook Implementation](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py)
- [PyTorch DDP Bucketing](https://pytorch.org/docs/stable/notes/ddp.html#internal-design)
- [Horovod Bucketing](https://horovod.readthedocs.io/en/stable/tuning.html#gradient-bucketing)