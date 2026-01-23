# Gradient Hooks in DDP

## What Are Hooks?

Gradient hooks are functions that get called automatically when a gradient is computed during `backward()`.

## Why Hooks Matter for DDP

Hooks enable **automatic** gradient synchronization during backward pass, without manual intervention.

## Hook Execution Order

**Critical insight:** Hooks are called in **REVERSE** order of the forward pass.

### Example

```python
# Forward pass order:
# Input → Layer1 → Layer2 → Layer3 → Loss

# Backward pass (hook execution order):
# Loss → Layer3 → Layer2 → Layer1 → Input
```

**Why this matters:**
- Last layers compute gradients first
- We can start all-reducing early layers while later layers are still computing
- This enables **computation/communication overlap**

## How DDP Uses Hooks

### Without Hooks (Naive)

```python
loss.backward()
# All gradients computed, THEN we communicate
for param in model.parameters():
    all_reduce_mean(param.grad)
```

**Timeline:**
```
[Compute all gradients] → [All-reduce all gradients] → [Optimizer step]
```

### With Hooks (DDP)

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
            all_reduce_mean(grad)  # Synchronize immediately
        return grad
    return hook

param.register_hook(make_hook())
```

### Bucketed Hook

```python
def make_bucketed_hook(bucket):
    def hook(grad):
        bucket.add_gradient(grad)
        if bucket.ready:
            bucket.all_reduce()  # All-reduce entire bucket
        return grad
    return hook
```

## Measuring Overlap

See `examples/hooks.py` for tools to:
- Track hook execution order
- Measure computation/communication overlap
- Compare hook-based vs manual all-reduce

## Key Takeaways

1. **Hooks enable automatic synchronization** - no manual all-reduce needed
2. **Reverse execution order enables overlap** - communicate early, compute late
3. **Bucketing amplifies benefits** - group hooks for efficiency

## Demo

Run `examples/demo_hooks.py` to see the hook vs manual all-reduce timing comparison. No user changes needed - the demo shows how hooks enable computation/communication overlap.

```bash
torchrun --nproc-per-node=4 examples/demo_hooks.py
```

## Common Pitfalls

### Closure Issues

```python
# WRONG - all hooks use the same param!
for param in model.parameters():
    def hook(grad):
        all_reduce_mean(param.grad)  # Always uses last param!
    param.register_hook(hook)

# CORRECT - capture param in closure
for param in model.parameters():
    def make_hook(p):
        def hook(grad):
            all_reduce_mean(p.grad)
        return hook
    param.register_hook(make_hook(param))
```

### Timing Issues

Hooks are called **during** backward, not after. Don't assume all gradients are ready when a hook is called!

## Further Reading

- [PyTorch Autograd Hooks](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function.register_hook)
- [DDP Hook Implementation](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py)
