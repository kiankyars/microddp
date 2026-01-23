# Gradient Bucketing in DDP

## The Problem

In [naive data parallelism](./naive.md), we all-reduce each gradient separately:

```python
for param in model.parameters():
    if param.grad is not None:
        all_reduce_mean(param.grad)
```

**Issues:**
1. **Many small messages:** Each parameter gradient is a separate all-reduce
2. **Poor bandwidth utilization:** Small messages have high overhead
3. **No overlap:** Can't start communicating early gradients while later ones compute

## The Solution: Bucketing

**Key Idea:** Group small gradients into larger "buckets" and all-reduce the buckets.

### Benefits

1. **Fewer messages:** Instead of N all-reduces (one per parameter), we have B all-reduces (one per bucket)
2. **Better bandwidth:** Larger messages are more efficient
3. **Overlap:** Can start reducing early buckets while later gradients are still computing

## How It Works

### 1. Create Buckets

Parameters are grouped into buckets based on size:

```python
# Example: 16-layer model
# Bucket 1: Layers 16-13 (last layers first - enables overlap!)
# Bucket 2: Layers 12-9
# Bucket 3: Layers 8-5
# Bucket 4: Layers 4-1
```

**Why reverse order?** Last layers compute gradients first during backward pass. By bucketing them first, we can start communicating while earlier layers are still computing!

### 2. Register Hooks

Each parameter gets a hook that:
1. Adds its gradient to the appropriate bucket
2. When bucket is full, triggers all-reduce on the bucket

```python
def hook(grad):
    bucket.add_gradient(grad)
    if bucket.ready:
        bucket.all_reduce()  # All-reduce the entire bucket
    return grad
```

### 3. During Backward

As gradients are computed (in reverse order):
- Early gradients (last layers) → added to buckets → buckets all-reduced
- Later gradients (first layers) → still computing → can overlap with communication

## Bucket Size

**Typical bucket size:** 25 MB (configurable)

**Trade-offs:**
- **Too small:** Many buckets → many all-reduces → high overhead
- **Too large:** Few buckets → less overlap opportunity → slower

**Rule of thumb:** Bucket size should be large enough to amortize communication overhead, but small enough to enable overlap.

## Implementation

See `src/bucketing.py` for the full implementation:
- `GradientBucket`: Groups parameters and their gradients
- `BucketedDDPHooks`: Manages buckets and registers hooks

## Performance Impact

Typical speedups from bucketing:
- **Small models (< 10M params):** 1.2-1.5x
- **Medium models (10M-100M params):** 1.5-2x
- **Large models (> 100M params):** 2-3x

The larger the model, the more benefit from bucketing!

## Demo

Run `examples/demo_bucketing.py` to see the performance improvement from bucketing. No user changes needed - the demo compares bucketed vs unbucketed gradient synchronization.

```bash
torchrun --nproc-per-node=4 examples/demo_bucketing.py
```

## Example

```python
from src.bucketing import BucketedDDPHooks

# Register bucketed hooks
bucketed_hooks = BucketedDDPHooks(model, comms, bucket_size_mb=25.0)

# Training loop - hooks handle bucketing automatically
for step in range(steps):
    loss = model(input, target)
    loss.backward()  # Hooks trigger bucketed all-reduce
    optimizer.step()
```

## Further Reading

- [PyTorch DDP Bucketing](https://pytorch.org/docs/stable/notes/ddp.html#internal-design)
- [Horovod Bucketing](https://horovod.readthedocs.io/en/stable/tuning.html#gradient-bucketing)
