# Performance Analysis of DDP

## Communication Overhead

DDP adds communication overhead. Understanding when it's worth it is crucial.

### Overhead Formula

```
Overhead = Communication Time / Total Time
```

**Typical overhead:**
- **Small models (< 10M params):** 50-80% overhead (often not worth it!)
- **Medium models (10M-100M params):** 20-50% overhead
- **Large models (> 100M params):** 5-20% overhead (very efficient!)

### When DDP Breaks Down

DDP becomes inefficient when:

1. **Model too small:** Communication time > computation time
2. **Network too slow:** High latency, low bandwidth
3. **Batch size per rank too small:** Can't hide communication

### Example Analysis

```python
from performance import DDPPerformanceAnalyzer

analyzer = DDPPerformanceAnalyzer(comms, device)

for step in range(steps):
    timing = analyzer.time_step(model, input, target, optimizer)
    # Returns: {'forward': ..., 'backward': ..., 'communication': ..., ...}

overhead = analyzer.analyze_communication_overhead()
# Returns: {'communication_overhead_percent': 25.3, ...}
```

## Scaling Efficiency

### Ideal Speedup

With `n` GPUs, ideal speedup is `n`x (linear scaling).

### Actual Speedup

```
Actual Speedup = Single GPU Time / Multi GPU Time
Efficiency = Actual Speedup / Ideal Speedup
```

**Typical efficiency:**
- **2 GPUs:** 85-95% efficiency
- **4 GPUs:** 75-90% efficiency
- **8 GPUs:** 65-85% efficiency
- **16+ GPUs:** 50-75% efficiency (communication becomes bottleneck)

### Why Efficiency Drops

1. **Communication overhead increases** with world size
2. **Load imbalance** (if batch doesn't divide evenly)
3. **Synchronization overhead** (barriers, etc.)

## Communication Time Estimation

### Ring All-Reduce Time

```
Time = 2 * (n - 1) * (latency + chunk_size / bandwidth)
```

Where:
- `n`: world size
- `latency`: network latency (typically 0.1-1 ms)
- `chunk_size`: tensor size / n
- `bandwidth`: network bandwidth (typically 10-100 Gbps)

### Example

```python
from performance import estimate_communication_time

# 100 MB tensor, 4 GPUs, 10 Gbps network
comm_time = estimate_communication_time(
    tensor_size_bytes=100 * 1024 * 1024,
    world_size=4,
    bandwidth_gbps=10.0,
    latency_ms=0.1
)
# Returns: ~0.08 seconds
```

## Bottleneck Analysis

### Computation-Bound

When computation time >> communication time:
- DDP is very efficient
- Scaling is near-linear
- **Solution:** Use more GPUs!

### Communication-Bound

When communication time >> computation time:
- DDP is inefficient
- Scaling plateaus
- **Solutions:**
  - Use gradient compression
  - Increase batch size per rank
  - Use faster network (InfiniBand, NVLink)

### Memory-Bound

When GPU memory is the limit:
- Can't increase batch size
- May need model parallelism instead
- **Solution:** Use pipeline parallelism or tensor parallelism

## Performance Tools

See `examples/performance.py` for:
- `DDPPerformanceAnalyzer`: Detailed timing breakdown
- `analyze_scaling_efficiency()`: Scaling analysis
- `estimate_communication_time()`: Communication estimation
- `communication_complexity_analysis()`: Complexity comparison

## Best Practices

1. **Profile first:** Use `DDPPerformanceAnalyzer` to measure overhead
2. **Optimize communication:** Use bucketing, gradient compression
3. **Balance batch size:** Large enough to hide communication, small enough to fit in memory
4. **Monitor efficiency:** Track scaling efficiency as you add GPUs

## Further Reading

- [PyTorch DDP Performance Tuning](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#performance-tuning)
- [Horovod Performance Guide](https://horovod.readthedocs.io/en/stable/tuning.html)
