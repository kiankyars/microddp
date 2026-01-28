# Performance Analysis of DDP

## Communication Overhead

DDP adds communication overhead. Understanding when it's worth it is crucial.

```
Overhead = Communication Time / Total Time
```

### When DDP Breaks Down

DDP becomes inefficient when:

1. **Model too small:** Communication time > computation time
2. **Network too slow:** High latency, low bandwidth
3. **Batch size per rank too small:** Can't hide communication

## Scaling Efficiency

With `n` GPUs, ideal speedup is `n`x (linear scaling).

**Hypothesized efficiency:**
- **2 GPUs:** 85-95% efficiency
- **4 GPUs:** 75-90% efficiency
- **8 GPUs:** 65-85% efficiency
- **16+ GPUs:** 50-75% efficiency (communication becomes bottleneck)

## Bottleneck Analysis

### Computation-Bound
Computation time >> communication time:
- DDP scales near-linearly
- **Scaling:** Add more GPUs

### Communication-Bound
Communication time >> computation time:
- DDP scaling plateaus
- **Solutions:** Gradient compression, larger batch size per GPU, faster interconnect (InfiniBand/NVLink)

### Memory-Bound
GPU memory limits batch size:
- **Solution:** Model parallelism (FSDP, pipeline, or tensor parallelism) instead of pure DDP

## Further Reading

- [PyTorch DDP Performance Tuning](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#performance-tuning)
- [Horovod Performance Guide](https://horovod.readthedocs.io/en/stable/tuning.html)
