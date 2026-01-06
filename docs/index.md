# microddp

Data Parallelism from Scratch

## Overview

This course teaches Data Parallelism (DP) by building it from scratch, step by step.

## Course Structure

1. **Monolith** - Baseline single-GPU training
2. **Manual** - Manual gradient averaging across two "GPUs"
3. **Distributed Basics** - Process groups, ranks, all-reduce
4. **Naive DP** - Manual gradient averaging with all-reduce
5. **DDP** - DistributedDataParallel with gradient hooks

## Quick Start

```bash
uv run torchrun --nproc-per-node=4 src/main.py
```

## See Also

- [Outline](./outline.md)
- [Naive DP](./naive.md)
- [DDP](./ddp.md)

