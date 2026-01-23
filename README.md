# microddp

Data Parallelism from Scratch

## Overview

This course teaches Data Parallelism (DP) by building it from scratch, step by step.

![](docs/microddp.png)

## Quick Start

```bash
uv run torchrun --nproc-per-node=4 src/main.py
```

## Architecture

### Core Modules

- **`comms.py`**: Distributed communication primitives (all-reduce, barriers, broadcast, scatter/gather)
- **`model.py`**: Full MLP model (replicated on each rank)
- **`schedule.py`**: Data parallelism schedules (naive, DDP with bucketing)
- **`main.py`**: Training entry point

### First Principles Modules

- **`allreduce.py`**: All-reduce algorithms from scratch (naive O(n²), ring O(n))
- **`bucketing.py`**: Gradient bucketing for efficient communication
- **`hooks_demo.py`**: Gradient hook execution order and overlap demonstration
- **`performance.py`**: Performance analysis and bottleneck identification

## Data Parallelism Schedules

- `naive_data_parallel_step`: Manual gradient averaging (baseline)
- `ddp_step`: DistributedDataParallel with gradient hooks
- `register_ddp_hooks`: Automatic gradient synchronization with optional bucketing

## Key Features

### From First Principles

1. **All-Reduce Algorithms**: Implement ring all-reduce from scratch, understand O(n) vs O(n²) complexity
2. **Gradient Hooks**: See how hooks enable automatic synchronization and computation/communication overlap
3. **Gradient Bucketing**: Group gradients into buckets for better bandwidth utilization
4. **Performance Analysis**: Measure communication overhead, scaling efficiency, and identify bottlenecks
5. **Synchronization Primitives**: Learn barriers, broadcast, scatter/gather beyond just all-reduce

See [kiankyars.github.io/microddp/](https://kiankyars.github.io/microddp/) for detailed explanations.

## Repo Structure

```text
├── CONTRIBUTING.md
├── README.md
├── docs
├── my_work
│   ├── step1_manual.py
│   ├── step2_comms.py
│   ├── step3_allreduce.py
│   ├── step4_model.py
│   ├── step5_main.py
│   └── step6_ddp.py
├── pyproject.toml
├── src
│   ├── allreduce.py      # All-reduce algorithms from scratch
│   ├── bucketing.py      # Gradient bucketing implementation
│   ├── comms.py          # Communication primitives
│   ├── hooks_demo.py     # Hook execution demonstration
│   ├── main.py           # Training entry point
│   ├── manual.py         # Manual 2-GPU example
│   ├── model.py          # MLP model definition
│   ├── monolith.py       # Single GPU baseline
│   ├── performance.py    # Performance analysis tools
│   └── schedule.py       # DDP schedules
└── uv.lock
```

## Acknowledgements



