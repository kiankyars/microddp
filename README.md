# microddp

![](/docs/microddp.png)

## Quick Start

```bash
uv run torchrun --nproc-per-node=4 src/main.py
```

## Architecture

- **`comms.py`**: Distributed communication primitives (all-reduce, barriers, broadcast, scatter/gather)
- **`model.py`**: Full MLP model (replicated on each rank)
- **`schedule.py`**: Data parallelism schedules (naive, DDP with bucketing)
- **`main.py`**: Training entry point
- **`bucketing.py`**: Gradient bucketing for efficient communication

## Data Parallelism Schedules

- `naive_data_parallel_step`: Manual gradient averaging (baseline)
- `ddp_step`: DistributedDataParallel with gradient hooks
- `register_ddp_hooks`: Automatic gradient synchronization with optional bucketing

## Repo Structure

```text
├── CONTRIBUTING.md
├── README.md
├── docs
├── my_work
│   ├── step1_manual.py
│   ├── step2_comms.py
│   ├── step3_allreduce.py
│   ├── step4_ring_allreduce.py
│   ├── step5_main.py
│   └── step6_ddp.py
├── pyproject.toml
├── src
│   ├── bucketing.py      # Gradient bucketing implementation
│   ├── comms.py          # Communication primitives
│   ├── main.py           # Training entry point
│   ├── model.py          # MLP model definition
│   └── schedule.py       # DDP schedules
├── examples
│   ├── allreduce.py      # All-reduce algorithms from scratch
│   ├── baseline.py        # Single GPU baseline
│   ├── hooks.py           # Hook execution demonstration
│   ├── manual.py         # Manual 2-GPU example
│   └── performance.py    # Performance analysis tools
└── uv.lock
```

## Library

**1. `comms.py`**

- **Initialization:** A wrapper around `dist.init_process_group()`.
- **All-Reduce:** Implemented using reduce + broadcast.

**2. `model.py`**

- **FullMLP:** A class `FullMLP(nn.Module)` with 16 `Linear` layers. Replicated on each rank.

**3. `schedule.py`**

- **Naive DP:** Manual gradient averaging after backward.
- **DDP:** Gradient hooks for automatic synchronization during backward.
- **Bucketing:** Optional bucketed hooks for better efficiency.

**4. `bucketing.py`**

- **GradientBucket:** Groups parameters and their gradients.
- **BucketedDDPHooks:** Manages buckets and registers hooks.
- **compare_bucketed_vs_unbucketed():** Performance comparison.

