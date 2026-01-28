# microddp

![](docs/imgs/microddp.png)

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

## Acknowledgments

- Umar Jamil for the figures