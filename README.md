# microddp

<img src="docs/imgs/microddp.png" width="260">

## Quick Start

```bash
uv run torchrun --nproc-per-node=4 src/main.py
```

## Architecture

- **`comms.py`**: Distributed communication primitives (all-reduce, barriers, broadcast, scatter/gather)
- **`model.py`**: Full MLP model (replicated on each rank)
- **`main.py`**: Training entry point
- **`bucketing.py`**: Gradient bucketing for efficient communication

## Repo Structure

```text
WIP
```

## Acknowledgments

- Umar Jamil for the figures