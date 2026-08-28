# microddp

<img src="docs/imgs/microddp.png" width="260">

## Quick Start

```bash
torchrun --nproc-per-node=4 src/main.py
```

## Architecture

- **`comms.py`**: Distributed communication primitives (all-reduce, barriers, broadcast, scatter/gather)
- **`model.py`**: Full MLP model (replicated on each rank)
- **`main.py`**: Training entry point
- **`optimisations.py`**: Gradient hooks and bucketed all-reduce

## Repo Structure

```text
.
├── docs/           # Course notes and GitHub Pages source
├── my_work/        # Step-by-step course exercises
├── src/            # Reference DDP implementation
├── .gitignore
├── CLAUDE.md
├── CONTRIBUTING.md
├── README.md
├── pyproject.toml
└── uv.lock
```

## Acknowledgments

- Umar Jamil for the figures