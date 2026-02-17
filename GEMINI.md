# microddp: Data Parallelism from Scratch

`microddp` is an educational project designed to explore and implement Distributed Data Parallelism (DDP) from the ground up using PyTorch. It bridges the gap between high-level framework APIs and the underlying communication primitives required for efficient distributed training.

## Project Overview

The project demonstrates the evolution of data parallelism:
1.  **Manual Synchronization**: Manually averaging gradients between simulated "GPUs" on a single process.
2.  **Naive Distributed Data Parallelism**: Synchronizing gradients sequentially after the entire backward pass across multiple processes.
3.  **Advanced DDP**: Overlapping communication with computation using gradient hooks and optimizing throughput via gradient bucketing.

### Main Technologies
- **Python >= 3.10**
- **PyTorch** (specifically `torch.distributed`)
- **NumPy**
- **uv** (Package and environment management)
- **ruff** (Linting and code style)

## Architecture

- **`src/comms.py`**: Handles distributed environment initialization (`init_distributed`) and provides a wrapper for communication primitives.
- **`src/allreduce.py`**: Educational implementations of All-Reduce algorithms (Reduce+Broadcast, Ring All-Reduce, etc.).
- **`src/schedule.py`**: Implements training steps for different DP strategies (Naive DP vs. DDP).
- **`src/optimisations.py`**: Contains `GradientBucket` and `BucketedDDPHooks` for overlapping backward computation with communication.
- **`src/model.py`**: Defines a `FullMLP` model used as a benchmark for training.
- **`src/main.py`**: The primary entry point for running distributed training experiments.

## Building and Running

### Setup
Ensure you have `uv` installed, then run:
```bash
uv sync
```

### Running Distributed Training
Launch a 4-process distributed training session using `torchrun`:
```bash
uv run torchrun --nproc-per-node=4 src/main.py
```

### Running Performance Benchmarks
To compare different all-reduce implementations and synchronization strategies:
```bash
uv run torchrun --nproc-per-node=4 src/examples.py
```

### Linting
```bash
uv run ruff check .
```

## Development Conventions

- **Educational Focus**: Code is structured for clarity and learning. Comments explain the *why* behind architectural decisions (e.g., why bucketing improves performance).
- **Coding Style**: The project uses `ruff` with a preferred line length of 125 characters.
- **Modular Design**: Communication, model definition, and training schedules are decoupled to allow for easy experimentation with different algorithms.
- **Progressive Learning**: The `my_work/` directory provides a structured path for users to implement DDP features step-by-step.

---
*Note: Some components like `DataParallelComms` in `src/comms.py` are central to the project's logic and are expected by training scripts for higher-level gradient synchronization.*
