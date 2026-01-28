# All-Reduce Algorithms from First Principles

## Overview

We have `n` ranks, each with a tensor. We want all ranks to have the **sum** (or average) of all tensors.

Notation: n = number of ranks/GPUs, S = size of the tensor (in bytes or elements).

```
Rank 0: [1, 2, 3]
Rank 1: [4, 5, 6]
Rank 2: [7, 8, 9]

After all-reduce SUM:
All ranks: [12, 15, 18]
```

## Naive All-Reduce

1. All ranks send their tensors to rank 0.
2. Rank 0 sums the tensors.
3. Rank 0 broadcasts the result back to all ranks.

**Communication (total across the network):**
- 2(n - 1) · S ≈ O(nS).

## Tree All-Reduce

- Binary tree topology, giving log n height.
- Per-rank bandwidth: internal nodes send and receive 2S, root and leaves S.

## Ring All-Reduce

**Phase 1 – Scatter-Reduce**
- The tensor is split into chunks; on each hop, ranks **accumulate partial sums** for one chunk.
- After n - 1 steps, each rank holds **one chunk** of the final reduced tensor.

**Phase 2 – All-Gather**
- Using the same ring, ranks circulate their final chunks.
- After another n - 1 steps, every rank has **all chunks**, i.e. the full all-reduce result.

**Communication:**
- Per rank: 2(n - 1) · (S / n) ≈ O(S).
- Total across the network: 2(n - 1) · S ≈ O(nS) (same order as naive, but bandwidth is used much more evenly).

## Implementation

See `examples/allreduce.py`

## Further Reading

- [Baidu's All-Reduce Paper](https://github.com/baidu-research/baidu-allreduce)
- [PyTorch Distributed Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Distributed Communication Tutorial](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)
- [How does NCCL decide which algorithm to use?](https://github.com/NVIDIA/nccl/issues/457)