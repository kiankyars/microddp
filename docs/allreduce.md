# All-Reduce Algorithms from First Principles

## Overview

All-reduce is the fundamental communication primitive in Distributed Data Parallelism. Understanding how it works is crucial to understanding DDP.

## The Problem

We have `n` ranks, each with a tensor. We want all ranks to have the **sum** (or average) of all tensors.

```
Rank 0: [1, 2, 3]
Rank 1: [4, 5, 6]
Rank 2: [7, 8, 9]

After all-reduce SUM:
All ranks: [12, 15, 18]  (1+4+7, 2+5+8, 3+6+9)
```

## Naive All-Reduce: O(n²)

**Algorithm:**
1. Each rank sends its tensor to rank 0
2. Rank 0 sums all tensors
3. Rank 0 broadcasts result to all ranks

**Communication Complexity:**
- Phase 1 (Gather): n-1 sends to rank 0
- Phase 2 (Broadcast): n-1 sends from rank 0
- Total: 2*(n-1) communication steps
- But each step involves all ranks, so total messages: O(n²)

**Why it's inefficient:**
- Rank 0 becomes a bottleneck
- Doesn't utilize network bandwidth efficiently
- Scales poorly with world size

## Ring All-Reduce: O(n)

**Key Insight:** Data moves in a ring, and each rank processes a chunk.

**Algorithm (2 phases):**

### Phase 1: Scatter-Reduce
- Data moves in a ring: rank `i` sends to rank `(i+1) mod n`
- Each rank accumulates partial sums
- After `n-1` steps, each rank has a chunk of the final sum

**Example with 4 ranks:**
```
Step 0: Each rank has its own data
Step 1: Rank 0 → Rank 1, Rank 1 → Rank 2, Rank 2 → Rank 3, Rank 3 → Rank 0
Step 2: Continue ring movement
Step 3: Final step - each rank has accumulated a chunk
```

### Phase 2: All-Gather
- Same ring pattern, but now broadcasting the final chunks
- After `n-1` steps, all ranks have the complete result

**Communication Complexity:**
- Total steps: 2*(n-1) = O(n)
- Each step: one send, one receive per rank
- Total messages: 2*(n-1) = O(n) (optimal!)

**Why it's efficient:**
- No single bottleneck (all ranks participate equally)
- Optimal bandwidth utilization
- Scales linearly with world size

## Complexity Comparison

| World Size | Naive Messages | Ring Messages | Speedup |
|------------|----------------|---------------|---------|
| 2          | 2              | 2             | 1x      |
| 4          | 12             | 6             | 2x      |
| 8          | 56             | 14            | 4x      |
| 16         | 240            | 30            | 8x      |
| 32         | 992            | 62            | 16x     |

**Key Insight:** Ring all-reduce scales linearly, while naive scales quadratically!

## Implementation

See `examples/allreduce.py` for educational implementations:
- `naive_all_reduce()`: O(n²) implementation
- `ring_all_reduce()`: O(n) ring implementation
- `ring_all_reduce()`: Optimized chunked version

## When to Use What

- **Naive:** Only for educational purposes or very small world sizes (n ≤ 2)
- **Ring:** Standard for DDP (used by PyTorch's NCCL backend)
- **Tree:** Alternative for very large world sizes (hierarchical)

## Further Reading

- [Baidu's All-Reduce Paper](https://github.com/baidu-research/baidu-allreduce)
- [PyTorch Distributed Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
