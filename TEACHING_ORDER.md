# Course Teaching Order

## 1. Baseline (Single GPU)
- **Doc:** [Outline - Baseline](./docs/outline.md#baseline)
- **Example:** `examples/baseline.py`
- **Concept:** Establish single GPU baseline
- **Lab:** Simple training loop with `nn.Sequential`

## 2. Motivation for Data Parallelism
- **Doc:** [Outline - Motivation](./docs/outline.md#motivation-for-dp)
- **Concept:** Why we need DDP (data wall, batch size, training time)

## 3. Manual Data Parallelism
- **Doc:** [Outline - Manual](./docs/outline.md#manual)
- **Example:** `examples/manual.py`
- **Lab:** `my_work/step1_manual.py`
- **Concept:** Manually split batch across 2 "GPUs" and average gradients
- **Lesson:** Must manually synchronize gradients

## 4. Distributed Basics
- **Doc:** [Outline - Distributed Basics](./docs/outline.md#distributed-basics)
- **Lab:** `my_work/step2_comms.py`
- **Concepts:** Rank, World Size, Process Group, torchrun
- **Implementation:** `init_distributed()`, `DataParallelComms` with `all_reduce_mean()`

## 5. All-Reduce Introduction
- **Doc:** [Naive DP](./docs/naive.md)
- **Lab:** `my_work/step3_allreduce.py`
- **Concept:** What all-reduce does (sum/average across ranks)
- **Uses:** `comms.all_reduce_mean()` from step 4

## 6. Ring All-Reduce Algorithm
- **Doc:** [All-Reduce Algorithms](./docs/allreduce.md)
- **Lab:** `my_work/step4_ring_allreduce.py`
- **Concepts:** O(n) vs O(n²) complexity, ring communication pattern
- **Example:** `examples/allreduce.py`

## 7. Naive Data Parallelism
- **Doc:** [Naive DP](./docs/naive.md)
- **Lab:** `my_work/step5_main.py`
- **Concepts:** Manual gradient averaging, all-reduce each gradient separately
- **Uses:** `comms.all_reduce_mean()` from step 4
- **Implementation:** `src/schedule.py::naive_data_parallel_step`

## 8. DistributedDataParallel (DDP)
- **Doc:** [DDP](./docs/ddp.md)
- **Lab:** `my_work/step6_ddp.py`
- **Concepts:** Gradient hooks for automatic synchronization
- **Implementation:** `src/schedule.py::ddp_step`

## 9. Gradient Hooks (Demo)
- **Doc:** [Gradient Hooks](./docs/hooks.md)
- **Demo:** Run `examples/demo_hooks.py` (no user changes needed)
- **Concepts:** Hook execution order (reverse), computation/communication overlap
- **Shows:** `compare_hook_vs_manual_timing()` comparison

## 10. Gradient Bucketing (Demo)
- **Doc:** [Gradient Bucketing](./docs/bucketing.md)
- **Demo:** Run `examples/demo_bucketing.py` (no user changes needed)
- **Concepts:** Group gradients into buckets, fewer messages, better bandwidth
- **Shows:** `compare_bucketed_vs_unbucketed()` performance comparison
- **Implementation:** `src/bucketing.py`

## 11. Synchronization Primitives
- **Doc:** [Synchronization Primitives](./docs/sync_primitives.md)
- **Concepts:** Barriers, broadcast (O(log n)), reduce (O(log n)), scatter/gather
- **Implementation:** `src/comms.py`

## 12. Performance Analysis
- **Doc:** [Performance Analysis](./docs/performance.md)
- **Example:** `examples/performance.py`
- **Concepts:** Communication overhead, scaling efficiency, bottlenecks
