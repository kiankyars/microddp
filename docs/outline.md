## Syllabus

### [Baseline](https://github.com/kiankyars/microddp/blob/main/src/baseline.py)

- **Concept:** Establish a baseline on a single CPU/GPU.
- **Lab:** Just `nn.Sequential` with 16 layers and a simple training loop.

### Motivation for DP

- **The Data Wall:** Why we need to process more data faster.
  - **Batch Size:** 1 Million samples.
  - **Training Time:** Days on a single GPU.
  - **Solution:** Data Parallelism (replicate model, split data).

### [Manual](https://github.com/kiankyars/microddp/blob/main/examples/manual.py)

Manually split the batch across two "GPUs" and average gradients.

- **The Exercise:** Try to train by manually averaging gradients from two model copies.
- **The Lesson:** You have to manually synchronize gradients; autograd computes them locally.

### Data Parallelism

#### [All-Reduce Algorithms](./allreduce.md)

- **Concept:** How all-reduce actually works under the hood.
- **Naive All-Reduce:** O(n²) communication complexity (gather + broadcast).
- **Ring All-Reduce:** O(n) communication complexity, optimal bandwidth usage.
- **Lab:** Implement ring all-reduce algorithm from scratch.

#### [Naive DP](./naive.md)

- **Concept:** Manual gradient averaging after backward pass.
- **Lab:** Implement Naive Data Parallel. Forward on local chunk, backward, then all-reduce gradients using comms.

#### [Gradient Hooks](./hooks.md)

- **Concept:** How hooks enable automatic gradient synchronization.
- **Lab:** Compare hook-based vs manual all-reduce timing.

#### [DDP](./ddp.md)

- **Concept:** DistributedDataParallel with gradient hooks for automatic all-reduce during backward.
- **Lab:** DDP with gradient hooks (uses hooks from previous step).

#### [Gradient Bucketing](./bucketing.md)

- **Concept:** Group small gradients into buckets for efficiency.
- **Lab:** Compare bucketed vs unbucketed DDP performance.

#### [Synchronization Primitives](./sync_primitives.md)

- **Barriers:** Synchronize all ranks at a point.
- **Broadcast:** Distribute data from one rank to all.
- **Scatter/Gather:** Distribute/collect data across ranks.
- **Lab:** Use primitives for model initialization and result collection.

#### [Performance Analysis](./performance.md)

- **Communication Overhead:** When is DDP worth it?
- **Scaling Efficiency:** How well does DDP scale?
- **Bottleneck Analysis:** Computation vs communication vs memory bound.
- **Lab:** Profile DDP training and analyze bottlenecks.

---

## Library

**1. `comms.py`**

- **Initialization:** A wrapper around `dist.init_process_group()`.
- **All-Reduce:** `all_reduce_mean(tensor)` for gradient synchronization.
- **Synchronization:** `scatter()`, `gather()`, `all_gather()`.

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

