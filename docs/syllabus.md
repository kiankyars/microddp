## Syllabus

### [Intro](./intro.md)

- What is distributed training?

### [Baseline](https://github.com/kiankyars/microddp/blob/main/src/baseline.py)

- `nn.Sequential` 16 layer MLP.

### [Manual](https://github.com/kiankyars/microddp/blob/main/examples/manual.py)

- Manually split the batch across two "GPUs" and average gradients.

### [All-Reduce](./allreduce.md)

- Implement ring all-reduce from scratch.

### [Naive DP](./naive.md)

- Naive Data Parallel. Forward on local chunk, backward, then all-reduce gradients using comms.

### [DDP](./ddp.md)

- DistributedDataParallel with gradient hooks for automatic all-reduce during backward.

### [Gradient Hooks](./hooks.md)

- Hooks enable automatic gradient synchronization.
- Run `examples/hooks.py`.

### [Gradient Bucketing](./bucketing.md)

- Group gradients into buckets.
- Run `src/bucketing.py`.

### [Performance Analysis](./performance.md)

- When is DDP worth it, how well does it scale?
- Profile DDP training and analyze bottlenecks.