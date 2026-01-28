## Syllabus

### [Intro](./intro.md)

- What is distributed training?

### [Baseline](https://github.com/kiankyars/microddp/blob/main/src/baseline.py)

- `nn.Sequential` 16 layer MLP.

### [Manual](https://github.com/kiankyars/microddp/blob/main/src/manual.py)

- Manually split the batch across two "GPUs" and average gradients.

### [All-Reduce](./allreduce.md)

- How we sync gradients accross devices.

### [DDP](./ddp.md)

- Forward on local chunk, backward, then all-reduce gradients using comms.

### [Optimisations](./optimisations.md)

- Hooks and gradient bucketing:
  - Hooks enable automatic gradient synchronization.
  - Bucketing groups gradients into larger messages.
- Run `src/hooks.py` and `src/bucketing.py`.

### [Performance Analysis](./performance.md)

- When is DDP worth it, how well does it scale?
- Profile DDP training and analyze bottlenecks.