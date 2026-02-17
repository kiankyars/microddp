## Syllabus

### [Intro](./intro.md)

- What is distributed training?

### [Manual](https://github.com/kiankyars/microddp/blob/main/src/manual.py)

- Manually split the batch across two "GPUs" and average gradients.

### [All-Reduce](./allreduce.md)

- How we sync gradients accross devices.

### [DDP](./ddp.md)

- Forward on local chunk, backward, then all-reduce gradients using comms.

### [Optimisations](./optimisations.md)

- Hooks and gradient bucketing

### [Performance Analysis](./performance.md)

- When is DDP worth it, how well does it scale?