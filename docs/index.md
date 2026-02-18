# MicroDDP: Data Parallelism from Scratch

- [Principles](./principles.md)

## Syllabus

### [Intro](./intro.md)

- What is distributed training?

### [Manual](https://github.com/kiankyars/microddp/blob/main/my_work/step1_manual.py)

- Manually split the batch across two "GPUs" and average gradients.

### [Sandbox](https://github.com/kiankyars/microddp/blob/main/my_work/step2_sandbox.py)

- Play with all-reduce ops.

### [All-Reduce](./allreduce.md)

- [Various implementations](https://github.com/kiankyars/microddp/blob/main/src/allreduce.py).
- [Lab](https://github.com/kiankyars/microddp/blob/main/my_work/step3_allreduce.py).

### [Optimisations](./optimisations.md)

- Hooks and gradient bucketing

### [Performance Analysis](./performance.md)

- When is DDP worth it, how well does it scale?
