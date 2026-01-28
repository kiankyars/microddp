# Introduction to Distributed Data Parallelism

## Why Distributed Training?

Training large models on a single GPU faces three challenges:

1. **Model too large**: May not fit in GPU memory.
2. **Batch size**: OOM errors.
3. **Time**: Can take years on huge datasets.

- Scale horizontally (multiple GPUs/servers) or vertically (bigger GPU).

![](../imgs/0.png)

- All formats are supported by torchrun.

![](../imgs/1.png)

## Data Parallel vs Model Parallel

### Data Parallelism

- Model fits in a single GPU.

### Model Parallelism

- Model too large for single GPU.

### DataParallel vs DistributedDataParallel

| Aspect                       | DataParallel (DP)                                     | DistributedDataParallel (DDP)                                           |
|------------------------------|------------------------------------------------------|-------------------------------------------------------------------------|
| Process Model                | Single-process, multi-threaded                       | Multi-process, each process controls one device (GPU)                   |
| Machine Support              | Only works on a single machine                       | Supports both single-machine and multi-machine setups                   |
| Model Replication            | Replicates model across devices within one process (overhead)   | Each process gets its own model replica, handles a subset of the data   |
| Communication                | Via threads, subject to Python GIL and I/O overhead | Uses collectives (e.g. all-reduce) outside Python GIL                   |
| Performance                  | Generally slower, even single-machine                | Much faster, highly scalable; preferred for all single/multi-node cases |

## Distributed Data Parallel (DDP) Workflow

1. **Broadcast**: Initialize model weights on one node, send to all nodes.

   ![](../imgs/2.png)
   ![](../imgs/3.png)

2. **Forward/Backward**: Each node trains on different data chunk, computes local gradients.

   ![](../imgs/4.png)

3. **All-Reduce**: Sum gradients across all nodes, distribute result to all nodes.

   ![](../imgs/5.png)
   ![](../imgs/6.png)

4. **Update**: Each node updates its model using the averaged gradients.

   ![](../imgs/7.png)

## Communication Primitives

### Reduce (All → One)

![](../imgs/8.png)
![](../imgs/9.png)
![](../imgs/10.png)
![](../imgs/11.png)

### Broadcast (One → All)

**Point-to-Point**:

- Time: O(n) where n = number of receivers.

![](../imgs/12.png)

**Naive Collective Communication**:

![](../imgs/13.png)

**Smart Collective Communication**:

![](../imgs/14.png)
![](../imgs/15.png)
![](../imgs/16.png)

### All-Reduce (All → All)

- Reduce + broadcast.

## PyTorch DDP Optimizations

### Computation-Communication Overlap

- Gradient hooks trigger all-reduce immediately when each gradient is ready.
- Communication overlaps with gradient computation, reducing idle time.

### Bucketing

![](../imgs/17.png)

## Failover and Checkpointing

- Only rank 0 saves model checkpoints to avoid conflicts.

