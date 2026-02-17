def baseline_step(model, input_batch, target_batch, device):
    """
    Single-process baseline step:
    1. Forward on the full batch
    2. Backward to compute gradients
    (no distributed communication)
    """
    loss = model(input_batch.to(device), target_batch.to(device))
    loss.backward()
    return loss


def naive_data_parallel_step(model, comms, input_chunk, target_chunk, device):
    """
    Naive Data Parallel step:
    1. Forward pass on local data chunk
    2. Backward pass (computes local gradients)
    3. All-reduce gradients (average across all ranks)
    4. Optimizer step (all ranks have same averaged gradients)
    
    This is the simplest form of data parallelism but inefficient because:
    - Communication happens sequentially after all gradients are computed
    - No overlap between computation and communication
    - Each gradient is all-reduced separately (many small messages)
    """
    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass
    loss.backward()

    # All-reduce gradients (average across all ranks)
    # This happens AFTER all gradients are computed (no overlap)
    for param in model.parameters():
        if param.grad is not None:
            comms.allreducemean(param.grad)

    return loss


def ddp_step(model, comms, input_chunk, target_chunk, device):
    """
    DistributedDataParallel step with gradient hooks.
    This is more efficient than naive DP because:
    - Gradients are reduced asynchronously during backward
    - Uses bucket-based all-reduce for better communication efficiency
    - Enables computation/communication overlap
    """
    # Forward pass
    loss = model(input_chunk, target_chunk)

    # Backward pass (gradients are automatically all-reduced via hooks)
    # Hooks are called as soon as each gradient is ready, enabling overlap
    loss.backward()

    return loss

