import torch
import torch.distributed as dist


def register_hooks(model):
    """All-reduce each gradient as it's computed during backward."""
    world_size = dist.get_world_size()
    def hook(grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(world_size)
        return grad
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(hook)


class GradientBucket:
    def __init__(self, params):
        self.params = params
        self.grads = []

    def add(self, grad):
        self.grads.append(grad)
        return len(self.grads) == len(self.params)

    def all_reduce(self):
        world_size = dist.get_world_size()
        flat = torch.cat([g.flatten() for g in self.grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for grad in self.grads:
            n = grad.numel()
            grad.copy_(flat[offset:offset + n].reshape(grad.shape))
            offset += n
        self.grads = []


def register_bucketed_hooks(model, bucket_size_mb=25.0):
    """All-reduce gradients in buckets during backward."""
    bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
    params = list(reversed([p for p in model.parameters() if p.requires_grad]))

    buckets, param_to_bucket = [], {}
    current, current_size = [], 0
    for p in params:
        if current and current_size + p.numel() * p.element_size() > bucket_size_bytes:
            b = GradientBucket(current)
            buckets.append(b)
            for q in current:
                param_to_bucket[q] = b
            current, current_size = [], 0
        current.append(p)
        current_size += p.numel() * p.element_size()
    if current:
        b = GradientBucket(current)
        buckets.append(b)
        for q in current:
            param_to_bucket[q] = b

    def make_hook(param):
        def hook(grad):
            bucket = param_to_bucket[param]
            if bucket.add(grad):
                bucket.all_reduce()
            return grad
        return hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(make_hook(p))
