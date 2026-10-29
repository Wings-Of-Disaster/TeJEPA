"""Differentiable global feature collection for geometry objectives."""
import torch
import torch.distributed as dist


class _Gather(torch.autograd.Function):
    """Differentiable gather using only all_gather and all_reduce collectives.

    Each rank evaluates the same global loss. Summing its adjoints here and
    subsequently averaging parameter gradients in DDP gives the global objective's
    gradient, with no additional world-size multiplier on the loss.
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.world = dist.get_world_size()
        ctx.rank = dist.get_rank()
        tensors = [torch.empty_like(tensor) for _ in range(ctx.world)]
        dist.all_gather(tensors, tensor.contiguous())
        return torch.cat(tensors)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        return gradient.chunk(ctx.world)[ctx.rank].contiguous()


def gather(tensor):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return _Gather.apply(tensor)
    return tensor
