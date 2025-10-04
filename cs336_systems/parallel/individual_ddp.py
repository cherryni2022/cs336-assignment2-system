import torch.distributed as dist
import torch
import torch.cuda.nvtx as nvtx
from torch.autograd.profiler import record_function

class DDPBase(torch.nn.Module):
    """Common utilities shared by DDP wrappers."""
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        # Defaults; will be overwritten if distributed is initialized
        self.rank = 0
        self.world_size = 1
        self._init_dist_and_broadcast()

    def __getattr__(self, name):
        # Delegate missing attributes to the wrapped module for convenience
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def _init_dist_and_broadcast(self):
        """Initialize rank/world_size and broadcast parameters from rank 0."""
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            with torch.no_grad():
                for p in self.module.parameters():
                    dist.broadcast(p.data, src=0)

class IndividualOverlapDDP(DDPBase):
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.handles = []
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param):
        if param.grad is None or self.world_size <= 1:
            return

        with record_function("allreduce_async"):
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        if self.world_size <= 1:
            return

        for handle in self.handles:
            handle.wait()
            
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad /= self.world_size
        
        self.handles.clear()

