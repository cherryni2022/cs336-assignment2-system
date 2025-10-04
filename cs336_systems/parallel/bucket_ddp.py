import torch.distributed as dist
import torch
import torch.cuda.nvtx as nvtx
from torch.autograd.profiler import record_function
from cs336_systems.parallel.individual_ddp import DDPBase

class BucketDDP(DDPBase):
    """
    - 参数分桶 ：按 bucket_size_mb 将参数分组
    - 逆序遍历 ：从后向前遍历参数（ params_in_reverse ），确保反向传播时的梯度计算顺序
    - 大小控制 ：当当前桶大小超过限制时，创建新桶
    - 映射关系 ：建立参数到桶的映射 ( param_to_bucket )
    """
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__(module)
        # params 分桶放到buckets中
        self.buckets = []
        self.handles = []
        # params到bucket的映射
        self.param_to_bucket = {}

        bucket_size_bytes = bucket_size_mb * 1024 * 1024  # Convert to bytes
        params_in_reverse = list(self.module.parameters())[::-1]
        visited = set()
        # 记录参数分配过程中当前桶的参数列表
        curr_bucket = []
        # 记录参数分配过程中当前桶的大小
        curr_size = 0
        for param in params_in_reverse:
            if param.requires_grad and id(param) not in visited:
                param_size = param.numel() * param.element_size()
                if curr_size + param_size > bucket_size_bytes:
                    self.buckets.append(curr_bucket)
                    curr_bucket = []
                    curr_size = 0
                
                curr_bucket.append(param)
                curr_size += param_size
                visited.add(id(param))
        
        if curr_bucket:
            self.buckets.append(curr_bucket)
        
        # Set up param_to_bucket mapping after all buckets are created
        for i, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[id(param)] = i

        # 记录每步迭代计算中,每个bucket中梯度已经ready的参数个数
        # 通信完成后, 需要清理 bucket_grad_ready_counts
        self.bucket_grad_ready_counts = [0] * len(self.buckets)
        
        # 为所有参数添加计算ready的hook
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param):
        if param.grad is None or self.world_size <= 1:
            return

        bucket_id = self.param_to_bucket[id(param)]
        bucket = self.buckets[bucket_id]
        # 代表当前param梯度计算完,向bucket报告,实际就是count+1
        self.bucket_grad_ready_counts[bucket_id] += 1

        if self.bucket_grad_ready_counts[bucket_id] == len(bucket):
            # 代表当前bucket所有参数梯度计算完,可以进行通信
            grads_to_reduce = [p.grad for p in bucket if p.grad is not None]
            if grads_to_reduce:  # Only proceed if we have gradients to reduce
                flat_grads = torch._utils._flatten_dense_tensors(grads_to_reduce)
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, flat_grads, bucket))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if self.world_size <= 1:
            return

        for handle, flat_grads, bucket in self.handles:
            handle.wait()
            # 梯度取分布式计算结果的平均
            flat_grads /= self.world_size
            
            unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, bucket)
            for param, grad in zip(bucket, unflattened_grads):
                param.grad = grad

        self.handles.clear()
        self.bucket_grad_ready_counts = [0] * len(self.buckets)