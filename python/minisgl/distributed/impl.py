"""
impl.py - 分布式通信实现

本模块实现 Tensor Parallelism 所需的分布式通信操作。

Tensor Parallelism (TP) 是一种模型并行策略：
- 将模型的权重按列或行分割到多个 GPU
- 每个 GPU 计算部分结果
- 通过 AllReduce 或 AllGather 合并结果

支持的通信后端：
1. TorchDistributedImpl: 使用 PyTorch 的 torch.distributed
2. PyNCCLDistributedImpl: 使用自定义的 PyNCCL 封装

为什么需要自定义 PyNCCL：
- 更好的性能控制
- 支持与 CUDA 流的精细集成
- 避免 torch.distributed 的一些限制

通信操作：
- AllReduce: 所有 GPU 对张量求和，结果广播给所有 GPU
  用于：合并 attention output、合并 FFN output
  
- AllGather: 收集所有 GPU 的张量片段，拼接成完整张量
  用于：embedding lookup、LM head output
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """
    分布式通信抽象基类
    
    定义了 TP 所需的通信操作接口。
    """
    
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        AllReduce 操作
        
        所有 GPU 对输入张量求和，结果广播给所有 GPU。
        
        Args:
            x: 输入张量（会被原地修改）
        
        Returns:
            torch.Tensor: 求和后的张量
        """
        ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """
        AllGather 操作
        
        收集所有 GPU 的张量，沿第一维拼接。
        
        Args:
            x: 本 GPU 的张量片段
        
        Returns:
            torch.Tensor: 拼接后的完整张量
        """
        ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """
    基于 PyTorch torch.distributed 的实现
    
    使用 PyTorch 内置的 NCCL 后端。
    简单可靠，但灵活性较低。
    """
    
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """AllReduce: 所有 GPU 求和"""
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x  # 单 GPU 不需要通信
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """AllGather: 收集所有 GPU 的张量"""
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x  # 单 GPU 不需要通信
        # 创建输出张量（第一维扩大 tp_size 倍）
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """
    基于自定义 PyNCCL 的实现
    
    使用 Mini-SGLang 的自定义 NCCL 封装，提供：
    - 更好的性能控制
    - 与 CUDA 流的精细集成
    - 预分配缓冲区，减少内存分配开销
    
    Attributes:
        comm: PyNCCL 通信器
    """
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """使用 PyNCCL 执行 AllReduce"""
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """使用 PyNCCL 执行 AllGather"""
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """
    分布式通信器（单例模式）
    
    使用插件列表管理多个通信实现。
    默认使用最后添加的插件。
    
    设计原因：
    - 支持在运行时切换通信后端
    - 默认使用 TorchDistributed
    - 如果启用 PyNCCL，则使用 PyNCCL
    
    Class Attributes:
        plugins: 通信实现列表，默认包含 TorchDistributedImpl
    """
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """使用当前插件执行 AllReduce"""
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """使用当前插件执行 AllGather"""
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    启用 PyNCCL 分布式通信
    
    初始化 PyNCCL 通信器并添加到插件列表。
    
    Args:
        tp_info: TP 信息（rank 和 size）
        tp_cpu_group: CPU 进程组（用于初始化同步）
        max_bytes: 最大通信数据量（用于预分配缓冲区）
    """
    if tp_info.size == 1:
        return  # 单 GPU 不需要
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    销毁所有分布式通信插件
    
    在关闭时调用，清理资源。
    """
    DistributedCommunicator.plugins = []
