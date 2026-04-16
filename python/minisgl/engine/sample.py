"""
sample.py - Token 采样模块

本模块实现从 logits 采样下一个 token 的功能。

采样策略：
1. Greedy（贪婪）: 选择概率最高的 token
   - temperature = 0
   - 结果确定性，适合代码生成等场景

2. Temperature Sampling（温度采样）:
   - 先将 logits 除以 temperature
   - 然后 softmax 得到概率分布
   - 最后按概率随机采样
   - temperature 越高越随机

3. Top-K Sampling:
   - 只考虑概率最高的 K 个 token
   - 从中按概率采样
   - 限制采样范围，避免选择低概率 token

4. Top-P (Nucleus) Sampling:
   - 按概率从高到低累加，直到累积概率 >= P
   - 只从这些 token 中采样
   - 自适应选择候选数量

采样流程：
  logits → softmax(temperature) → top_k/top_p 过滤 → 采样

使用 FlashInfer 的 sampling 模块加速采样。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """
    批量采样参数
    
    预处理后的采样参数，用于高效批量采样。
    
    Attributes:
        temperatures: 温度参数张量 [batch_size]，None 表示全部 greedy
        top_k: Top-K 参数张量 [batch_size]，None 表示不使用
        top_p: Top-P 参数张量 [batch_size]，None 表示不使用
    """
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    创建 GPU 张量
    
    使用 pin_memory 和 non_blocking 加速 CPU→GPU 传输。
    """
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """
    采样实现（使用 FlashInfer）
    
    Args:
        logits: 模型输出 [batch_size, vocab_size]
        temperatures: 温度参数
        top_k: Top-K 参数
        top_p: Top-P 参数
    
    Returns:
        torch.Tensor: 采样得到的 token IDs [batch_size]
    
    采样组合：
    - 无 top_k/top_p: 纯温度采样
    - 仅 top_k: Top-K 采样
    - 仅 top_p: Top-P 采样
    - 两者都有: Top-K + Top-P 联合采样
    """
    import flashinfer.sampling as sampling

    # 计算概率分布（带温度）
    # enable_pdl: SM90+ 使用 Persistent Data Layout 优化
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    
    # 根据参数组合选择采样方法
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """
    采样器
    
    管理采样过程，支持批量采样。
    
    Attributes:
        device: 计算设备
        vocab_size: 词表大小
    """
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """
        准备采样参数
        
        将每个请求的采样参数转换为批量张量。
        
        Args:
            batch: 请求批次
        
        Returns:
            BatchSamplingArgs: 批量采样参数
        
        优化：
        - 如果全部是 greedy，返回 temperatures=None
        - 只在需要时创建 top_k/top_p 张量
        """
        params = [r.sampling_params for r in batch.reqs]
        
        # 优化：全部 greedy 时直接返回
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        # 处理 temperature
        MIN_P = MIN_T = 1e-6  # 避免除零
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        
        # 处理 top_k（-1 表示不使用，转换为 vocab_size）
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        
        # 处理 top_p（限制在 (0, 1] 范围）
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        
        # 创建张量
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        
        # 只在需要时创建 top_k/top_p
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
            
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """
        执行采样
        
        Args:
            logits: 模型输出 [batch_size, vocab_size]
            args: 采样参数
        
        Returns:
            torch.Tensor: 采样得到的 token IDs [batch_size]
        """
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
