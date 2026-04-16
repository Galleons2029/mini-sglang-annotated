"""
core.py - 核心数据结构定义

本模块定义 Mini-SGLang 的核心数据类型，贯穿整个推理流程：

1. SamplingParams: 采样参数（temperature, top_k, top_p 等）
2. Req: 单个推理请求，跟踪 token 生成进度
3. Batch: 一批请求的集合，区分 prefill/decode 阶段
4. Context: 全局上下文，持有 KV Cache、注意力后端等共享资源

请求生命周期中的长度变量关系：
  |<- cached_len ->|<- extend_len ->|<----- remain_len ----->|
  |      已缓存      |    本次计算     |       剩余输出空间       |
  0            cached_len       device_len            max_device_len
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    """采样参数，控制 token 生成行为"""

    temperature: float = 0.0    # 温度，0 表示贪婪
    top_k: int = -1             # Top-K，-1 表示不限制
    top_p: float = 1.0          # Top-P (Nucleus)，1.0 表示不限制
    ignore_eos: bool = False    # 是否忽略 EOS token
    max_tokens: int = 1024      # 最大生成 token 数

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """
    单个推理请求

    跟踪一个请求从 prefill 到 decode 的完整生命周期。
    使用 eq=False 避免 dataclass 按字段比较（用对象 identity 判断相等）。

    Attributes:
        input_ids: 完整的 token 序列（CPU），包含 prompt + 已生成的 token
        table_idx: 在 page_table 中的行索引
        cached_len: 已在 KV Cache 中的 token 数（前缀缓存命中 + 已计算）
        output_len: 剩余最大输出 token 数
        uid: 用户请求唯一标识
        cache_handle: 前缀缓存句柄，用于管理缓存的锁定/释放
    """

    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)           # 当前总长度（prompt + 已生成）
        self.max_device_len = len(self.input_ids) + self.output_len  # 最大可能长度
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """剩余可生成的 token 数"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本次需要计算注意力的 token 数（= 总长 - 已缓存）"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一个 decode step：cached_len 追上 device_len，device_len +1"""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """将新生成的 token 追加到 CPU 端 input_ids"""
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """
    推理批次

    由 Scheduler 构造，包含一组 Req 和阶段标识。
    Scheduler 在构造后设置 input_ids / positions / out_loc 等 GPU 张量。
    padded_reqs 用于 CUDA Graph（对齐到预定义的 batch size）。
    """

    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # 以下字段由 Scheduler 在 _prepare_batch 中设置
    input_ids: torch.Tensor = field(init=False)     # GPU 上的 token ids
    positions: torch.Tensor = field(init=False)     # 每个 token 的位置（用于 RoPE）
    out_loc: torch.Tensor = field(init=False)       # KV Cache 写入位置
    padded_reqs: List[Req] = field(init=False)      # CUDA Graph 填充后的请求列表
    # 以下字段由注意力后端设置
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """实际请求数"""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """填充后的请求数（CUDA Graph 对齐）"""
        return len(self.padded_reqs)


@dataclass
class Context:
    """
    全局上下文（单例）

    持有整个推理过程共享的资源：
    - page_table: token → KV Cache 位置的映射表
    - attn_backend: 注意力计算后端
    - moe_backend: MoE 计算后端（仅 MoE 模型）
    - kv_cache: KV Cache 显存池
    - _batch: 当前正在 forward 的 Batch（通过 forward_batch 上下文管理）
    """

    page_size: int
    # NOTE: page_table 内部以 token 粒度存储位置，不受 page_size 影响
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """设置当前 forward 的 batch，forward 完成后自动清除"""
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """设置全局 Context（只能调用一次）"""
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """获取全局 Context"""
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
