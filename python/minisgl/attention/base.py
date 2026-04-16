"""
base.py - 注意力后端基类定义

本模块定义注意力计算后端的抽象基类和混合后端。

Mini-SGLang 支持多种注意力后端：
- FlashInfer (FI): 基于 FlashInfer 库，支持 prefill 和 decode
- FlashAttention (FA): 基于 FlashAttention 库，主要用于 prefill
- TRT-LLM: 基于 TensorRT-LLM 的 XQA kernel，针对 SM100+ 优化
- HybridBackend: 组合两个后端，prefill 和 decode 使用不同实现

注意力后端职责：
1. forward: 执行注意力计算
2. prepare_metadata: 准备计算所需元数据
3. init_capture_graph / prepare_for_capture / prepare_for_replay: CUDA Graph 支持
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """注意力元数据基类，每个后端实现自己的元数据格式"""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """获取每个序列最后一个 token 的索引，用于 LM Head 提取"""
        ...


class BaseAttnBackend(ABC):
    """注意力后端抽象基类"""

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    """
    混合注意力后端

    组合两个后端，根据 batch 阶段自动选择：
    - prefill 阶段使用 prefill_backend（如 FA）
    - decode 阶段使用 decode_backend（如 FI）
    CUDA Graph 仅作用于 decode 后端。
    """

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
