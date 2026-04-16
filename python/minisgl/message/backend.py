"""
backend.py - 后端消息定义

本模块定义 Tokenizer → Scheduler 的消息格式。

消息流向：
  Frontend → Tokenizer → [Backend Msg] → Scheduler → Engine

消息类型：
1. UserMsg: 用户请求（已分词）
2. ExitMsg: 退出信号
3. AbortBackendMsg: 取消请求
4. BatchBackendMsg: 批量消息封装

数据流：
1. Frontend 发送 TokenizeMsg（包含文本）
2. Tokenizer 分词，转换为 UserMsg（包含 input_ids）
3. Scheduler 接收 UserMsg，创建 Req

序列化：使用 JSON + 自定义类型标记
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """
    后端消息基类
    
    提供序列化/反序列化接口。
    使用 serialize_type/deserialize_type 处理类型信息。
    """
    
    def encoder(self) -> Dict:
        """序列化为 JSON 字典"""
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        """从 JSON 字典反序列化"""
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """
    批量消息封装
    
    用于一次发送多个消息，减少 ZMQ 通信次数。
    
    Attributes:
        data: 消息列表
    """
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """
    退出消息
    
    用于通知 Scheduler 优雅退出。
    """
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """
    用户请求消息（已分词）
    
    这是经过 Tokenizer 处理后的请求。
    
    Attributes:
        uid: 用户请求 ID（用于关联响应）
        input_ids: 输入 token IDs（CPU int32 张量）
        sampling_params: 采样参数
    """
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """取消请求消息，通知 Scheduler 中止指定 uid 的请求"""

    uid: int
