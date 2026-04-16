"""
frontend.py - 前端消息定义

本模块定义 Scheduler → DeTokenizer → Frontend 的消息格式。

消息流向：
  Engine → Scheduler → DeTokenizer → [Frontend Msg] → Frontend → 用户
  
消息类型：
1. UserReply: 用户响应（增量输出）
2. BatchFrontendMsg: 批量消息封装

数据流：
1. Engine 采样得到 token ID
2. Scheduler 发送给 DeTokenizer
3. DeTokenizer 反分词，生成 UserReply
4. Frontend 接收 UserReply，流式返回给用户

特点：
- 增量输出：每次只返回新生成的文本
- finished 标记：表示生成完成
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """
    前端消息基类
    
    提供序列化/反序列化接口。
    """
    
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        """序列化为 JSON 字典"""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        """从 JSON 字典反序列化"""
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """
    批量消息封装
    
    用于一次发送多个响应，提高效率。
    
    Attributes:
        data: 消息列表
    """
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """
    用户响应消息
    
    包含增量生成的文本和完成状态。
    
    Attributes:
        uid: 用户请求 ID（用于关联请求）
        incremental_output: 增量输出文本（只包含新生成的部分）
        finished: 是否完成（True 表示生成结束）
    
    流式输出示例：
    - 第1条: uid=1, incremental_output="Hello", finished=False
    - 第2条: uid=1, incremental_output=" World", finished=False
    - 第3条: uid=1, incremental_output="!", finished=True
    """
    uid: int
    incremental_output: str
    finished: bool
