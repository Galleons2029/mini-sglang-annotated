"""
decode.py - Decode 阶段管理

管理正在进行 decode（自回归生成）的请求集合。

主要职责：
1. 维护 running_reqs 集合（可继续 decode 的请求）
2. 调度 decode batch
3. 计算 inflight_tokens（用于 prefill 预算估算）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """
    Decode 管理器

    跟踪所有正在生成的请求。
    inflight_tokens 包括剩余输出长度 + 每个请求 1 page 的预留空间，
    用于帮助 prefill 阶段估算可用的 KV Cache 容量。
    """
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
