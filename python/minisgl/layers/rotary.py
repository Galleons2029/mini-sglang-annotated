"""
rotary.py - 旋转位置编码 (RoPE) 实现

本模块实现 Rotary Position Embedding (RoPE)，
这是现代 LLM（如 Llama、Qwen）使用的位置编码方式。

RoPE 原理：
- 将位置信息编码到 Q 和 K 的旋转角度中
- 相对位置信息通过 Q·K 的点积自然保留
- 支持外推（超出训练长度的位置）

数学形式：
- 对于位置 m 的 token：
  Q_rotated = Q * cos(m*θ) + rotate_half(Q) * sin(m*θ)
  K_rotated = K * cos(m*θ) + rotate_half(K) * sin(m*θ)
  
- θ = base^(-2i/d)，i 是维度索引，d 是 head_dim

RoPE Scaling (扩展上下文长度)：
- Llama3: 使用 low/high freq factor 平滑缩放
- YaRN: 通过频率插值 + beta_fast/beta_slow 边界扩展
- 低频（长波）部分缩放更多
- 高频（短波）部分保持不变

使用 FlashInfer 的 apply_rope_with_cos_sin_cache_inplace：
- 预计算 cos/sin cache
- 原地修改 Q、K，避免内存分配
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    """
    旋转位置编码层
    
    预计算 cos/sin cache，使用 FlashInfer kernel 高效计算。
    
    Attributes:
        head_size: 每个 head 的维度
        _cos_sin_cache: 预计算的 cos/sin 值 [max_len, head_size]
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        """
        初始化 RoPE
        
        Args:
            head_size: head 维度
            rotary_dim: 旋转维度（通常等于 head_size）
            max_position_embeddings: 最大位置
            base: 基数（通常 10000）
            post_process: 频率后处理函数（用于 rope scaling）
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        # 计算逆频率：θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 应用后处理（rope scaling）
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        
        # 计算所有位置的频率：freqs[m, i] = m * θ_i
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 预计算 cos 和 sin
        cos = freqs.cos()
        sin = freqs.sin()
        # 合并为 cache: [cos_0, cos_1, ..., sin_0, sin_1, ...]
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        # 使用 FlashInfer 的高效实现
        from flashinfer import apply_rope_with_cos_sin_cache_inplace
        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE（原地修改）
        
        Args:
            positions: token 位置 [total_tokens]
            query: Q 张量 [total_tokens, num_heads, head_dim]
            key: K 张量 [total_tokens, num_kv_heads, head_dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 旋转后的 Q 和 K
        """
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    """
    创建 RoPE 实例（内部函数）

    支持标准 RoPE、Llama3 style 缩放和 YaRN 缩放。
    """
    if rope_scaling is None:
        # 标准 RoPE
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    
    # 处理 RoPE 缩放
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            # Llama3 的分段缩放：
            # - 高频（短波长）：保持不变
            # - 低频（长波长）：缩放 1/factor
            # - 中间：平滑过渡
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # 计算波长
                wave_len = 2 * math.pi / inv_freq
                
                # 无平滑（low == high 时）
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,  # 高频不缩放
                        inv_freq / scaling_factor,  # 低频缩放
                    )

                # 平滑过渡
                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            # YaRN (Yet another RoPE extensioN):
            # 通过频率插值 + 温度缩放扩展上下文长度
            # beta_fast/beta_slow 控制高频/低频的缩放边界
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


# 全局 RoPE 设备（用于 meta device 场景）
_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """设置 RoPE 计算设备"""
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    """
    获取 RoPE 实例（缓存）
    
    使用 lru_cache 确保相同配置复用同一实例。
    
    Args:
        head_dim: head 维度
        rotary_dim: 旋转维度
        max_position: 最大位置
        base: 基数
        rope_scaling: 缩放配置（tuple of tuples 用于 hashable）
    
    Returns:
        RotaryEmbedding: RoPE 实例
    """
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # meta device 不能用于 RoPE
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]