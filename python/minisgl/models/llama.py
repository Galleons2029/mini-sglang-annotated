"""
llama.py - Llama 模型实现

本模块实现 Llama 系列模型的架构，包括：
- LlamaDecoderLayer: 单个 Transformer 解码器层
- LlamaModel: Llama 主模型（不含 LM Head）
- LlamaForCausalLM: 完整的因果语言模型（含 LM Head）

Llama 架构特点：
1. Pre-RMSNorm: 在 attention 和 FFN 之前进行归一化
2. RoPE: 旋转位置编码
3. SwiGLU: 门控线性单元激活函数
4. GQA: 分组查询注意力（Grouped-Query Attention）

模型结构：
┌──────────────────────────────────────────┐
│              Embedding                    │
└────────────────┬─────────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│         Decoder Layer × N                 │
│  ┌────────────────────────────────────┐  │
│  │  RMSNorm → Self-Attention → +      │  │
│  │             ↓                      │  │
│  │  RMSNorm → FFN (SwiGLU) → +        │  │
│  └────────────────────────────────────┘  │
└────────────────┬─────────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│              RMSNorm                      │
└────────────────┬─────────────────────────┘
                 ↓
┌──────────────────────────────────────────┐
│              LM Head                      │
└──────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    """
    Llama 解码器层
    
    实现单个 Transformer 解码器层，包含：
    1. Self-Attention（带 RoPE）
    2. FFN（SwiGLU）
    3. 两个 RMSNorm
    
    使用 Pre-Norm 结构（先归一化再计算）和残差连接。
    
    数据流：
    input → RMSNorm → Attention → + → RMSNorm → FFN → + → output
              ↑                   |      ↑              |
              └─────── residual ──┘      └── residual ──┘
    
    Attributes:
        self_attn: 自注意力层（含 RoPE）
        mlp: 前馈网络（SwiGLU）
        input_layernorm: attention 前的 RMSNorm
        post_attention_layernorm: FFN 前的 RMSNorm
    """
    
    def __init__(self, config: ModelConfig, layer_id: int):
        """
        初始化解码器层
        
        Args:
            config: 模型配置
            layer_id: 层索引（用于 NVTX 标记和 KV Cache 索引）
        """
        self.self_attn = LlamaAttn(config, layer_id)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        使用融合的 RMSNorm 实现高效的残差计算。
        
        Args:
            x: 输入张量，shape: [seq_len, hidden_size]
            residual: 残差张量（第一层为 None）
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (输出, 新的残差)
        """
        # 第一个子层：Self-Attention
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        # 第二个子层：FFN
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseOP):
    """
    Llama 主模型（不含 LM Head）
    
    包含：
    1. Token Embedding
    2. N 个 Decoder Layer
    3. 最终的 RMSNorm
    
    Attributes:
        embed_tokens: 词嵌入层（支持 Tensor Parallel）
        layers: 解码器层列表
        norm: 最终的 RMSNorm
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入 token ID，shape: [seq_len]
        
        Returns:
            torch.Tensor: 隐藏状态，shape: [seq_len, hidden_size]
        """
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # 最后一层的输出需要加上 residual
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """
    Llama 因果语言模型
    
    完整的 Llama 模型，用于文本生成任务。
    
    结构：
    input_ids → LlamaModel → LM Head → logits
    
    Attributes:
        model: Llama 主模型
        lm_head: 语言模型头（输出 vocabulary 维度的 logits）
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化因果语言模型
        
        Args:
            config: 模型配置
        """
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """
        前向传播
        
        从全局 Context 获取当前批次的 input_ids。
        
        Returns:
            torch.Tensor: logits，shape: [seq_len, vocab_size]
        """
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
