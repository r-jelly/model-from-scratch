import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask=None):
    """
    Transformer의 Scaled-Dot Product Attnetion 구현
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        query: (..., T_q, D_k) 차원
        key: (..., T_k, D_k) 차원
        value: (..., T_k, D_v) 차원
        attn_mask: attention score (..., T_q, T_k) 차원에 broadcast 가능한 shape, True인 위치에서 attention 연산 수행
    Returns:
        output: 최종 attention 연산 결과
        attention_weight
    """
    assert query.size(-1) == key.size(-1)
    d_k = query.size(-1)

    attention_score = (query @ key.transpose(-1, -2)) / math.sqrt(d_k) # (..., T_q, T_k)
    if attn_mask is not None:
        attention_score = attention_score.where(attn_mask, float('-inf'))
    attention_weight = attention_score.softmax(dim=-1) # (..., T_q, T_k)
    output = attention_weight @ value                  # (..., T_q, D_v)

    return output, attention_weight


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Causal Language Model을 위한 마스킹 텐서를 생성하는 함수
    생성되는 시점 이후의 토큰을 학습에서 참조하지 않도록 하는 역할

    Args:
        seq_len (int): 전체 시퀀스 길이
    Returns:
        causal_mask (BoolTensor): 시퀀스 내 timestep t 이후의 토큰은 사용하지 않도록 마스킹된 텐서
    """
    causal_mask = torch.fill(torch.empty((seq_len, seq_len), dtype=torch.bool), False)
    for i in range(seq_len):
        causal_mask[i, :i+1] = True
    return causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model (int): 모델 내부에서의 Hidden Dimension (=임베딩 차원)
            num_heads (int): 모델이 동시에 계산할 Attention 병렬 처리의 개수 (=Attention Head 수)
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)


    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor]=None):
        """
        Args:
            query: (B, T_q, D), Attention의 Query
            key: (B, T_k, D), Attention의 Key
            value: (B, T_k, D), Attention의 Value
            attn_mask: (B, H, T_q, T_k)에 broadcast 가능한 bool mask.
                       True는 attention 허용, False는 차단.
        Returns:
            output: (B, T_q, D) 크기의 Attention Output
            attention_weight: (B, H, T_q, T_k) 크기의 Attention Weight
        """
        # 1. Q/K/V를 Projection
        query = self.q_proj(query) # (B, T_q, D) -> (B, T_q, D)
        key = self.k_proj(key) # (B, T_k, D) -> (B, T_k, D)
        value = self.v_proj(value) # (B, T_k, D) -> (B, T_k, D)

        # 2. Head를 분리
        query = query.reshape(query.size(0), query.size(1), self.num_heads, self.head_dim) # (B, T_q, D) -> (B, T_q, H, D_h)
        key = key.reshape(key.size(0), key.size(1), self.num_heads, self.head_dim) # (B, T_k, D) -> (B, T_k, H, D_h)
        value = value.reshape(value.size(0), value.size(1), self.num_heads, self.head_dim) # (B, T_k, D) -> (B, T_k, H, D_h)

        # 3. Head 축 변환
        query = query.transpose(1, 2) # (B, T_q, H, D_h) -> (B, H, T_q, D_h)
        key = key.transpose(1, 2) # (B, T_k, H, D_h) -> (B, H, T_k, D_h)
        value = value.transpose(1, 2) # (B, T_k, H, D_h) -> (B, H, T_k, D_h)

        # 4. scaled dot-product attention 호출
        output, attention_weight = scaled_dot_product_attention(query, key, value, attn_mask) # (B, H, T_q, D_h), (B, H, T_q, T_k)

        # 5. output 변환
        output = output.transpose(1, 2) # (B, H, T_q, D_h) -> (B, T_q, H, D_h)
        output = output.reshape(output.size(0), output.size(1), -1) # (B, T_q, H, D_h) -> (B, T_q, D)
        output = self.o_proj(output)

        return output, attention_weight