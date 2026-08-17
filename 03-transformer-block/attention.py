import math
import torch
import torch.nn as nn
from torch import Tensor


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor):
    """
    Transformer의 Scaled-Dot Product Attnetion 구현
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        query: (..., T_q, D_k) 차원
        key: (..., T_k, D_k) 차원
        value: (..., T_k, D_v) 차원
    Returns:
        output
        attention_weight
    """
    assert query.size(-1) == key.size(-1)
    d_k = query.size(-1)

    attention_score = (query @ key.transpose(-1, -2)) / math.sqrt(d_k) # (..., T_q, T_k)
    attention_weight = attention_score.softmax(dim=-1) # (..., T_q, T_k)
    output = attention_weight @ value                  # (..., T_q, D_v)

    return output, attention_weight


if __name__ == "__main__":
    batch_size, len_q, len_k, dim_k, dim_v = 2, 3, 4, 5, 6
    query = torch.randn((batch_size, len_q, dim_k), requires_grad=True)
    key = torch.randn((batch_size, len_k, dim_k), requires_grad=True)
    value = torch.randn((batch_size, len_k, dim_v), requires_grad=True)

    output, attention_weight = scaled_dot_product_attention(query, key, value)
    assert output.shape == (batch_size, len_q, dim_v)
    assert attention_weight.shape == (batch_size, len_q, len_k)
    assert torch.allclose(attention_weight.sum(dim=-1), torch.ones_like(attention_weight.sum(dim=-1)))

    output.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()