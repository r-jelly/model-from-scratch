import math
import torch
import torch.nn as nn
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

    masked_query = torch.zeros((1, 2, 2))
    masked_key = torch.zeros((1, 3, 2))
    masked_value = torch.tensor([[[10.0], [20.0], [1000.0]]])
    attn_mask = torch.tensor([[[True, True, False]]])

    masked_output, masked_weight = scaled_dot_product_attention(
        masked_query,
        masked_key,
        masked_value,
        attn_mask,
    )

    expected_weight = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]])
    expected_output = torch.tensor([[[15.0], [15.0]]])
    torch.testing.assert_close(masked_weight, expected_weight)
    torch.testing.assert_close(masked_output, expected_output)
