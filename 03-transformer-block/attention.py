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

    causal_mask = create_causal_mask(4)
    expected_causal_mask = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    torch.testing.assert_close(causal_mask, expected_causal_mask)

    causal_query = torch.zeros((1, 4, 2))
    causal_key = torch.zeros((1, 4, 2))
    causal_value = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    causal_output, causal_weight = scaled_dot_product_attention(
        causal_query,
        causal_key,
        causal_value,
        causal_mask,
    )

    expected_causal_weight = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]])
    expected_causal_output = torch.tensor([[[10.0], [15.0], [20.0], [25.0]]])
    torch.testing.assert_close(causal_weight, expected_causal_weight)
    torch.testing.assert_close(causal_output, expected_causal_output)
