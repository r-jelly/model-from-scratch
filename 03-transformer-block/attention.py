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


if __name__ == "__main__":
    batch_size, len_q, len_k, dim_k, dim_v = 2, 3, 4, 5, 6
    query = torch.randn((batch_size, len_q, dim_k), requires_grad=True)
    key = torch.randn((batch_size, len_k, dim_k), requires_grad=True)
    value = torch.randn((batch_size, len_k, dim_v), requires_grad=True)

    print("=" * 50)
    print("1. Scaled Dot-Product Attention Testing")
    output, attention_weight = scaled_dot_product_attention(query, key, value)
    assert output.shape == (batch_size, len_q, dim_v)
    assert attention_weight.shape == (batch_size, len_q, len_k)
    assert torch.allclose(attention_weight.sum(dim=-1), torch.ones_like(attention_weight.sum(dim=-1)))

    output.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()
    print("Test Complete!!!")

    print("=" * 50)
    print("2. Attention Mask Testing")
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
    print("Test Complete!!!")

    print("=" * 50)
    print("3. Create Causal Masking Function Testing")
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
    print("Test Complete!!!")

    print("=" * 50)
    print("4. Compare with Pytorch Scaled Dot-Product Attention")
    batch_size, num_head, len_q, len_k, dim_k, dim_v = 2, 3, 4, 5, 8, 6
    query = torch.randn((batch_size, num_head, len_q, dim_k), requires_grad=True)
    key = torch.randn((batch_size, num_head, len_k, dim_k), requires_grad=True)
    value = torch.randn((batch_size, num_head, len_k, dim_v), requires_grad=True)

    my_attention, _ = scaled_dot_product_attention(query, key, value)
    torch_attention = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

    assert my_attention.shape == (batch_size, num_head, len_q, dim_v)
    assert torch_attention.shape == (batch_size, num_head, len_q, dim_v)
    torch.testing.assert_close(torch_attention, my_attention)

    padding_mask = torch.ones(
      (batch_size, 1, 1, len_k),
      dtype=torch.bool,
    )
    padding_mask[1, :, :, -2:] = False

    my_masked_output, my_masked_weight = scaled_dot_product_attention(
        query,
        key,
        value,
        padding_mask,
    )
    torch_masked_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=padding_mask,
        dropout_p=0.0,
    )

    torch.testing.assert_close(my_masked_output, torch_masked_output)

    expanded_mask = padding_mask.expand_as(my_masked_weight)
    assert torch.all(my_masked_weight[~expanded_mask] == 0)
    assert torch.allclose(
        my_masked_weight.sum(dim=-1),
        torch.ones_like(my_masked_weight.sum(dim=-1)),
    )

    causal_len = 4
    causal_query = torch.randn(
        (batch_size, num_head, causal_len, dim_k),
    )
    causal_key = torch.randn(
        (batch_size, num_head, causal_len, dim_k),
    )
    causal_value = torch.randn(
        (batch_size, num_head, causal_len, dim_v),
    )
    causal_mask = create_causal_mask(causal_len)

    my_causal_output, my_causal_weight = scaled_dot_product_attention(
        causal_query,
        causal_key,
        causal_value,
        causal_mask,
    )
    torch_causal_output = F.scaled_dot_product_attention(
        causal_query,
        causal_key,
        causal_value,
        dropout_p=0.0,
        is_causal=True,
    )

    torch.testing.assert_close(my_causal_output, torch_causal_output)

    expanded_causal_mask = causal_mask.expand_as(my_causal_weight)
    assert torch.all(my_causal_weight[~expanded_causal_mask] == 0)

    causal_weight_sum = my_causal_weight.sum(dim=-1)
    torch.testing.assert_close(
        causal_weight_sum,
        torch.ones_like(causal_weight_sum),
    )
    print("Test Complete!!!")

    print("=" * 50)
    print("5. Multi-Head Attention Testing")
    batch_size, num_heads, len_q, len_k, d_model = 2, 3, 4, 5, 12
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn((batch_size, len_q, d_model), requires_grad=True)
    key = torch.randn((batch_size, len_k, d_model), requires_grad=True)
    value = torch.randn((batch_size, len_k, d_model), requires_grad=True)

    output, attention_weight = multi_head_attention(query, key, value)

    assert output.shape == (batch_size, len_q, d_model)
    assert attention_weight.shape == (batch_size, num_heads, len_q, len_k)
    torch.testing.assert_close(
        attention_weight.sum(dim=-1),
        torch.ones_like(attention_weight.sum(dim=-1)),
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_weight).all()

    output.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()
    print("Test Complete!!!")

    print("=" * 50)
    print("6. Multi-Head Attention Padding Mask Testing")
    padding_mask = torch.ones(
        (batch_size, 1, 1, len_k),
        dtype=torch.bool,
    )
    padding_mask[1, :, :, -2:] = False

    masked_output, masked_attention_weight = multi_head_attention(
        query,
        key,
        value,
        padding_mask,
    )

    expanded_mask = padding_mask.expand_as(masked_attention_weight)
    assert torch.all(masked_attention_weight[~expanded_mask] == 0)
    torch.testing.assert_close(
        masked_attention_weight.sum(dim=-1),
        torch.ones_like(masked_attention_weight.sum(dim=-1)),
    )
    assert torch.isfinite(masked_output).all()
    print("Test Complete!!!")

    print("=" * 50)
    print("7. Multi-Head Attention Causal Mask Testing")
    causal_len = 4
    causal_query = torch.randn((batch_size, causal_len, d_model))
    causal_key = torch.randn((batch_size, causal_len, d_model))
    causal_value = torch.randn((batch_size, causal_len, d_model))
    causal_mask = create_causal_mask(causal_len)

    causal_output, causal_attention_weight = multi_head_attention(
        causal_query,
        causal_key,
        causal_value,
        causal_mask,
    )

    expanded_causal_mask = causal_mask.expand_as(causal_attention_weight)
    assert torch.all(causal_attention_weight[~expanded_causal_mask] == 0)
    torch.testing.assert_close(
        causal_attention_weight.sum(dim=-1),
        torch.ones_like(causal_attention_weight.sum(dim=-1)),
    )
    assert torch.isfinite(causal_output).all()
    print("Test Complete!!!")

    print("=" * 50)
    print("8. Compare with Pytorch Multi-Head Attention")
    torch_multi_head_attention = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,
    )

    with torch.no_grad():
        torch_multi_head_attention.in_proj_weight.copy_(
            torch.cat([
                multi_head_attention.q_proj.weight,
                multi_head_attention.k_proj.weight,
                multi_head_attention.v_proj.weight,
            ])
        )
        torch_multi_head_attention.in_proj_bias.copy_(
            torch.cat([
                multi_head_attention.q_proj.bias,
                multi_head_attention.k_proj.bias,
                multi_head_attention.v_proj.bias,
            ])
        )
        torch_multi_head_attention.out_proj.weight.copy_(
            multi_head_attention.o_proj.weight
        )
        torch_multi_head_attention.out_proj.bias.copy_(
            multi_head_attention.o_proj.bias
        )

    my_output, my_attention_weight = multi_head_attention(query, key, value)
    torch_output, torch_attention_weight = torch_multi_head_attention(
        query,
        key,
        value,
        need_weights=True,
        average_attn_weights=False,
    )
    torch.testing.assert_close(my_output, torch_output)
    torch.testing.assert_close(my_attention_weight, torch_attention_weight)

    my_masked_output, my_masked_attention_weight = multi_head_attention(
        query,
        key,
        value,
        padding_mask,
    )
    torch_masked_output, torch_masked_attention_weight = (
        torch_multi_head_attention(
            query,
            key,
            value,
            key_padding_mask=~padding_mask[:, 0, 0, :],
            need_weights=True,
            average_attn_weights=False,
        )
    )
    torch.testing.assert_close(my_masked_output, torch_masked_output)
    torch.testing.assert_close(
        my_masked_attention_weight,
        torch_masked_attention_weight,
    )

    my_causal_output, my_causal_attention_weight = multi_head_attention(
        causal_query,
        causal_key,
        causal_value,
        causal_mask,
    )
    torch_causal_output, torch_causal_attention_weight = (
        torch_multi_head_attention(
            causal_query,
            causal_key,
            causal_value,
            attn_mask=~causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )
    )
    torch.testing.assert_close(my_causal_output, torch_causal_output)
    torch.testing.assert_close(
        my_causal_attention_weight,
        torch_causal_attention_weight,
    )
    print("Test Complete!!!")
