import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        """
        Args:
            d_model (int): 모델 내부에서의 Hidden Dimension (=임베딩 차원)
            max_seq_len (int): 모델이 한 번에 처리할 수 있는 문장의 최대 길이
        """
        super().__init__()
        assert d_model % 2 == 0 # d_model이 짝수일 때만 SinusoidalPE가 정상적으로 작동

        pos = torch.arange(0, max_seq_len).unsqueeze(-1) # (max_seq_len, 1)
        freq = 10000 ** (2*torch.arange(0, d_model//2)/d_model) # (d_model//2,)

        pe = torch.zeros((max_seq_len, d_model))
        pe[:, 0::2] = torch.sin(pos / freq)
        pe[:, 1::2] = torch.cos(pos / freq)
        pe = pe.reshape(1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    print("=" * 50)
    print("1. Sinusoidal Positional Encoding Testing")
    batch_size, seq_len, d_model, max_len = 2, 4, 6, 10
    positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.zeros(
        (batch_size, seq_len, d_model),
        requires_grad=True,
    )

    output = positional_encoding(x)

    assert output.shape == x.shape
    assert positional_encoding.pe.shape == (1, max_len, d_model)
    assert "pe" in dict(positional_encoding.named_buffers())
    assert not positional_encoding.pe.requires_grad
    torch.testing.assert_close(
        output,
        positional_encoding.pe[:, :seq_len].expand_as(output),
    )
    torch.testing.assert_close(output[0], output[1])

    expected_position_zero = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    torch.testing.assert_close(output[0, 0], expected_position_zero)

    frequencies = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    expected_position_one = torch.empty(d_model)
    expected_position_one[0::2] = torch.sin(frequencies)
    expected_position_one[1::2] = torch.cos(frequencies)
    torch.testing.assert_close(output[0, 1], expected_position_one)

    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    print("Test Complete!!!")
