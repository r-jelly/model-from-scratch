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
