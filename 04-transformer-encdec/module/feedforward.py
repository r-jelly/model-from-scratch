import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model (int): 입력과 출력의 hidden dimension
            d_ff (int): Feed-Forward 내부 확장 dimension
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
        """
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        return out
