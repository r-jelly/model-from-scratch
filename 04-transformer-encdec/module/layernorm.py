import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((d_model)))
        self.beta = nn.Parameter(torch.zeros((d_model)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            output: (..., D)
        """
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True, correction=0)
        norm_x = (x - mu) / torch.sqrt(sigma**2 + self.eps)
        output = self.gamma * norm_x + self.beta
        return output