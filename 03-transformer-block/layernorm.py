import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((d_model))) # gamma는 1로 초기화
        self.beta = nn.Parameter(torch.zeros((d_model))) # beta는 0으로 초기화
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            output: (..., D)
        """
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True, correction=0) # Layernorm은 모집단 분산을 사용하므로 `correction=0`
        norm_x = (x - mu) / torch.sqrt(sigma**2 + self.eps)
        output = self.gamma * norm_x + self.beta
        return output

    # 왜 LayerNorm은 모집단 분포?
    # 한 토큰의 feature 전체가 정규화할 대상 그 자체이기 때문!

def test_forward_matches_pytorch() -> None:
    torch.manual_seed(42)

    d_model = 8
    layer_norm = LayerNorm(d_model)
    reference = nn.LayerNorm(d_model)

    with torch.no_grad():
        for parameter, reference_parameter in zip(
            layer_norm.parameters(),
            reference.parameters(),
            strict=True,
        ):
            parameter.copy_(reference_parameter)

    x = torch.randn(2, 4, d_model)

    torch.testing.assert_close(layer_norm(x), reference(x))


def test_backward_matches_pytorch() -> None:
    torch.manual_seed(42)

    d_model = 8
    layer_norm = LayerNorm(d_model)
    reference = nn.LayerNorm(d_model)

    with torch.no_grad():
        for parameter, reference_parameter in zip(
            layer_norm.parameters(),
            reference.parameters(),
            strict=True,
        ):
            parameter.copy_(reference_parameter)

    x = torch.randn(2, 4, d_model, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(x)

    layer_norm(x).backward(grad_output)
    reference(reference_x).backward(grad_output)

    torch.testing.assert_close(x.grad, reference_x.grad)
    for parameter, reference_parameter in zip(
        layer_norm.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)


if __name__ == "__main__":
    test_forward_matches_pytorch()
    test_backward_matches_pytorch()
    print("Test Complete!!!")
