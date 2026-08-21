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

        # TODO

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
        """
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 50)
    print("1. Feed-Forward Shape Testing")
    batch_size, seq_len, d_model, d_ff = 2, 4, 8, 32
    feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
    x = torch.randn(
        (batch_size, seq_len, d_model),
        requires_grad=True,
    )

    output = feed_forward(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    print("Test Complete!!!")

    print("=" * 50)
    print("2. Feed-Forward Gradient Testing")
    output.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameters = list(feed_forward.parameters())
    assert len(parameters) > 0
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    print("Test Complete!!!")

    print("=" * 50)
    print("3. Feed-Forward Token Independence Testing")
    feed_forward.eval()

    original_x = torch.randn((batch_size, seq_len, d_model))
    changed_x = original_x.clone()
    changed_x[:, 0, :] += 10.0

    with torch.no_grad():
        original_output = feed_forward(original_x)
        changed_output = feed_forward(changed_x)

    torch.testing.assert_close(
        original_output[:, 1:, :],
        changed_output[:, 1:, :],
    )
    assert not torch.allclose(
        original_output[:, 0, :],
        changed_output[:, 0, :],
    )
    print("Test Complete!!!")
