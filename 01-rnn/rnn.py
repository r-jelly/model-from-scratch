from typing import Optional
import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_xh = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

        nn.init.kaiming_uniform_(self.W_xh)
        nn.init.kaiming_uniform_(self.W_hh)

    def forward(self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        이전 시점의 hidden vector와 현재 시점에서의 input vector를 받아, 현재 시점의 hidden vector를 반환

        Args:
            x_t (torch.Tensor): 현재 시점의 입력값 (batch_size, input_size)
            h_prev (torch.Tensor): 이전 시점의 hidden state (batch_size, hidden_size)
        Returns:
            h_next (torch.Tensor): 현재 시점의 hidden state (batch_size, hidden_size)
        """
        is_batched = True
        if x_t.dim() > 2 or x_t.dim() < 1:
            raise ValueError("RNNCell: Expected input to be 1D or 2D.")
        elif x_t.dim() == 1:
            is_batched = False
            x_t = x_t.unsqueeze(0)

        batch_size = x_t.size(0)
        if h_prev is None:
            h_prev = torch.zeros((batch_size, self.hidden_size), dtype=x_t.dtype, device=x_t.device)

        h_next = torch.tanh(
            x_t@self.W_xh.T + self.b_xh
            + h_prev@self.W_hh.T + self.b_hh
        )

        if not is_batched:
            h_next = h_next.squeeze(0)

        return h_next


if __name__ == "__main__":
    batch_size, input_size, hidden_size = 2, 4, 8

    print("=" * 30)
    print("1. RNNCell Implementation")

    my_rnn = RNNCell(input_size, hidden_size)
    torch_rnn = nn.RNNCell(input_size, hidden_size)

    x_t = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        my_rnn.W_xh.copy_(torch_rnn.weight_ih)
        my_rnn.W_hh.copy_(torch_rnn.weight_hh)
        my_rnn.b_xh.copy_(torch_rnn.bias_ih)
        my_rnn.b_hh.copy_(torch_rnn.bias_hh)

        # Batched input with an explicit hidden state
        my_rnn_forward = my_rnn(x_t, h_prev)
        torch_rnn_forward = torch_rnn(x_t, h_prev)
        torch.testing.assert_close(my_rnn_forward, torch_rnn_forward)

        # The hidden state can be omitted
        torch.testing.assert_close(my_rnn(x_t), torch_rnn(x_t))

        # Unbatched input and hidden state are also supported
        torch.testing.assert_close(my_rnn(x_t[0], h_prev[0]), torch_rnn(x_t[0], h_prev[0]))

    print("RNNCell implementation complete!!!")
    print("=" * 30)
