from typing import Optional, Tuple
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size

        # (i, f, g, o)의 4가지 gate의 parameter를 한 번에 계산하기 위함
        self.W_xh = nn.Parameter(torch.randn(hidden_size*4, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size*4, hidden_size))
        self.b_xh = nn.Parameter(torch.randn(4*hidden_size))
        self.b_hh = nn.Parameter(torch.randn(4*hidden_size))

        # Sigmoid/Tanh와 같이 대칭성을 갖는 activation 사용시 Xavier uniform 사용
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        이전 시점의 (hidden state, cell state)와 현재 시점에서의 input을 입력받아,
        현재 시점의 (hidden state, cell state)를 반환하는 함수

        Args:
            x_t (torch.Tensor): 현재 시점의 입력값 (batch_size, input_size)
            h_prev (Tuple[torch.Tensor, torch.Tensor]): 이전 시점의 (hidden, cell)값 (batch_size, hidden_size)
        Returns:
            h_next (Tuple[torch.Tensor, torch.Tensor]): 현재 시점의 (hidden, cell)값 (batch_size, hidden_size)
        """
        is_batched = True
        if x_t.dim() > 2 or x_t.dim() < 1:
            raise ValueError("LSTMCell: Expected input to be 1D or 2D.")
        elif x_t.dim() == 1:
            is_batched = False
            x_t = x_t.unsqueeze(0)

        batch_size = x_t.size(0)
        if h_prev is None:
            h_prev = (
                torch.zeros((batch_size, self.hidden_size), dtype=x_t.dtype, device=x_t.device), 
                torch.zeros((batch_size, self.hidden_size), dtype=x_t.dtype, device=x_t.device)
            )
        h_t, c_t = h_prev

        affine = x_t @ self.W_xh.T + self.b_xh + h_t @ self.W_hh.T + self.b_hh
        i, f, g, o = affine.chunk(4, dim=-1)

        f_t = torch.sigmoid(f)
        i_t = torch.sigmoid(i)
        g_t = torch.tanh(g)
        o_t = torch.sigmoid(o)

        c_next = f_t * c_t + i_t * g_t
        h_next = o_t * torch.tanh(c_next)

        if not is_batched:
            h_next = h_next.squeeze(0)
            c_next = c_next.squeeze(0)

        return (h_next, c_next)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)

    def forward(self,
                x_seq: torch.Tensor,
                h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        전체 입력 Sequence에 대해서 RNNCell 연산을 수행

        Args:
            x_seq (torch.Tensor): 전체 input sequence (batch_size, seq_len, input_size)
            h_prev (Tuple[torch.Tensor, torch.Tensor]): 이전 시점의 (hidden, cell)값 (batch_size, hidden_size)
        Returns:
            output (torch.Tensor): 모든 timestep에서의 output 계산 결과 (batch_size, seq_len, hidden_size)
            h_next (Tuple[torch.Tensor, torch.Tensor]): 마지막 timestep의 (hidden, cell)값 (batch_size, hidden_size)
        """
        is_batched = True
        if x_seq.dim() > 3 or x_seq.dim() < 2:
            raise ValueError()
        elif x_seq.dim() == 2:
            is_batched = False
            x_seq = x_seq.unsqueeze(0)

        batch_size, seq_len, _ = x_seq.shape
        if h_prev is None:
            h_prev = (
                torch.zeros((batch_size, self.hidden_size), dtype=x_seq.dtype, device=x_seq.device),
                torch.zeros((batch_size, self.hidden_size), dtype=x_seq.dtype, device=x_seq.device),
            )

        output = []
        for i in range(seq_len):
            h_next, c_next = self.lstm_cell(x_seq[:, i, :], h_prev)
            output.append(h_next)
            h_prev = (h_next, c_next)

        output = torch.stack(output, dim=1)
        if not is_batched:
            output = output.squeeze(0)
            h_next = h_next.squeeze(0)
            c_next = c_next.squeeze(0)
        return output, (h_next, c_next)


if __name__ == "__main__":
    batch_size, input_size, hidden_size = 2, 4, 8

    print("=" * 30)
    print("1. LSTMCell Implementation")

    my_lstm = LSTMCell(input_size, hidden_size)
    torch_lstm = nn.LSTMCell(input_size, hidden_size)

    x_t = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        my_lstm.W_xh.copy_(torch_lstm.weight_ih)
        my_lstm.W_hh.copy_(torch_lstm.weight_hh)
        my_lstm.b_xh.copy_(torch_lstm.bias_ih)
        my_lstm.b_hh.copy_(torch_lstm.bias_hh)

        # Batched input with explicit hidden and cell states
        my_h_next, my_c_next = my_lstm(x_t, (h_prev, c_prev))
        torch_h_next, torch_c_next = torch_lstm(x_t, (h_prev, c_prev))
        torch.testing.assert_close(my_h_next, torch_h_next)
        torch.testing.assert_close(my_c_next, torch_c_next)

        # Both states can be omitted
        my_h_next, my_c_next = my_lstm(x_t)
        torch_h_next, torch_c_next = torch_lstm(x_t)
        torch.testing.assert_close(my_h_next, torch_h_next)
        torch.testing.assert_close(my_c_next, torch_c_next)

        # Unbatched input and states are also supported
        my_h_next, my_c_next = my_lstm(x_t[0], (h_prev[0], c_prev[0]))
        torch_h_next, torch_c_next = torch_lstm(x_t[0], (h_prev[0], c_prev[0]))
        torch.testing.assert_close(my_h_next, torch_h_next)
        torch.testing.assert_close(my_c_next, torch_c_next)

    print("LSTMCell implementation complete!!!")
    print("=" * 30)

    print("2. LSTM Implementation")

    seq_len = 5
    my_lstm = LSTM(input_size, hidden_size)
    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    my_parameters = list(my_lstm.parameters())
    torch_parameters = list(torch_lstm.parameters())
    assert len(my_parameters) == len(torch_parameters)

    x_seq = torch.randn(batch_size, seq_len, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        for my_parameter, torch_parameter in zip(my_parameters, torch_parameters):
            my_parameter.copy_(torch_parameter)

        my_output, (my_hidden, my_cell) = my_lstm(x_seq, (h_prev, c_prev))
        torch_output, (torch_hidden, torch_cell) = torch_lstm(
            x_seq,
            (h_prev.unsqueeze(0), c_prev.unsqueeze(0)),
        )
        torch.testing.assert_close(my_output, torch_output)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))
        torch.testing.assert_close(my_cell, torch_cell.squeeze(0))

        my_output, (my_hidden, my_cell) = my_lstm(x_seq)
        torch_output, (torch_hidden, torch_cell) = torch_lstm(x_seq)
        torch.testing.assert_close(my_output, torch_output)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))
        torch.testing.assert_close(my_cell, torch_cell.squeeze(0))

        # Unbatched sequences and hidden/cell states are also supported
        my_output, (my_hidden, my_cell) = my_lstm(x_seq[0], (h_prev[0], c_prev[0]))
        torch_output, (torch_hidden, torch_cell) = torch_lstm(
            x_seq[0],
            (h_prev[0].unsqueeze(0), c_prev[0].unsqueeze(0)),
        )
        torch.testing.assert_close(my_output, torch_output)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))
        torch.testing.assert_close(my_cell, torch_cell.squeeze(0))

        my_output, (my_hidden, my_cell) = my_lstm(x_seq[0])
        torch_output, (torch_hidden, torch_cell) = torch_lstm(x_seq[0])
        torch.testing.assert_close(my_output, torch_output)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))
        torch.testing.assert_close(my_cell, torch_cell.squeeze(0))

    print("LSTM implementation complete!!!")
    print("=" * 30)
