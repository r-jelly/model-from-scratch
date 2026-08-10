from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from torch import Tensor


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size

        # (i, f, g, o)의 4가지 gate의 parameter를 한 번에 계산하기 위함
        self.W_xh = nn.Parameter(torch.empty(hidden_size*4, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size*4, hidden_size))
        self.b_xh = nn.Parameter(torch.empty(4*hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4*hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, 
                x_t: Tensor, 
                h_prev: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        이전 시점의 (hidden state, cell state)와 현재 시점에서의 input을 입력받아,
        현재 시점의 (hidden state, cell state)를 반환하는 함수

        Args:
            x_t (Tensor): 현재 시점의 입력값 (batch_size, input_size)
            h_prev (Tuple[Tensor, Tensor]): 이전 시점의 (hidden, cell)값 (batch_size, hidden_size)
        Returns:
            h_next (Tuple[Tensor, Tensor]): 현재 시점의 (hidden, cell)값 (batch_size, hidden_size)
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
                x_seq: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None,
                lengths: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        전체 입력 Sequence에 대해서 LSTMCell 연산을 수행

        Args:
            x_seq (Tensor): 전체 input sequence (batch_size, seq_len, input_size)
            hx (Tuple[Tensor, Tensor]): 이전 시점의 (hidden, cell)값 (batch_size, hidden_size)
            lengths (Tensor): Batch 내 sequence의 길이 (batch 내 모든 seq 길이가 다른 경우 처리 위함)
        Returns:
            output (Tensor): 모든 timestep에서의 output 계산 결과 (batch_size, seq_len, hidden_size)
            h_next (Tuple[Tensor, Tensor]): 마지막 timestep의 (hidden, cell)값 (batch_size, hidden_size)
        """
        is_batched = True
        if x_seq.dim() > 3 or x_seq.dim() < 2:
            raise ValueError()
        elif x_seq.dim() == 2:
            is_batched = False
            x_seq = x_seq.unsqueeze(0)

        batch_size, seq_len, _ = x_seq.shape
        if hx is None:
            hx = (
                torch.zeros((batch_size, self.hidden_size), dtype=x_seq.dtype, device=x_seq.device),
                torch.zeros((batch_size, self.hidden_size), dtype=x_seq.dtype, device=x_seq.device),
            )

        # 만약 각 시퀀스의 길이가 없을때는, 모든 시퀀스의 길이가 같다고 가정
        if lengths is None:
            lengths = torch.LongTensor([x_seq.size(1)] * x_seq.size(0), device=x_seq.device)

        output = []
        for t in range(seq_len):
            # 현재 timestep이 sequence length를 넘지 않았는지 확인
            active = t < lengths
            active = active.reshape(-1, 1)

            h_prev, c_prev = hx
            h_next, c_next = self.lstm_cell(x_seq[:, t, :], hx)

            # active=True인 sequence만 다음 timestep에 전달
            h_next = torch.where(active, input=h_next, other=h_prev)
            c_next = torch.where(active, input=c_next, other=c_prev)

            output.append(h_next.where(active, other=0)) # active=False인 위치는 output=PAD
            hx = (h_next, c_next)

        output = torch.stack(output, dim=1)
        if not is_batched:
            output = output.squeeze(0)
            h_next = h_next.squeeze(0)
            c_next = c_next.squeeze(0)
        return output, (h_next, c_next)