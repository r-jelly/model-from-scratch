from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanilaRNN(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq: torch.tensor, h_0: Optional[torch.tensor] = None):
        """
        Args:
            x_seq: (B, seq_len, emb_size)
            h0: (hidden_size,)
        Return:
            (B, seq_len, hidden_size): 입력 시퀀스 내 모든 위치에서의 hidden state 
            (B, hidden_size): 입력 시퀀스 마지막 위치에서의 hidden state
        """
        B, seq_len, emb_size = x_seq.shape
        if h_0 is not None:
            h_t_prev = h_0
        else:
            h_t_prev = torch.zeros_like(self.b)

        h_stack = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :] 
            h_t = torch.tanh(x_t@self.W_xh.T + h_t_prev@self.W_hh.T + self.b)
            h_stack.append(h_t)
            h_t_prev = h_t

        return torch.stack(h_stack, dim=1), h_t
    

if __name__ == "__main__":
    batch_size, input_size, hidden_size, seq_len = 2, 4, 8, 5

    my_rnn = VanilaRNN(input_size, hidden_size)
    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    x_seq = torch.randn(batch_size, seq_len, input_size)
    my_rnn_forward = my_rnn.forward(x_seq)
    torch_rnn_forward = torch_rnn.forward(x_seq)

    assert my_rnn_forward[0].shape == torch_rnn_forward[0].shape

    my_rnn.W_xh.data.copy_(torch_rnn.weight_ih_l0.data)
    my_rnn.W_hh.data.copy_(torch_rnn.weight_hh_l0.data)
    my_rnn.b.data.copy_(torch_rnn.bias_ih_l0.data + torch_rnn.bias_hh_l0.data)

    with torch.no_grad():
        my_rnn_forward = my_rnn.forward(x_seq)
        torch_rnn_forward = torch_rnn.forward(x_seq)
    
    assert torch.allclose(my_rnn_forward[0], torch_rnn_forward[0])