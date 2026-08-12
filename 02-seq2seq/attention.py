import torch
import torch.nn as nn
from torch import Tensor


class BahdanauAttention(nn.Module):
    def __init__(
        self, 
        enc_hidden_size: int,
        dec_hidden_size: int,
        attention_size: int,
        *args, **kwargs
    ):
        """
        Args:
            enc_hidden_size (int): Encoder의 Hidden State 크기
            dec_hidden_size (int): Decoder의 Hidden State 크기
            attention_size (int): Attention Map 차원
        """
        super().__init__(*args, **kwargs)
        self.enc_proj = nn.Linear(in_features=enc_hidden_size, out_features=attention_size)
        self.dec_proj = nn.Linear(in_features=dec_hidden_size, out_features=attention_size)
        self.score_proj = nn.Linear(in_features=attention_size, out_features=1)

    def forward(self, enc_outputs: Tensor, dec_hidden: Tensor, valid_mask: Tensor):
        """
        Args:
            enc_outputs (FloatTensor): Encoder의 각 timestep별 Hidden State
            dec_hidden (FloatTensor): 이전 timestep t-1의 Hidden State
            valid_mask (BoolTensor): 배치 내 valid한 토큰의 위치
        """
        # 모든 Hidden State를 attention 차원으로 projection
        enc_matrix = self.enc_proj(enc_outputs) # (B, S, H_enc) -> (B, S, A)
        dec_matrix = self.dec_proj(dec_hidden) # (B, H_dec) -> (B, A)
        dec_matrix = dec_matrix.unsqueeze(dim=1) # (B, A) -> (B, 1, A)

        # Encoder, Decoder의 결과를 broadcasting으로 더함
        addition_matrix = enc_matrix + dec_matrix
        # Tanh 적용
        addition_matrix_tanh = torch.tanh(addition_matrix)
        # Encoder의 각 timestep에 대한 score 계산
        scores = self.score_proj(addition_matrix_tanh) # (B, S, A) -> (B, S, 1)
        scores = scores.squeeze(-1) # (B, S, 1) -> (B, S)

        # Masking 적용
        masked_scores = torch.where(
            condition=valid_mask,
            input=scores,
            other=float("-inf"),
        )
        # Softmax를 적용해 attention 분포 구함
        attention_weight = torch.softmax(masked_scores, dim=1) # (B, S)

        # Attention Value(=context)를 구함
        attention_value = torch.einsum('bs, bsh -> bh', attention_weight, enc_outputs) # (B, H_enc)

        return attention_value, attention_weight


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    encoder_hidden_size = 5
    decoder_hidden_size = 7
    attention_size = 3

    model = BahdanauAttention(
        enc_hidden_size=encoder_hidden_size,
        dec_hidden_size=decoder_hidden_size,
        attention_size=attention_size
    )
    encoder_output = torch.randn(
        (batch_size, seq_len, encoder_hidden_size),
        requires_grad=True
    )
    decoder_hidden = torch.randn(
        (batch_size, decoder_hidden_size),
        requires_grad=True
    )
    valid_mask = torch.BoolTensor([
        [True, True, True, True],
        [True, True, False, False]
    ])
    attention_value, attention_weight = model.forward(encoder_output, decoder_hidden, valid_mask)

    assert attention_value.shape == (batch_size, encoder_hidden_size)
    assert attention_weight.shape == (batch_size, seq_len)

    assert torch.allclose(attention_weight.sum(dim=1), torch.ones((batch_size)))
    assert torch.all(abs(attention_weight[~valid_mask]) < 1e-7)

    scalar = attention_value.sum()
    scalar.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert encoder_output.grad is not None
    assert decoder_hidden.grad is not None

    assert torch.isfinite(encoder_output.grad).all()
    assert torch.isfinite(decoder_hidden.grad).all()