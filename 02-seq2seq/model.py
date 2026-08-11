import random
import torch
import torch.nn as nn
from torch import Tensor

from lstm import LSTM, LSTMCell
from data import synthetic_reverse_dataset, PAD_TOKEN, BOS_TOKEN


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_size, padding_idx=PAD_TOKEN)
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size)

    def forward(self, source: Tensor, valid_mask: Tensor):
        """
        Args:
            source (LongTensor): (batch_size, seq_len) 크기의 원본 시퀀스
            valid_mask (BoolTensor): (batch_size, seq_len), 실제 source가 위치한 인덱스를 표시
        Returns:
            outputs (FloatTensor): 모든 timestep의 Encoder hidden state (batch_size, seq_len, hidden_size)
            hidden (FloatTensor): 각 sequence의 마지막 유효 hidden state (batch_size, hidden_size)
            cell (FloatTensor): 각 sequence의 마지막 유효 cell state (batch_size, hidden_size)
        """

        # 1. Source Token을 Embedding으로 변환
        embedded = self.embedding(source) # (B, S) -> (B, S, E)
        # 2. 각 배치의 실제 길이 구하기
        lengths = valid_mask.sum(dim=1) # (B, S) -> (B,)
        # 3. LSTM 실행
        outputs, (hidden, cell) = self.lstm(embedded, lengths=lengths)

        return outputs, (hidden, cell)


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_size, padding_idx=PAD_TOKEN)
        self.lstm_cell = LSTMCell(input_size=embedding_size, hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size) 

    def forward(self, token: Tensor, hidden: Tensor, cell: Tensor):
        """
        Args:
            token (Tensor): 현재 시점의 token id, (batch_size,)
            hidden (Tensor): 이전 시점의 hidden state, (batch_size, hidden_size)
            cell (Tensor): 이전 시점의 cell state, (batch_size, hidden_size)
        Returns:
            logit (Tensor): 다음 token의 raw logits, (batch_size, vocab_size)
            h_next (Tensor): 갱신된 hidden state, (batch_size, hidden_size)
            c_next (Tensor): 갱신된 cell state, (batch_size, hidden_size)
        """
        # 1. Token을 Embedding으로 변환
        embedded = self.embedding(token) # (B, ) -> (B, E)
        # 2. LSTMCell 연산
        h_next, c_next = self.lstm_cell(embedded, h_prev=(hidden, cell))
        # 3. logit 계산
        logit = self.linear(h_next) # (B, H) -> (B, V)

        return logit, (h_next, c_next)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Seq2SeqEncoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size
        )
        self.decoder = Seq2SeqDecoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size
        )

    def forward(self, source: Tensor, target_input: Tensor, valid_mask: Tensor):
        """
        Args:
            source (LongTensor): (batch_size, seq_len) 크기의 원본 시퀀스
            target_input (LongTensor): (batch_size, seq_len+1) 크기의 [BOS 토큰 + 정답 시퀀스]
            valid_mask (BoolTensor): (batch_size, seq_len), 실제 source가 위치한 인덱스를 표시
        Returns:
            logits (FloatTensor): (batch_size, seq_len+1, vocab_size) 크기의 전체 logit
        """
        _, T = target_input.shape
        _, (h_x, c_x) = self.encoder(source=source, valid_mask=valid_mask)

        logits = []
        for t in range(T):
            logit, (h_x, c_x) = self.decoder(token=target_input[:,t], hidden=h_x, cell=c_x)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)

        return logits

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 50)
    print("1. Seq2Seq Encoder & Decoder Test")

    batch_size, min_seq, max_seq = 4, 3, 8
    source, target_input, target_output, valid_mask = \
        synthetic_reverse_dataset(batch_size, min_seq, max_seq)

    vocab_size, emb_size, hidden_size = 13, 8, 16
    encoder = Seq2SeqEncoder(vocab_size=vocab_size, embedding_size=emb_size, hidden_size=hidden_size)
    decoder = Seq2SeqDecoder(vocab_size=vocab_size, embedding_size=emb_size, hidden_size=hidden_size)

    encoder_output, (hidden, cell) = encoder(source, valid_mask)

    bos_tokens = target_input[:, 0]
    assert torch.all(bos_tokens == BOS_TOKEN)

    logits, (h_next, c_next) = decoder(bos_tokens, hidden, cell)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(h_next).all()
    assert torch.isfinite(c_next).all()

    answer = target_output[:, 0]
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    loss = loss_fn(logits, answer)
    assert torch.isfinite(loss).all()

    loss.backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for name, parameter in decoder.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    print("=" * 50)
    print("2. Seq2Seq Model Test")
    seq_to_seq = Seq2Seq(vocab_size=vocab_size, embedding_size=emb_size, hidden_size=hidden_size)
    logits = seq_to_seq.forward(source=source, target_input=target_input, valid_mask=valid_mask)

    assert torch.isfinite(logits).all()
    assert logits.shape == (batch_size, target_input.shape[1], vocab_size)

    logits = logits.reshape(-1, vocab_size)
    target_output_flatten = target_output.reshape(-1)
    loss = loss_fn(logits, target_output_flatten)
    assert torch.isfinite(loss).all()

    loss.backward()
    for name, parameter in seq_to_seq.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    
