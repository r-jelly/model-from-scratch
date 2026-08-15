import random
import torch
import torch.nn as nn
from torch import Tensor

from lstm import LSTM, LSTMCell
from attention import BahdanauAttention
from data import synthetic_reverse_dataset, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


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


class AttentionSeq2SeqDecoder(nn.Module):
    def __init__(
        self,  
        vocab_size: int,
        embedding_size: int,
        enc_hidden_size: int,
        dec_hidden_size: int,
        attention_size: int,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=PAD_TOKEN)
        self.attention = BahdanauAttention(
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
            attention_size=attention_size
        )
        self.lstm_cell = LSTMCell(embedding_size+enc_hidden_size, dec_hidden_size) 
        self.linear = nn.Linear(dec_hidden_size, vocab_size)

    def forward(self, token: Tensor, hidden: Tensor, cell: Tensor, enc_outputs: Tensor, valid_mask: Tensor):
        """
        Args:
            token (Tensor): 현재 시점의 token id, (batch_size,)
            hidden (Tensor): 이전 시점의 hidden state, (batch_size, dec_hidden_size)
            cell (Tensor): 이전 시점의 cell state, (batch_size, dec_hidden_size)
            enc_outputs (Tensor): Encoder의 모든 timestep에 대한 output, (batch_size, seq_len, enc_hidden_size)
            valid_mask (Tensor): batch 내 실제 토큰이 위치한 지점에 대한 masking (batch_size, seq_len)

        Returns:
            logit (Tensor): 다음 token의 raw logits, (batch_size, vocab_size)
            h_next (Tensor): 갱신된 hidden state, (batch_size, dec_hidden_size)
            c_next (Tensor): 갱신된 cell state, (batch_size, dec_hidden_size)
            attention_weights (Tensor): Encoder의 모든 timestep에서의 attention weight, (batch_size, seq_len)
        """
        # Attention 구하기
        attention_value, attention_weight = self.attention(
            enc_outputs=enc_outputs,
            dec_hidden=hidden,
            valid_mask=valid_mask
        )

        # Token을 Embedding으로 변환
        embedded = self.embedding(token) # (B, ) -> (B, E)
        # Embedding과 attention value(context)를 concat
        lstm_input = torch.concat((embedded, attention_value), dim=-1) # (B, E+H_enc)
        # LSTMCell 연산 (enc_hidden_size == dec_hidden_size라고 가정)
        h_next, c_next = self.lstm_cell(lstm_input, h_prev=(hidden, cell)) # (B, H_dec)
        # logit 계산
        logit = self.linear(h_next) # (B, H_dec) -> (B, V)

        return logit, (h_next, c_next), attention_weight


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

    def forward(self, source: Tensor, target_input: Tensor, valid_mask: Tensor, teacher_force_ratio: float = 1.0):
        """
        Args:
            source (LongTensor): (batch_size, seq_len) 크기의 원본 시퀀스
            target_input (LongTensor): (batch_size, seq_len+1) 크기의 [BOS 토큰 + 정답 시퀀스]
            valid_mask (BoolTensor): (batch_size, seq_len), 실제 source가 위치한 인덱스를 표시
            teacher_force_ratio (float): 전체 시퀀스에서 teacher forcing을 사용할 비율 (0.0-1.0 범위)
        Returns:
            logits (FloatTensor): (batch_size, seq_len+1, vocab_size) 크기의 전체 logit
        """
        _, T = target_input.shape
        _, (h_x, c_x) = self.encoder(source=source, valid_mask=valid_mask)

        logits = []
        cur_token = target_input[:, 0]
        for t in range(T):
            teacher_force_mask = random.random()
            logit, (h_x, c_x) = self.decoder(token=cur_token, hidden=h_x, cell=c_x)
            if t+1 < T:
                cur_token = target_input[:, t+1] if teacher_force_mask < teacher_force_ratio else logit.argmax(dim=-1)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)

        return logits

    def greedy_decode(self, source: Tensor, valid_mask: Tensor, max_new_tokens: int):
        """
        Args:
            source (LongTensor): (batch_size, seq_len) 크기의 원본 시퀀스
            valid_mask (BoolTensor): (batch_size, seq_len), 실제 source가 위치한 인덱스를 표시
            max_new_tokens (int): 새로 생성될 수 있는 토큰 수의 최대값
        Returns:
            token_ids (Tensor): (batch_size, generated_length) 크기의 생성된 토큰 ID들
        """
        assert max_new_tokens > 0

        batch_size = source.size(0)
        _, (h_x, c_x) = self.encoder(source=source, valid_mask=valid_mask)

        cur_token = torch.LongTensor([BOS_TOKEN] * batch_size).to(source.device)
        finished = torch.BoolTensor([False] * batch_size).to(source.device)

        token_ids = []
        for _ in range(max_new_tokens):
            logit, (h_x, c_x) = self.decoder(token=cur_token, hidden=h_x, cell=c_x)
            cur_token = logit.argmax(dim=-1)
            cur_token = cur_token.where(~finished, other=PAD_TOKEN)
            finished = finished | (cur_token == EOS_TOKEN)
            token_ids.append(cur_token)

            if torch.all(finished):
                break
        token_ids = torch.stack(token_ids, dim=1)
        return token_ids


class AttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        attention_size: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = Seq2SeqEncoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size
        )
        self.decoder = AttentionSeq2SeqDecoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            enc_hidden_size=hidden_size,
            dec_hidden_size=hidden_size,
            attention_size=attention_size
        )

    def forward(self, source: Tensor, target_input: Tensor, valid_mask: Tensor, teacher_force_ratio: float = 1.0):
        """
        Args:
            source (LongTensor): (batch_size, seq_len) 크기의 원본 시퀀스
            target_input (LongTensor): (batch_size, seq_len+1) 크기의 [BOS 토큰 + 정답 시퀀스]
            valid_mask (BoolTensor): (batch_size, seq_len), 실제 source가 위치한 인덱스를 표시
            teacher_force_ratio (float): 전체 시퀀스에서 teacher forcing을 사용할 비율 (0.0-1.0 범위)
        Returns:
            logits (FloatTensor): (batch_size, seq_len+1, vocab_size) 크기의 전체 logit
            attention_weights (FloatTensor): (batch_size, seq_len+1, seq_len) 크기의 전체 attention weight
        """
        _, T = target_input.shape
        enc_outputs, (h_x, c_x) = self.encoder(source=source, valid_mask=valid_mask)

        logits = []
        attention_weights = []
        cur_token = target_input[:, 0]
        for t in range(T):
            teacher_force_mask = random.random()
            logit, (h_x, c_x), attention_weight = \
                self.decoder(token=cur_token, hidden=h_x, cell=c_x, enc_outputs=enc_outputs, valid_mask=valid_mask)
            if t+1 < T:
                cur_token = target_input[:, t+1] if teacher_force_mask < teacher_force_ratio else logit.argmax(dim=-1)

            logits.append(logit)
            attention_weights.append(attention_weight)

        logits = torch.stack(logits, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return logits, attention_weights

    def greedy_decode(self, source: Tensor, valid_mask: Tensor, max_new_tokens: int):
        """
        Args:
            source (LongTensor): (batch_size, seq_len) 크기의 원본 시퀀스
            valid_mask (BoolTensor): (batch_size, seq_len), 실제 source가 위치한 인덱스를 표시
            max_new_tokens (int): 새로 생성될 수 있는 토큰 수의 최대값
        Returns:
            token_ids (Tensor): (batch_size, generated_length) 크기의 생성된 토큰 ID들
            attention_weights (Tensor): (batch_size, generated_length, seq_len) 크기의 attention weight
        """
        assert max_new_tokens > 0

        batch_size = source.size(0)
        enc_outputs, (h_x, c_x) = self.encoder(source=source, valid_mask=valid_mask)

        cur_token = torch.LongTensor([BOS_TOKEN] * batch_size).to(source.device)
        finished = torch.BoolTensor([False] * batch_size).to(source.device)

        token_ids = []
        attention_weights = []
        for _ in range(max_new_tokens):
            logit, (h_x, c_x), attention_weight = \
                self.decoder(token=cur_token, hidden=h_x, cell=c_x, enc_outputs=enc_outputs, valid_mask=valid_mask)
            cur_token = logit.argmax(dim=-1)
            cur_token = cur_token.where(~finished, other=PAD_TOKEN)
            finished = finished | (cur_token == EOS_TOKEN)

            token_ids.append(cur_token)
            attention_weights.append(attention_weight)

            if torch.all(finished):
                break
        token_ids = torch.stack(token_ids, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        return token_ids, attention_weights


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

    print("=" * 50)
    print("3. Seq2Seq Model Teacher Force Ratio Test")
    seq_to_seq = Seq2Seq(vocab_size=vocab_size, embedding_size=emb_size, hidden_size=hidden_size)
    logits = seq_to_seq.forward(source=source, target_input=target_input, valid_mask=valid_mask, teacher_force_ratio=0.0)

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

    print("=" * 50)
    print("4. Attention Seq2Seq Decoder Test")
    attention_size = 12
    attention_decoder = AttentionSeq2SeqDecoder(
        vocab_size=vocab_size,
        embedding_size=emb_size,
        enc_hidden_size=hidden_size,
        dec_hidden_size=hidden_size,
        attention_size=attention_size
    )
    logits, (h_next, c_next), attention_weights = attention_decoder(
        token=bos_tokens,
        hidden=hidden.detach(),
        cell=cell.detach(),
        enc_outputs=encoder_output.detach(),
        valid_mask=valid_mask
    )

    assert logits.shape == (batch_size, vocab_size)
    assert h_next.shape == c_next.shape == (batch_size, hidden_size)
    assert attention_weights.shape == source.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(h_next).all()
    assert torch.isfinite(c_next).all()
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size))
    assert torch.all(attention_weights[~valid_mask] == 0)

    print("=" * 50)
    print("5. Attention Seq2Seq Model Test")
    attention_seq_to_seq = AttentionSeq2Seq(
        vocab_size=vocab_size,
        embedding_size=emb_size,
        hidden_size=hidden_size,
        attention_size=attention_size
    )
    logits, attention_weights = attention_seq_to_seq(
        source=source,
        target_input=target_input,
        valid_mask=valid_mask,
        teacher_force_ratio=0.0
    )

    assert logits.shape == (batch_size, target_input.shape[1], vocab_size)
    assert attention_weights.shape == (batch_size, target_input.shape[1], source.shape[1])
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        torch.ones(batch_size, target_input.shape[1])
    )
    expanded_valid_mask = valid_mask.unsqueeze(1).expand_as(attention_weights)
    assert torch.all(attention_weights[~expanded_valid_mask] == 0)

    loss = loss_fn(logits.reshape(-1, vocab_size), target_output.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in attention_seq_to_seq.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    print("=" * 50)
    print("6. Greedy Decoding Test")
    with torch.no_grad():
        seq_to_seq.decoder.linear.weight.zero_()
        seq_to_seq.decoder.linear.bias.zero_()
        seq_to_seq.decoder.linear.bias[EOS_TOKEN] = 1
        attention_seq_to_seq.decoder.linear.weight.zero_()
        attention_seq_to_seq.decoder.linear.bias.zero_()
        attention_seq_to_seq.decoder.linear.bias[EOS_TOKEN] = 1

        token_ids = seq_to_seq.greedy_decode(source, valid_mask, max_new_tokens=target_input.shape[1])
        attention_token_ids, attention_weights = attention_seq_to_seq.greedy_decode(
            source,
            valid_mask,
            max_new_tokens=target_input.shape[1]
        )

    assert token_ids.shape == attention_token_ids.shape == (batch_size, 1)
    assert torch.all(token_ids == EOS_TOKEN)
    assert torch.all(attention_token_ids == EOS_TOKEN)
    assert attention_weights.shape == (batch_size, 1, source.shape[1])
