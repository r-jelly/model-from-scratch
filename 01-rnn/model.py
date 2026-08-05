from typing import Optional, Tuple
from torch import Tensor
import torch
import torch.nn as nn

from rnn import RNN
from lstm import LSTM

__all__ = ["ManyToOneClassifier", "CharacterLM"]

class ManyToOneClassifier(nn.Module):
    """
    Recurrent 모듈을 이용해서 Many-to-One Classification을 수행하는 모듈
    """
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, 
                x_seq: Tensor, 
                initial_state: Tensor | Tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """
        Args:
            x_seq (Tensor): 입력 시퀀스 (batch_size, seq_len, input_size)
            initial_state (Tensor | Tuple[Tensor, Tensor]): RNN, LSTM의 초기 state 정보 (batch_size, hidden_size)
        Returns:
            preds (Tensor): 최종적으로 예측한 각 클래스의 logit 값
        """
        is_batched = True
        if x_seq.dim() > 3 or x_seq.dim() < 2:
            raise ValueError()
        elif x_seq.dim() == 2:
            is_batched = False
            x_seq = x_seq.unsqueeze(0)

        output, _ = self.backbone(x_seq, initial_state)
        preds = self.fc_layer(output[:, -1, :])

        if not is_batched:
            preds = preds.squeeze(0)

        return preds


class CharacterLM(nn.Module):
    """
    Recurrent 모듈을 이용해서 Many-to-Many Language Modeling을 수행하는 모듈
    """
    def __init__(
            self,
            backbone: nn.Module,
            vocab_size: int,
            emb_size: int,
            hidden_size: int,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size
        )
        self.backbone = backbone
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
            self,
            token_ids: Tensor,
            initial_state: Tensor | Tuple[Tensor, Tensor] | None = None
    ) -> Tuple[Tensor, Tensor | Tuple[Tensor, Tensor]]:
        """
        Args:
            x_seq (LongTensor): 입력된 시퀀스에 해당하는 각 토큰의 ID 값 (batch_size, seq_len)
            initial_state (Tensor | Tuple[Tensor, Tensor]): RNN, LSTM의 초기 state 정보 (batch_size, hidden_size)
        Returns:
            preds (Tensor): 각 토큰 당 예측한 다음 토큰의 예측값
            hidden_state (Tensor | Tuple[Tensor, Tensor]): 마지막 토큰의 hidden state 값
        """

        # token_id는 정수여야 함
        assert token_ids.dtype == torch.long

        embedded = self.embedding(token_ids)
        output, hidden_state = self.backbone(embedded, initial_state)
        preds = self.fc(output)

        return preds, hidden_state


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    input_size = 4
    hidden_size = 8
    num_classes = 3

    x_seq = torch.randn(batch_size, seq_len, input_size)

    print("=" * 30)
    print("1. ManyToOneClassifier with RNN")

    rnn_classifier = ManyToOneClassifier(
        RNN(input_size=input_size, hidden_size=hidden_size),
        hidden_size=hidden_size,
        num_classes=num_classes,
    )
    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    torch_fc = nn.Linear(hidden_size, num_classes)
    h_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        rnn_classifier.backbone.rnn_cell.W_xh.copy_(torch_rnn.weight_ih_l0)
        rnn_classifier.backbone.rnn_cell.W_hh.copy_(torch_rnn.weight_hh_l0)
        rnn_classifier.backbone.rnn_cell.b_xh.copy_(torch_rnn.bias_ih_l0)
        rnn_classifier.backbone.rnn_cell.b_hh.copy_(torch_rnn.bias_hh_l0)
        rnn_classifier.fc_layer.weight.copy_(torch_fc.weight)
        rnn_classifier.fc_layer.bias.copy_(torch_fc.bias)

        # Batched input with an explicit hidden state
        my_preds = rnn_classifier(x_seq, h_prev)
        torch_output, _ = torch_rnn(x_seq, h_prev.unsqueeze(0))
        torch_preds = torch_fc(torch_output[:, -1, :])
        assert my_preds.shape == (batch_size, num_classes)
        torch.testing.assert_close(my_preds, torch_preds)

        # Unbatched input without an explicit hidden state
        my_preds = rnn_classifier(x_seq[0])
        torch_output, _ = torch_rnn(x_seq[0])
        torch_preds = torch_fc(torch_output[-1, :])
        assert my_preds.shape == (num_classes,)
        torch.testing.assert_close(my_preds, torch_preds)

    rnn_classifier(x_seq).sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in rnn_classifier.parameters()
    )

    print("RNN classifier implementation complete!!!")
    print("=" * 30)

    print("2. ManyToOneClassifier with LSTM")

    lstm_classifier = ManyToOneClassifier(
        LSTM(input_size=input_size, hidden_size=hidden_size),
        hidden_size=hidden_size,
        num_classes=num_classes,
    )
    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    torch_fc = nn.Linear(hidden_size, num_classes)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        lstm_classifier.backbone.lstm_cell.W_xh.copy_(torch_lstm.weight_ih_l0)
        lstm_classifier.backbone.lstm_cell.W_hh.copy_(torch_lstm.weight_hh_l0)
        lstm_classifier.backbone.lstm_cell.b_xh.copy_(torch_lstm.bias_ih_l0)
        lstm_classifier.backbone.lstm_cell.b_hh.copy_(torch_lstm.bias_hh_l0)
        lstm_classifier.fc_layer.weight.copy_(torch_fc.weight)
        lstm_classifier.fc_layer.bias.copy_(torch_fc.bias)

        # Batched input with explicit hidden and cell states
        my_preds = lstm_classifier(x_seq, (h_prev, c_prev))
        torch_output, _ = torch_lstm(
            x_seq,
            (h_prev.unsqueeze(0), c_prev.unsqueeze(0)),
        )
        torch_preds = torch_fc(torch_output[:, -1, :])
        assert my_preds.shape == (batch_size, num_classes)
        torch.testing.assert_close(my_preds, torch_preds)

        # Unbatched input without explicit hidden/cell states
        my_preds = lstm_classifier(x_seq[0])
        torch_output, _ = torch_lstm(x_seq[0])
        torch_preds = torch_fc(torch_output[-1, :])
        assert my_preds.shape == (num_classes,)
        torch.testing.assert_close(my_preds, torch_preds)

    lstm_classifier(x_seq).sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in lstm_classifier.parameters()
    )

    print("LSTM classifier implementation complete!!!")
    print("=" * 30)

    vocab_size = 11
    emb_size = input_size
    token_ids = torch.randint(vocab_size, (batch_size, seq_len))

    print("3. CharacterLM with RNN")

    rnn_lm = CharacterLM(
        RNN(input_size=emb_size, hidden_size=hidden_size),
        vocab_size=vocab_size,
        emb_size=emb_size,
        hidden_size=hidden_size,
    )
    torch_embedding = nn.Embedding(vocab_size, emb_size)
    torch_rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
    torch_fc = nn.Linear(hidden_size, vocab_size)
    h_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        rnn_lm.embedding.weight.copy_(torch_embedding.weight)
        rnn_lm.backbone.rnn_cell.W_xh.copy_(torch_rnn.weight_ih_l0)
        rnn_lm.backbone.rnn_cell.W_hh.copy_(torch_rnn.weight_hh_l0)
        rnn_lm.backbone.rnn_cell.b_xh.copy_(torch_rnn.bias_ih_l0)
        rnn_lm.backbone.rnn_cell.b_hh.copy_(torch_rnn.bias_hh_l0)
        rnn_lm.fc.weight.copy_(torch_fc.weight)
        rnn_lm.fc.bias.copy_(torch_fc.bias)

        # Batched token IDs with an explicit hidden state
        my_logits, my_hidden = rnn_lm(token_ids, h_prev)
        torch_output, torch_hidden = torch_rnn(
            torch_embedding(token_ids),
            h_prev.unsqueeze(0),
        )
        torch_logits = torch_fc(torch_output)
        assert my_logits.shape == (batch_size, seq_len, vocab_size)
        torch.testing.assert_close(my_logits, torch_logits)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))

        # Unbatched token IDs without an explicit hidden state
        my_logits, my_hidden = rnn_lm(token_ids[0])
        torch_output, torch_hidden = torch_rnn(torch_embedding(token_ids[0]))
        torch_logits = torch_fc(torch_output)
        assert my_logits.shape == (seq_len, vocab_size)
        torch.testing.assert_close(my_logits, torch_logits)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))

    rnn_lm(token_ids)[0].sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in rnn_lm.parameters()
    )

    print("RNN language model implementation complete!!!")
    print("=" * 30)

    print("4. CharacterLM with LSTM")

    lstm_lm = CharacterLM(
        LSTM(input_size=emb_size, hidden_size=hidden_size),
        vocab_size=vocab_size,
        emb_size=emb_size,
        hidden_size=hidden_size,
    )
    torch_embedding = nn.Embedding(vocab_size, emb_size)
    torch_lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
    torch_fc = nn.Linear(hidden_size, vocab_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        lstm_lm.embedding.weight.copy_(torch_embedding.weight)
        lstm_lm.backbone.lstm_cell.W_xh.copy_(torch_lstm.weight_ih_l0)
        lstm_lm.backbone.lstm_cell.W_hh.copy_(torch_lstm.weight_hh_l0)
        lstm_lm.backbone.lstm_cell.b_xh.copy_(torch_lstm.bias_ih_l0)
        lstm_lm.backbone.lstm_cell.b_hh.copy_(torch_lstm.bias_hh_l0)
        lstm_lm.fc.weight.copy_(torch_fc.weight)
        lstm_lm.fc.bias.copy_(torch_fc.bias)

        # Batched token IDs with explicit hidden and cell states
        my_logits, (my_hidden, my_cell) = lstm_lm(token_ids, (h_prev, c_prev))
        torch_output, (torch_hidden, torch_cell) = torch_lstm(
            torch_embedding(token_ids),
            (h_prev.unsqueeze(0), c_prev.unsqueeze(0)),
        )
        torch_logits = torch_fc(torch_output)
        assert my_logits.shape == (batch_size, seq_len, vocab_size)
        torch.testing.assert_close(my_logits, torch_logits)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))
        torch.testing.assert_close(my_cell, torch_cell.squeeze(0))

        # Unbatched token IDs without explicit hidden/cell states
        my_logits, (my_hidden, my_cell) = lstm_lm(token_ids[0])
        torch_output, (torch_hidden, torch_cell) = torch_lstm(
            torch_embedding(token_ids[0])
        )
        torch_logits = torch_fc(torch_output)
        assert my_logits.shape == (seq_len, vocab_size)
        torch.testing.assert_close(my_logits, torch_logits)
        torch.testing.assert_close(my_hidden, torch_hidden.squeeze(0))
        torch.testing.assert_close(my_cell, torch_cell.squeeze(0))

    lstm_lm(token_ids)[0].sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in lstm_lm.parameters()
    )

    print("LSTM language model implementation complete!!!")
    print("=" * 30)
