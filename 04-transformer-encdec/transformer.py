import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from module.embedding import Embedding
from module.attention import create_causal_mask


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        vocab_size: int,
        num_blocks: int,
    ):
        super().__init__()
        self.emb_layer = Embedding(
            vocab_size=vocab_size,
            d_model=d_model
        )
        self.encoder = Encoder(
            emb_layer=self.emb_layer,
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks
        )
        self.decoder = Decoder(
            emb_layer=self.emb_layer,
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks
        )

    def forward(
        self,
        enc_token_ids,
        dec_token_ids,
        enc_attn_mask,
        dec_attn_mask
    ):
        # 1. Encoder 수행
        enc_output = self.encoder(
            token_ids=enc_token_ids, attn_mask=enc_attn_mask
        )

        # 2. Decoder 수행
        dec_output = self.decoder(
            token_ids=dec_token_ids,
            enc_output=enc_output,
            enc_attn_mask=enc_attn_mask,
            dec_attn_mask=dec_attn_mask
        )

        return dec_output


VOCAB_SIZE = 20


def build_test_model() -> Transformer:
    torch.manual_seed(42)
    model = Transformer(d_model=16, num_heads=4, max_seq_len=32, vocab_size=VOCAB_SIZE, num_blocks=4)
    model.eval()
    return model


def make_inputs(batch_size: int = 2, src_len: int = 7, tgt_len: int = 5):
    # 0은 padding, VOCAB_SIZE - 1은 "바꿔 넣을 토큰"으로 남겨둔다
    src_ids = torch.randint(1, VOCAB_SIZE - 1, (batch_size, src_len))
    tgt_ids = torch.randint(1, VOCAB_SIZE - 1, (batch_size, tgt_len))
    src_mask = torch.ones((batch_size, 1, 1, src_len), dtype=torch.bool)
    tgt_mask = torch.ones((batch_size, 1, 1, tgt_len), dtype=torch.bool)
    return src_ids, tgt_ids, src_mask, tgt_mask


def test_output_shape() -> None:
    """src/tgt 길이가 달라도 출력 logits가 (B, T_tgt, vocab_size)이고 유한한지 확인한다."""
    model = build_test_model()
    src_ids, tgt_ids, src_mask, tgt_mask = make_inputs()

    logits = model(src_ids, tgt_ids, src_mask, tgt_mask)

    assert logits.shape == (tgt_ids.size(0), tgt_ids.size(1), VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_weight_tying() -> None:
    """encoder 임베딩, decoder 임베딩, 출력 linear가 같은 weight를 공유하는지(3-way tying) 확인한다."""
    model = build_test_model()

    assert model.encoder.embedding is model.emb_layer
    assert model.decoder.embedding is model.emb_layer
    assert model.decoder.linear.weight is model.emb_layer.embedding.weight


def test_decoder_causality() -> None:
    """decoder가 미래 target 토큰을 참조하지 않는지(causal mask) 확인한다."""
    model = build_test_model()
    src_ids, tgt_ids, src_mask, tgt_mask = make_inputs()
    t = 3
    changed_tgt_ids = tgt_ids.clone()
    changed_tgt_ids[:, t] = VOCAB_SIZE - 1

    with torch.no_grad():
        logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
        changed_logits = model(src_ids, changed_tgt_ids, src_mask, tgt_mask)

    # t번째 target 토큰을 바꿔도 t 이전 위치의 출력은 그대로여야 한다
    torch.testing.assert_close(logits[:, :t], changed_logits[:, :t])
    assert not torch.allclose(logits[:, t], changed_logits[:, t])


def test_src_padding_mask() -> None:
    """src padding 토큰은 무시되고, 실제 src 토큰은 출력에 반영되는지 확인한다."""
    model = build_test_model()
    src_ids, tgt_ids, src_mask, tgt_mask = make_inputs()
    src_ids[1, -2:] = 0
    src_mask[1, :, :, -2:] = False

    padded_changed_src_ids = src_ids.clone()
    padded_changed_src_ids[1, -2:] = VOCAB_SIZE - 1
    real_changed_src_ids = src_ids.clone()
    real_changed_src_ids[1, 0] = VOCAB_SIZE - 1

    with torch.no_grad():
        logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
        padded_changed_logits = model(padded_changed_src_ids, tgt_ids, src_mask, tgt_mask)
        real_changed_logits = model(real_changed_src_ids, tgt_ids, src_mask, tgt_mask)

    # padding 위치의 source 토큰은 encoder self-attention과 cross-attention 모두에서 무시돼야 한다
    torch.testing.assert_close(logits, padded_changed_logits)
    # 실제 source 토큰이 바뀌면 cross-attention을 통해 출력이 달라져야 한다
    assert not torch.allclose(logits[1], real_changed_logits[1])


def test_tgt_padding_causal_mask() -> None:
    """decoder self-attention이 미래 위치와 target padding 위치를 함께 막는지(causal & padding mask 결합) 확인한다."""
    model = build_test_model()
    src_ids, tgt_ids, src_mask, tgt_mask = make_inputs()
    # 오른쪽 padding에서는 padding 행에서만 결합 효과가 드러나므로, padding 2개로 padding 행이 다른 padding key를 보는 경우를 만든다
    tgt_ids[1, -2:] = 0
    tgt_mask[1, :, :, -2:] = False

    attention_weights = []
    for block in model.decoder.decoder_blocks:
        block.masked_mha.register_forward_hook(
            lambda module, inputs, outputs: attention_weights.append(outputs[1])
        )
        # `register_forward_hook`
        # 모델의 forward pass 중에 특정한 layer의 input, output 값에 접근해 원하는 작업을 수행하게 할 수 있다.

    with torch.no_grad():
        model(src_ids, tgt_ids, src_mask, tgt_mask)

    # 미래도 padding도 아닌 key에서만 weight가 0보다 커야 한다
    expected_mask = create_causal_mask(tgt_ids.size(1)) & tgt_mask  # (B, 1, T, T)
    assert len(attention_weights) == len(model.decoder.decoder_blocks)
    for attention_weight in attention_weights:  # (B, H, T, T)
        assert torch.equal(attention_weight > 0, expected_mask.expand_as(attention_weight))


def test_gradient_flow() -> None:
    """backward 후 모든 파라미터에 유한한 gradient가 생기는지 확인한다."""
    model = build_test_model()
    model.train()
    src_ids, tgt_ids, src_mask, tgt_mask = make_inputs()

    model(src_ids, tgt_ids, src_mask, tgt_mask).sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


if __name__ == "__main__":
    test_output_shape()
    test_weight_tying()
    test_decoder_causality()
    test_src_padding_mask()
    test_tgt_padding_causal_mask()
    test_gradient_flow()
    print("Test Complete!!!")
