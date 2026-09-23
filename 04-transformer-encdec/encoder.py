import torch
import torch.nn as nn

from module.attention import MultiHeadAttention
from module.feedforward import FeedForward
from module.layernorm import LayerNorm
from module.positional_encoding import SinusoidalPositionalEncoding
from module.embedding import Embedding


class EncoderBlock(nn.Module):
    def __init__(self, emb_size: int, n_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=emb_size, num_heads=n_heads)
        self.ffn = FeedForward(d_model=emb_size, d_ff=4*emb_size)
        self.norm1 = LayerNorm(d_model=emb_size)
        self.norm2 = LayerNorm(d_model=emb_size)

    def forward(self, x, attn_mask=None):
        out, _ = self.mha(query=x, key=x, value=x, attn_mask=attn_mask)
        out = self.norm1(x + out) # Post-LN: LayerNorm(x + Sublayer(x))

        y = self.ffn(out)
        y = self.norm2(out + y)

        return y


class Encoder(nn.Module):
    def __init__(self, emb_layer: Embedding, d_model, num_heads, max_seq_len):
        super().__init__()
        self.embedding = emb_layer
        self.pe = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads) for _ in range(6)
        ])
        

    def forward(self, token_ids, attn_mask=None):
        # 1. Embedding + Positional Encoding
        x = self.embedding(token_ids)
        x = self.pe(x)
        
        # 2. Transformer Block * 6
        for block in self.encoder_blocks:
            x = block(x, attn_mask=attn_mask)

        # 3. output
        return x


if __name__ == "__main__":
    vocab_size, d_model, n_heads, max_seq_len = 10, 8, 4, 16
    emb_layer = Embedding(vocab_size, d_model)
    encoder = Encoder(emb_layer, d_model, n_heads, max_seq_len)
    token_ids = torch.randint(0, 10, size=(max_seq_len, )).unsqueeze(0)
    output = encoder(token_ids)
    assert output.shape == (1, max_seq_len, d_model)