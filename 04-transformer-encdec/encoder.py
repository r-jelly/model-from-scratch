import torch
import torch.nn as nn

from module.attention import MultiHeadAttention
from module.feedforward import FeedForward
from module.layernorm import LayerNorm
from module.positional_encoding import SinusoidalPositionalEncoding


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

