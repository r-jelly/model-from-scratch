import torch
import torch.nn as nn

from module.attention import MultiHeadAttention, create_causal_mask
from module.feedforward import FeedForward
from module.layernorm import LayerNorm
from module.positional_encoding import SinusoidalPositionalEncoding
from module.embedding import Embedding


class DecoderBlock(nn.Module):
    def __init__(self, emb_size: int, n_heads: int):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model=emb_size, num_heads=n_heads)
        self.cross_attention = MultiHeadAttention(d_model=emb_size, num_heads=n_heads)
        self.ffn = FeedForward(d_model=emb_size, d_ff=4*emb_size)
        self.norm1 = LayerNorm(d_model=emb_size)
        self.norm2 = LayerNorm(d_model=emb_size)
        self.norm3 = LayerNorm(d_model=emb_size)

    def forward(self, x, enc_output, enc_attn_mask=None, dec_attn_mask=None):
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len=seq_len)
        
        # 1. Masked Multi-head Attention
        out1, _ = self.masked_mha(
            query=x, key=x, value=x,
            attn_mask=causal_mask & dec_attn_mask if dec_attn_mask is not None else causal_mask
        )
        out1 = self.norm1(x + out1)

        # 2. Cross Attention
        out2, _ = self.cross_attention(
            query=out1,
            key=enc_output,
            value=enc_output,
            attn_mask=enc_attn_mask
        )
        out2 = self.norm2(out1 + out2)

        # 3. Feed-forward Network
        out3 = self.ffn(out2)
        out3 = self.norm3(out2 + out3)

        return out3