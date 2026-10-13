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
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, enc_output, enc_attn_mask=None, dec_attn_mask=None):
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len=seq_len).to(x.device)
        
        # 1. Masked Multi-head Attention
        out1, _ = self.masked_mha(
            query=x, key=x, value=x,
            attn_mask=causal_mask & dec_attn_mask if dec_attn_mask is not None else causal_mask
        ) # causal_mask와 dec_attn_mask를 동시에 적용해야 함
        out1 = self.norm1(x + self.dropout(out1))

        # 2. Cross Attention
        out2, _ = self.cross_attention(
            query=out1,
            key=enc_output,
            value=enc_output,
            attn_mask=enc_attn_mask
        )
        out2 = self.norm2(out1 + self.dropout(out2))

        # 3. Feed-forward Network
        out3 = self.ffn(out2)
        out3 = self.norm3(out2 + self.dropout(out3))

        return out3


class Decoder(nn.Module):
    def __init__(self, emb_layer: Embedding, vocab_size, d_model, num_heads, max_seq_len, num_blocks=6):
        super().__init__()
        self.embedding = emb_layer
        self.pe = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads) for _ in range(num_blocks)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.linear.weight = self.embedding.embedding.weight
        self.dropout = nn.Dropout(p=0.1)
        

    def forward(self, token_ids, enc_output, enc_attn_mask=None, dec_attn_mask=None):
        # 1. Embedding + Positional Encoding
        x = self.embedding(token_ids)
        x = self.dropout(self.pe(x))
        
        # 2. Transformer Block * # blocks
        for block in self.decoder_blocks:
            x = block(x, enc_output, enc_attn_mask, dec_attn_mask)

        # 3. Linear
        x = self.linear(x)

        return x