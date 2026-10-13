import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            padding_idx=0,
        )
        # padding_idx의 의미란?
        # 해당 인덱스 위치의 임베딩 값을 0으로 반환
        # 임베딩 조회 경로의 gradient 값은 0이 됨
        
        nn.init.normal_(self.embedding.weight, std=1/(d_model**0.5))
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * self.d_model**0.5