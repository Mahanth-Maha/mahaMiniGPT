import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, n_embs , head_size, block_size, vocab_size, dropout = 0.7, device = 'cpu'):
        super(SingleHeadSelfAttention, self).__init__()
        self.n_embds = n_embs
        self.head_size = head_size
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.q = nn.Linear(self.n_embds, self.head_size , bias=False)
        self.k = nn.Linear(self.n_embds, self.head_size , bias=False)
        self.v = nn.Linear(self.n_embds, self.head_size , bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(B, T, self.head_size)
        K = K.view(B, T, self.head_size)
        V = V.view(B, T, self.head_size)

        w = (Q @ K.transpose(-2,-1) )/ (self.head_size ** 0.5)
        w = w.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)

        w = self.dropout_layer(w)

        out = w @ V
        out = out.view(B, T, self.head_size)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads,n_embs , head_size, block_size, vocab_size, dropout = 0.7, device = 'cpu'):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.n_embds = n_embs
        self.each_head_size = head_size
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(
                self.n_embds,
                self.each_head_size,
                self.block_size,
                self.vocab_size,
                dropout,
                device
            )  for _ in range(self.n_heads)]
        )
        
        self.proj = nn.Linear(n_embs,n_embs)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self,x):
        x = torch.cat(
            [
                shs_attn(x) for shs_attn in self.heads
            ], 
            dim = -1
        )
        return self.dropout_layer(self.proj(x))
