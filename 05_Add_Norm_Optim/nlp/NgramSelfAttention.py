
import torch
import torch.nn as nn
import torch.nn.functional as F
from .NgramModel import NGramLanguageModel

class SingleHeadAttention(nn.Module):
    def __init__(self, n_embs , head_size, block_size, vocab_size, device = 'cpu'):
        super(SingleHeadAttention, self).__init__()
        self.n_embds = n_embs
        self.head_size = head_size
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.q = nn.Linear(self.n_embds, self.head_size , bias=False)
        self.k = nn.Linear(self.n_embds, self.head_size , bias=False)
        self.v = nn.Linear(self.n_embds, self.head_size , bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))


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

        out = w @ V
        out = out.view(B, T, self.head_size)
        return out


class SingleHeadAttentionwDropout(nn.Module):
    def __init__(self, n_embs , head_size, block_size, vocab_size, dropout = 0.7, device = 'cpu'):
        super(SingleHeadAttentionwDropout, self).__init__()
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

class NGramSelfAttention(NGramLanguageModel):

    def __init__(self, vocab_size, block_size ,n_embeddings =128, device = 'cpu', n_gram = 2):
        super(NGramSelfAttention, self).__init__( vocab_size, block_size ,n_embeddings = n_embeddings, device = device, n_gram = n_gram)
        
        self.attention = SingleHeadAttention(n_embeddings,n_embeddings , block_size, vocab_size ,device)

    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(self.device))
        embeds = token_embeddings + pos_embeddings

        embeds = self.attention(embeds)
        
        logits = self.lang_modelling_head(embeds)

        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss
