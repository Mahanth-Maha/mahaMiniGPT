
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .NgramModel import NGramLanguageModel
from .NgramSelfAttention import SingleHeadAttention


class MHSAttention(nn.Module):
    def __init__(self, n_heads,n_embs , head_size, block_size, vocab_size, device = 'cpu'):
        super(MHSAttention, self).__init__()
        self.n_heads = n_heads
        
        self.n_embds = n_embs
        self.each_head_size = head_size
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.heads = nn.ModuleList([
            SingleHeadAttention(
                self.n_embds,
                self.each_head_size,
                self.block_size,
                self.vocab_size,
                device
            )  for _ in range(self.n_heads)]
        )
        
    
    def forward(self,x):
        return torch.cat(
            [
                shs_attn(x) for shs_attn in self.heads
            ], 
            dim = -1
        )


class NGramMHSelfAttention(NGramLanguageModel):

    def __init__(self, vocab_size, block_size, n_heads = 8, n_embeddings =128, device = 'cpu', n_gram = 2):
        super(NGramMHSelfAttention, self).__init__( vocab_size, block_size ,n_embeddings = n_embeddings, device = device, n_gram = n_gram)
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        self.attention = MHSAttention(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)

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
