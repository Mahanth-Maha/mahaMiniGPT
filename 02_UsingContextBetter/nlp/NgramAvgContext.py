
import torch
import torch.nn as nn
import torch.nn.functional as F
from .NgramModel import NGramLanguageModel

class AverageContext(nn.Module):
    def __init__(self, block_size, device):
        super(AverageContext, self).__init__()
        self.block_size = block_size
        self.device = device
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size,self.block_size)))
    
    def forward(self, x):
        B,T,C = x.shape

        w = torch.ones((T,T)).to(self.device)
        w = w.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        
        out = w @ x
        
        out = out.view(B, T, C)
        return out


class NGramAverageContext(NGramLanguageModel):

    def __init__(self, vocab_size, block_size ,n_embeddings =128, device = 'cpu', n_gram = 2):
        super(NGramAverageContext, self).__init__( vocab_size, block_size ,n_embeddings = n_embeddings, device = device, n_gram = n_gram)
        
        self.avg_ctx = AverageContext(block_size,device)

    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(self.device))
        embeds = token_embeddings + pos_embeddings

        embeds = self.avg_ctx(embeds)
        
        logits = self.lang_modelling_head(embeds)

        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss
