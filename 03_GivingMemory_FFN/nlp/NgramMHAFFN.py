

import torch
import torch.nn as nn
import torch.nn.functional as F
from .NgramModel import NGramLanguageModel
from .NgramMultiHeadSelfAttention import MHSAttention
from .FeedForwardNetwork import FFNRelu, FFNet

class NGramMHAFFNReLU(NGramLanguageModel):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', n_gram = 2):
        super(NGramMHAFFNReLU, self).__init__( vocab_size, block_size ,n_embeddings = n_embeddings, device = device, n_gram = n_gram)
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        self.attention = MHSAttention(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNRelu(n_embeddings,ffn_hid_dim,n_embeddings,device)
    
    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(self.device))
        embeds = token_embeddings + pos_embeddings

        embeds = self.attention(embeds)
        
        embeds = self.ffn(embeds)
        
        logits = self.lang_modelling_head(embeds)

        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss


class NGramMHAFFN(NGramLanguageModel):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu',nonlinearity ='gelu', n_gram = 2):
        super(NGramMHAFFN, self).__init__( vocab_size, block_size ,n_embeddings = n_embeddings, device = device, n_gram = n_gram)
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        self.attention = MHSAttention(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,device)
    
    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(self.device))
        embeds = token_embeddings + pos_embeddings

        embeds = self.attention(embeds)
        
        embeds = self.ffn(embeds)
        
        logits = self.lang_modelling_head(embeds)

        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss
