
import torch
import torch.nn as nn
import torch.nn.functional as F

from .SelfAttention import MultiHeadSelfAttention
from .FeedForwardNetwork import FeedForwardNet
from .LayerNormalization import LayerNormal1D

class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, dropout = 0.7, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(DecoderBlock, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MultiHeadSelfAttention(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,dropout, self.device)
        self.ffn = FeedForwardNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,dropout,self.device)
        
        self.layer_norm1 = LayerNormal1D(n_embeddings,device=self.device)
        self.layer_norm2 = LayerNormal1D(n_embeddings,device=self.device)
        
    def forward(self, x, y=None):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x
