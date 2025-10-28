
import torch
import torch.nn as nn
import torch.nn.functional as F

from .NgramMultiHeadSelfAttention import MHSAttention, MHSAwProj, MHSAwProjwDropout
from .FeedForwardNetwork import FFNet, FFNetwDropout
from .LayerNormalization import LayerNormal1D

class TBlock(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(TBlock, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MHSAttention(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,device)
    
    def forward(self, x, y=None):
        x = self.attention(x)
        x = self.ffn(x)
        return x



class TBlockResiduals(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(TBlockResiduals, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MHSAttention(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,device)
    
    def forward(self, x, y=None):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x

class TBlockResidualsProj(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(TBlockResidualsProj, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MHSAwProj(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,device)
    
    def forward(self, x, y=None):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x

class TBlockRPostLN(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(TBlockRPostLN, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MHSAwProj(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,device)

        self.layer_norm1 = nn.LayerNorm(n_embeddings,device=self.device)
        self.layer_norm2 = nn.LayerNorm(n_embeddings,device=self.device)
        
    def forward(self, x, y=None):
        x = x + self.attention(x)
        x = self.layer_norm1(x)
        x = x + self.ffn(x)
        x = self.layer_norm2(x)
        return x

class TBlockRPreLN(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(TBlockRPreLN, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MHSAwProj(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,self.device)
        self.ffn = FFNet(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,device)
        
        self.layer_norm1 = nn.LayerNorm(n_embeddings,device=self.device)
        self.layer_norm2 = nn.LayerNorm(n_embeddings,device=self.device)
        
    def forward(self, x, y=None):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class TBlockRPreLNwDropout(nn.Module):
    def __init__(self, vocab_size, block_size, ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, dropout = 0.7, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(TBlockRPreLNwDropout, self).__init__()
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.attention = MHSAwProjwDropout(self.n_heads,n_embeddings, n_embeddings//n_heads, self.block_size, self.vocab_size,dropout, self.device)
        self.ffn = FFNetwDropout(nonlinearity,n_embeddings,ffn_hid_dim,n_embeddings,dropout,self.device)
        
        self.layer_norm1 = nn.LayerNorm(n_embeddings,device=self.device)
        self.layer_norm2 = nn.LayerNorm(n_embeddings,device=self.device)
        

        
    def forward(self, x, y=None):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x
