import torch
import torch.nn as nn
import torch.nn.functional as F

from .TBlock import TBlockResiduals
from .NgramModel import NGramLanguageModel

class StackedResidualBlocks(nn.Module):
    def __init__(self, vocab_size, block_size, Nx = 2 , ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(StackedResidualBlocks, self).__init__()
        self.Nx = Nx
        self.blocks = nn.ModuleList(
            [
                TBlockResiduals(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    ffn_hid_dim=ffn_hid_dim,
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    device=device,
                    nonlinearity=nonlinearity,
                    n_gram=n_gram
                ) for _ in range(Nx)
            ]
        )

    def forward(self, x, y=None):
        for block in self.blocks:
            x = block(x)
        return x
    

    

class NGramSRBlocks(NGramLanguageModel):

    def __init__(self, vocab_size, block_size, Nx = 2 , ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(NGramSRBlocks, self).__init__( vocab_size, block_size ,n_embeddings = n_embeddings, device = device, n_gram = n_gram)
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        
        self.stacked_blocks = StackedResidualBlocks(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    Nx=Nx,
                    ffn_hid_dim=ffn_hid_dim,
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    device=device,
                    nonlinearity=nonlinearity,
                    n_gram=n_gram
                    )

    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(self.device))
        embeds = token_embeddings + pos_embeddings

        embeds = self.stacked_blocks(embeds)
        
        logits = self.lang_modelling_head(embeds)

        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss


