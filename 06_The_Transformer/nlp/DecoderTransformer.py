import torch
import torch.nn as nn
import torch.nn.functional as F

from .DecoderBlock import DecoderBlock

class NxBlocks(nn.Module):
    def __init__(self, vocab_size, block_size, Nx = 2 , ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, dropout = 0.7,device = 'cpu', nonlinearity ='gelu', n_gram = 2):
        super(NxBlocks, self).__init__()
        self.Nx = Nx
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    ffn_hid_dim=ffn_hid_dim,
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    dropout=dropout,
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
    

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size, block_size, Nx = 2 , ffn_hid_dim = 258, n_heads = 8, n_embeddings =128, dropout = 0.7, device = 'cpu', non_linearity ='gelu', n_gram = 2):
        super(DecoderOnlyTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_embeddings = n_embeddings
        self.ffn_hid_dim = ffn_hid_dim
        self.dropout = dropout
        self.n_gram = n_gram
        self.non_linearity = non_linearity
        self.context =  - self.n_gram  + 1
        self.device = device
        
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embeddings)
        self.position_embeddings = nn.Embedding(self.block_size, self.n_embeddings)

        self.stacked_blocks = NxBlocks(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    Nx=Nx,
                    ffn_hid_dim=ffn_hid_dim,
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    device=device,
                    nonlinearity=non_linearity,
                    dropout=dropout,
                    n_gram=n_gram
                    )
        
        self.lang_modelling_head = nn.Linear(self.n_embeddings, self.vocab_size)

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


    def generate(self, x, n_pred):
        for _ in range(n_pred):
            logits = self(x[:,-self.block_size:])[:, self.context, :]
            prob_dist = F.softmax(logits, -1)
            x = torch.cat([x, torch.multinomial(prob_dist, 1)], -1).to(self.device)
        return x