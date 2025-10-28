
import torch
import torch.nn as nn
import torch.nn.functional as F

class NGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size ,n_embeddings =128, device = 'cpu', n_gram = 2):
        super(NGramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.n_embeddings = n_embeddings
        self.n_gram = n_gram
        self.context =  - self.n_gram  + 1

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embeddings)
        self.position_embeddings = nn.Embedding(self.block_size, self.n_embeddings)
        self.lang_modelling_head = nn.Linear(self.n_embeddings, self.vocab_size)

    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(self.device))
        embeds = token_embeddings + pos_embeddings

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