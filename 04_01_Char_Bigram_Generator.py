import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from datetime import datetime
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(f'[>] Device : {device}')

print(f'[>] Loading Data ...')
data = open('./data/wikitext/processed.txt', 'r').read()
print(f'[+] Data Loaded !')

chars = ''.join(sorted(list(set(data))))
vocab_size = len(chars) 
print(f'[>] Number of Unique Characters : {vocab_size}')

def enc(x, chars = chars):
    idxs = []
    for c in x:
        idxs.append(chars.index(c))
    return idxs

def dec(x, chars = chars):
    txt = ''
    for i in x:
        txt += chars[i]
    return txt

# import tiktoken 
# tik = tiktoken.get_encoding('gpt2')
# tik.encode('Mahanth Yalla')
# tik.decode(tik.encode('Mahanth Yalla'))

# ### select encoder
enc = enc
dec = dec
vocab_size = vocab_size

# # tiktoken
# enc = tik.encode
# dec = tik.decode
# vocab_size = tik.vocab_size

# development purpose, taking first 1M chars only 
data = data[:1000000]

print(f'[>] Encoding Data ...')
st  = datetime.now()
data = torch.tensor(enc(data), dtype=torch.long).to(device)
n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))
Xtr = data[:n1]
Xdev = data[n1:n2]
Xte = data[n2:]

Xtr.to(device)
Xdev.to(device)
Xte.to(device)

et = datetime.now()
print(f'[+] Data Encoded in {et - st} !')


class BiGramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(BiGramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        logits = self.token_embedding_table(x)
        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss

    def generate(self, x, n_pred):
        for _ in range(n_pred):
            logits = self(x)[:, -1, :]
            prob_dist = F.softmax(logits, -1)
            x = torch.cat([x, torch.multinomial(prob_dist, 1)], -1).to(device)
        return x

def get_batch(split):
    if split == 'train':
        X = Xtr
    elif split == 'dev':
        X = Xdev
    else:
        X = Xte
    start = np.random.randint(0, len(X) - block_size - 1, (batch_size,))
    X_batch = torch.stack([X[s:s + block_size] for s in start]).to(device)
    y_batch = torch.stack([X[s+1:s + block_size+1] for s in start]).to(device)
    return X_batch.to(device), y_batch.to(device)

def estimate_batch_loss(model):
    with torch.no_grad():
        X_batch, y_batch = get_batch('dev')
        logits, loss = model(X_batch, y_batch)
        return loss.item()


model = BiGramLanguageModel(vocab_size)
model = model.to(device)
alpha = 1e-3

batch_size = 64
block_size = 16
n_iters = 10000

train_model = True
from_scratch = False

os.makedirs('./saved_models', exist_ok=True)
model_exists = os.path.exists(f'./saved_models/04_01_bi_gram_Generator_{device}.pth')

if train_model:
    print(f'[>] Training ...')
    if not model_exists or from_scratch:
        print(f'[!] Training from scratch ...')
    else:
        # Load the model
        model.load_state_dict(torch.load(f'./saved_models/04_01_bi_gram_Generator_{device}.pth'))
        print('[+] Model loaded!')

    st  = datetime.now()
    optimiser = optim.AdamW(model.parameters(), lr=alpha) 
    for iter in range(n_iters):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        if iter % (n_iters // 10) == 0:
            dev_loss = estimate_batch_loss(model)
            print(f'Iter : {iter:7d}, Train Loss : {loss.item():.4f}, Valid Loss : {dev_loss:.4f}')
    et = datetime.now()
    print(f'[+] Training Done in {et - st} !')

    # Save the model
    torch.save(model.state_dict(), f'./saved_models/04_01_bi_gram_Generator_{device}.pth')
    print('[+] Model saved!')

else :
    # Load the model
    model.load_state_dict(torch.load(f'./saved_models/04_01_bi_gram_Generator_{device}.pth'))
    print('[+] Model loaded!')

model.to(device)

X_test, y_test = get_batch('test')
logits, loss = model(X_test, y_test)
print(f'[>] Test Loss : {loss.item():.4f}')

print(f'[>] Generating ...')

print('-'*100, '\nStarting with no context\n', '-'*100)
print(dec(model.generate(torch.zeros((1, 1), dtype=torch.long , device=device), n_pred=500)[0].tolist()))

print('-'*100, '\nStarting with "Telugu "\n', '-'*100)

contxt = torch.tensor(enc('Telugu '), dtype=torch.long).unsqueeze(0).to(device)
print(dec(model.generate(contxt, n_pred=500)[0].tolist()))

print('-'*100, '\nStarting with "Telugu "\n', '-'*100)

contxt = torch.tensor(enc('Mahanth '), dtype=torch.long).unsqueeze(0).to(device)
print(dec(model.generate(contxt, n_pred=500)[0].tolist()))

print('[+] hehe boi !')