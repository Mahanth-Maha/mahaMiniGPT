model_name = '04_02_better_usage_of_context'
'''
In this script, I used a better context for the model to generate text.
'''

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

train_model = True
train_from_scratch = False

alpha = 1e-4

batch_size = 64
block_size = 16
max_iters = 100000
pred_char_len = 1000
eval_iters = max_iters // 100

n_embeddings = 72 

models_folder = './saved_models'
os.makedirs(models_folder, exist_ok=True)

model_file = f'{models_folder}/{model_name}_{device}.pth'
encoding_file = f'{models_folder}/04_01_bi_gram_Generator_{device}_encoding.pth'
data_file = './data/wikitext/processed.txt'
chars_file = './data/wikitext/chars.txt'

model_exists = os.path.exists(model_file)
encoding_exists = os.path.exists(encoding_file)
data_exists = os.path.exists(data_file)
chars_exists = os.path.exists(chars_file)


if not data_exists:
    print(f'[!] Data not found !')
    exit(0)

data = None
if not chars_exists:
    print(f'[!] chars not found !')
    print(f'[>] Loading Data & Getting chars...')
    data = open(data_file, 'r').read()
    chars = ''.join(sorted(list(set(data))))
    open(chars_file, 'w').write(chars)
    chars_exists = True
    print(f'[+] Data Loaded and chars saved !')
else:
    chars = open(chars_file, 'r').read()


def enc(x, chars=chars):
    idxs = []
    for c in x:
        idxs.append(chars.index(c))
    return idxs


def dec(x, chars=chars):
    txt = ''
    for i in x:
        txt += chars[i]
    return txt


if encoding_exists:
    encoded_data = torch.load(encoding_file).clone().detach()
    encoded_data = encoded_data.to(device)
    print(f'[+] Encoding Loaded !')
else:
    print(f'[>] Loading Data ...')
    if data is None:
        data = open(data_file, 'r').read()
    print(f'[+] Data Loaded !')
    print(f'[>] Encoding Data ...')
    st = datetime.now()
    encoded_data = torch.tensor(enc(data), dtype=torch.long).to(device)
    et = datetime.now()
    print(f'[+] Data Encoded in {et - st} !')
    torch.save(encoded_data, encoding_file)
    print(f'[+] Encoding saved !')

vocab_size = len(chars)
print(f'[>] Number of Unique Characters : {vocab_size}')

n1 = int(0.8 * len(encoded_data))
n2 = int(0.9 * len(encoded_data))
Xtr = encoded_data[:n1]
Xdev = encoded_data[n1:n2]
Xte = encoded_data[n2:]

Xtr.to(device)
Xdev.to(device)
Xte.to(device)


class BiGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size ):
        super(BiGramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embeddings = nn.Embedding(block_size, n_embeddings)
        self.lang_modelling_head = nn.Linear(n_embeddings, vocab_size)

    def forward(self, x, y=None):
        # (batch_size, block_size, vocab_size)
        xB , xT = x.shape
        token_embeddings = self.token_embedding_table(x)
        pos_embeddings = self.position_embeddings(torch.arange(xT).to(device))
        embeds = token_embeddings + pos_embeddings

        logits = self.lang_modelling_head(embeds)

        if y is None:
            return logits

        # entropy expects : (N, C) input : (batch_size * block_size, vocab_size)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss

    def generate(self, x, n_pred):
        for _ in range(n_pred):
            logits = self(x[:,-self.block_size:])[:, -1, :]
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


@torch.no_grad()
def estimate_batch_loss(model):
    X_batch, y_batch = get_batch('dev')
    logits, loss = model(X_batch, y_batch)
    return loss.item()


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'dev', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BiGramLanguageModel(vocab_size , block_size)
model = model.to(device)

if train_model:
    print(f'\n[>] Training ...')
    if not model_exists or train_from_scratch:
        print(f'[!] Training from scratch ...')
    else:
        if model_exists:
            model.load_state_dict(torch.load(model_file))
            print('[+] Model loaded! Resuming training ...')

    st = datetime.now()
    optimiser = optim.AdamW(model.parameters(), lr=alpha)
    for iter in range(max_iters):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        if iter % (max_iters // 10) == 0:
            train_loss = estimate_loss(model)['train']
            dev_loss = estimate_loss(model)['dev']
            print(f'Iter : {iter:7d}, Train Loss : {train_loss:.4f}, Valid Loss : {dev_loss:.4f}')
    et = datetime.now()
    print(f'[+] Training Done in {et - st} !')
    torch.save(model.state_dict(), model_file)
    print('[+] Model saved!')

else:
    if not model_exists:
        print(f'[!] Model not found ! \n\nTRIANING REQUIRED !\n\n')
        exit(0)
    model.load_state_dict(torch.load(model_file))
    print('[+] Model loaded!')

# model.to(device)

print(f'\n[>] Testing ...')
out = estimate_loss(model)
for ty, loss in out.items():
    print(f'\t{ty:5} Loss : {loss:.4f}')
print(f'\n')

print(f'[>] Generating ...')

print('-'*100 + '\n\tStarting with no context\n' + '-'*100)
print(dec(model.generate(torch.zeros((1, 1), dtype=torch.long,device=device), n_pred=pred_char_len)[0].tolist()))
print()

starts_with = " = Marvel = \n "
print('-'*100 + f'\n\tStarting with {repr(starts_with)}\n' + '-'*100)

contxt = torch.tensor(enc(starts_with), dtype=torch.long).unsqueeze(0).to(device)
print(dec(model.generate(contxt, n_pred=pred_char_len)[0].tolist()))
print()

starts_with = " = Computer = \n "
print('-'*100, f'\n\tStarting with {repr(starts_with)}\n')
print('-'*100)

contxt = torch.tensor(enc(starts_with), dtype=torch.long).unsqueeze(0).to(device)
print(dec(model.generate(contxt, n_pred=pred_char_len)[0].tolist()))
print()

print('[+] hehe boi !')

""" 

Sample Output :



"""