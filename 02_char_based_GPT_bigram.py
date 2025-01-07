import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[>] Device : {device}')

batch_size = 64
block_size = 8
n_iters = 100000
alpha = 1e-2
eval_iters = n_iters // 100

print(f'[>] Loading Data ...')
with open('./data/mini_gpt.txt', 'r') as f:
    X_train_data = f.read()

with open('./data/valid_mini_gpt.txt', 'r') as f:
    X_valid_data = f.read()

with open('./data/test_mini_gpt.txt', 'r') as f:
    X_test_data = f.read()
print(f'[+] Data Loaded !')

s = ''.join(set(X_train_data))
coding_str = sorted(s)

encoder = {}
decoder = {}
for i, char in enumerate(coding_str):
    encoder[char] = i
    decoder[i] = char

def encode(x): return [encoder[char] for char in x]
encode('Mahanth')

def decode(x): return ''.join([decoder[i] for i in x])
decode(encode('Mahanth'))


print(f'[>] Encoding Data ...')
st  = datetime.now()
X_train = torch.tensor(encode(X_train_data), dtype=torch.long)
X_test = torch.tensor(encode(X_test_data), dtype=torch.long)
X_valid = torch.tensor(encode(X_valid_data), dtype=torch.long)
et = datetime.now()
print(f'[+] Data Encoded in {et - st} !')

# import tiktoken
# tikt = tiktoken.get_encoding('gpt2')
# tikt.encode('Mahanth')
# tikt.decode(tikt.encode('Mahanth'))

def batch_split(X, batch_size=batch_size, block_size=block_size):
    ix = torch.randint(len(X) - block_size, (batch_size,))
    x = [X[i: i + block_size] for i in ix]
    y = [X[i + 1: i + block_size + 1] for i in ix]
    return torch.stack(x).to(device), torch.stack(y).to(device)


class BiGramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
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
            x = torch.cat([x, torch.multinomial(prob_dist, 1)], -1)

            # logits = self.token_embedding_table(x)
            # x = torch.cat([x, torch.argmax(logits, -1)[:,-1].unsqueeze(-1)], -1)
        return x

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in [X_train, X_valid]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch_split(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

m = BiGramLanguageModel(len(coding_str))
m.to(device)
optimiser = torch.optim.Adam(m.parameters(), lr=1e-3)


train_model = False
from_scratch = False
if train_model:
    print(f'[>] Training ...')
    if from_scratch:
        print(f'[!] Training from scratch ...')
    else:
        # Load the model
        m = BiGramLanguageModel(len(coding_str))
        m.load_state_dict(torch.load('./saved_models/bigram_language_model.pth'))
        m.to(device)
        print('[+] Model loaded!')

    st  = datetime.now()
    for iter in range(n_iters):
        x, y = batch_split(X_train)
        logits, loss = m(x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if iter % eval_iters == 0:
            losses = estimate_loss(m)
            print(f'Iter : {iter}, Train Loss : {losses[X_train]:.4f}, Valid Loss : {losses[X_valid]:.4f}')
    et = datetime.now()

    print(f'[+] Training Done in {et - st} !')

    # Save the model
    torch.save(m.state_dict(), './saved_models/bigram_language_model.pth')
    print('[+] Model saved!')

else :
    # Load the model
    m = BiGramLanguageModel(len(coding_str))
    m.load_state_dict(torch.load('./saved_models/bigram_language_model.pth'))
    m.to(device)
    print('[+] Model loaded!')

print(f'[>] Testing ...')
x1 = torch.zeros((1, 1), dtype=torch.long , device=device)
print(decode(m.generate(x1, n_pred=500)[0].tolist()))

print('-'*100)

contxt = torch.tensor(encode('Telugu '), dtype=torch.long).unsqueeze(0).to(device)
print(decode(m.generate(contxt, n_pred=500)[0].tolist()))


