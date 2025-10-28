model_name = '04_01_NxBlocks'
'''
In this script, I used a Self Attention mechanism to the model to generate text.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from tqdm import tqdm

from nlp.tokenizer import enc, dec 
from nlp.StackedBlocks import NGramSBlocks
from nlp.utils import get_metrics, log_results_to_csv

def get_args():
    parser = argparse.ArgumentParser(description="Train a character-level LM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (ignore saved models)")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--block_size", type=int, default=16, help="Block size (context length)")
    parser.add_argument("--max_iters", type=int, default=10000, help="Maximum training iterations")
    parser.add_argument("--pred_char_len", type=int, default=1000, help="Prediction length for generation")
    parser.add_argument("--n_embeddings", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--models_folder", default="../saved_models", help="Folder for saving models")
    parser.add_argument("--data_file", default="../data/wikitext/processed.txt", help="Processed text file")
    parser.add_argument("--chars_file", default="../data/wikitext/chars.txt", help="Characters file")
    parser.add_argument("--encoding_file", default='../data/encodings/encoding.pth', help="Encoded data file")
    parser.add_argument("--model_file", type=str, default=None, help="Model checkpoint file")

    parser.add_argument("--n_gram", default=2, help="N gram model")
    parser.add_argument("--csv_file", type=str, default=f'{model_name}.csv', help="N gram model")
    parser.add_argument("--generate", action="store_true", help="Toggle on to generate from model")

    return parser.parse_args()


args = get_args()
DEVICE = args.device
BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size

ALPHA = args.alpha
MAX_ITERS = args.max_iters
EVAL_ITERS = MAX_ITERS // 100
MAX_CONTEXT_LEN = args.pred_char_len
N_EMBEDDINGS = args.n_embeddings



print(f"[>] Device : {DEVICE}")

os.makedirs(args.models_folder, exist_ok=True)
if args.model_file is not None:
    MODEL_FILE = args.model_file
else :
    MODEL_FILE = f'{args.models_folder}/{model_name}_{DEVICE}.pth'

model_exists = os.path.exists(MODEL_FILE)

data_exists = os.path.exists(args.data_file)
chars_exists = os.path.exists(args.chars_file)

if not os.path.exists(args.data_file):
    print("[!] Data not found!")
    exit(0)

if not os.path.exists(args.chars_file):
    print("[>] Loading Data & Getting chars...")
    data = open(args.data_file, "r").read()
    chars = "".join(sorted(set(data)))
    open(args.chars_file, "w").write(chars)
    print("[+] Data Loaded and chars saved!")
else:
    chars = open(args.chars_file, "r").read()

if os.path.exists(args.encoding_file):
    encoded_data = torch.load(args.encoding_file, map_location=DEVICE)
    print("[+] Encoding Loaded!")
else:
    print("[>] Encoding not found! Loading Data... to encode")
    data = open(args.data_file, "r").read()
    print("[+] Data Loaded!")
    print("[>] Encoding Data...")
    st = datetime.now()
    encoded_data = torch.tensor(enc(data, chars), dtype=torch.long, device=DEVICE)
    print(f"[+] Data Encoded in {datetime.now() - st}!")
    torch.save(encoded_data.cpu(), args.encoding_file)  # save on CPU for portability
    print("[+] Encoding saved and Loaded!")

vocab_size = len(chars)
print(f"[>] Number of Unique Characters (vocab size) : {vocab_size}")

n = len(encoded_data)
n1, n2 = int(0.8 * n), int(0.9 * n)
Xtr, Xval, Xte = encoded_data[:n1], encoded_data[n1:n2], encoded_data[n2:]

print(f"[>] Dataset split into train:{len(Xtr)}, dev:{len(Xval)}, test:{len(Xte)}")

Xtr.to(DEVICE)
Xval.to(DEVICE)
Xte.to(DEVICE)

def get_batch(split):
    if split == 'train':
        X = Xtr
    elif split == 'val':
        X = Xval
    else:
        X = Xte
    start = np.random.randint(0, len(X) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    X_batch = torch.stack([X[s:s + BLOCK_SIZE] for s in start]).to(DEVICE)
    y_batch = torch.stack([X[s+1:s + BLOCK_SIZE+1] for s in start]).to(DEVICE)
    return X_batch.to(DEVICE), y_batch.to(DEVICE)


@torch.no_grad()
def estimate_batch_loss(model):
    X_batch, y_batch = get_batch('dev')
    logits, loss = model(X_batch, y_batch)
    return loss.item()


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val', 'test']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# bi-gram model
model = NGramSBlocks(
    vocab_size=vocab_size, 
    block_size=BLOCK_SIZE,
    Nx=2,
    ffn_hid_dim = 256,
    n_heads = 8,
    n_embeddings=N_EMBEDDINGS,
    device=DEVICE,
    nonlinearity='gelu',
    n_gram=args.n_gram,
    )
model = model.to(DEVICE)
print('[>] Model Created\n\n')
print('-'*50)
model_parameters = sum(p.numel() for p in model.parameters())
print(f"[>] Total parameters:{model_parameters:,}")
print('-'*50)
print('\n')


train_time = None
if args.train:
    print(f'\n[>] Training ...')
    if not model_exists or args.scratch:
        print(f'[!] Training from scratch ...')
    else:
        if model_exists:
            model.load_state_dict(torch.load(MODEL_FILE))
            print('[+] Model loaded! Resuming training ...')

    st = datetime.now()
    optimiser = optim.AdamW(model.parameters(), lr=ALPHA)
    for iter in tqdm(range(MAX_ITERS), desc='Training', unit='iteration' ):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        if iter % (MAX_ITERS // 10) == 0:
            train_loss = estimate_loss(model)['train']
            dev_loss = estimate_loss(model)['val']
            print(f'Iter : {iter:7d}, Train Loss : {train_loss:.4f}, Valid Loss : {dev_loss:.4f}')
    et = datetime.now()
    print(f'[+] Training Done in {et - st} !')
    train_time = et - st
    torch.save(model.state_dict(), MODEL_FILE)
    print('[+] Model saved!')

else:
    if not model_exists:
        print(f'[!] Model not found ! \n\nTRIANING REQUIRED !\n\n')
        exit(0)
    model.load_state_dict(torch.load(MODEL_FILE))
    print('[+] Model loaded!')

# model.to(DEVICE)

print(f'\n[>] Testing ...')
out = get_metrics(model,
                  {
                      'train' : Xtr,
                      'val' : Xval,
                      'test' : Xte,
                  },
                  BLOCK_SIZE,
                  BATCH_SIZE,
                  EVAL_ITERS,
                  DEVICE
                  )
for split, metrics in out.items():
    print(f"\t{split:5} | "
          f"Loss: {metrics['loss']:.4f} | "
          f"PPL: {metrics['perplexity']:.2f} | "
          f"Acc: {metrics['accuracy']*100:.2f}% | "
          f"BPC: {metrics['bpc']:.4f}")
print()


if args.generate:
    print(f'[>] Generating ...')

    print('-'*100 + '\n\tExample 1 : Starting with no context\n' + '-'*100)
    print(dec(
        model.generate(torch.zeros((1, 1), dtype=torch.long,device=DEVICE), n_pred=MAX_CONTEXT_LEN)[0].tolist() ,
        chars
        ))
    print()

    starts_with = " = Marvel = \n "
    print('-'*100 + f'\n\tExample 2 : Starting with {repr(starts_with)}\n' + '-'*100)

    contxt = torch.tensor(enc(starts_with,chars), dtype=torch.long).unsqueeze(0).to(DEVICE)
    print(dec(
        model.generate(contxt, n_pred=MAX_CONTEXT_LEN)[0].tolist(),
        chars
        ))
    print()

    starts_with = " = Computer = \n "
    print('-'*100, f'\n\tExample 3 : Starting with {repr(starts_with)}\n' + '-'*100)

    contxt = torch.tensor(enc(starts_with,chars), dtype=torch.long).unsqueeze(0).to(DEVICE)
    print(dec(
        model.generate(contxt, n_pred=MAX_CONTEXT_LEN)[0].tolist(),
        chars
        ))
    print()
print(f'[>] Writing results to csv {args.csv_file} !')
log_results_to_csv(args.csv_file, model_name, model_parameters, train_time, args, out)
print('[+] Done !')

""" 
python 02_02_3_Multi_Self_Attention.py --train
"""