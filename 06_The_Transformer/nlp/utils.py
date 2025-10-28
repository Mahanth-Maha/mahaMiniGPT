import torch.nn.functional as F
import math
import numpy as np
import torch

def get_batch( X, block_size, batch_size,device):
    start = np.random.randint(0, len(X) - block_size - 1, (batch_size,))
    X_batch = torch.stack([X[s:s + block_size] for s in start]).to(device)
    y_batch = torch.stack([X[s+1:s + block_size+1] for s in start]).to(device)
    return X_batch.to(device), y_batch.to(device)

@torch.no_grad()
def get_metrics(model,datasets, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for ty, dataset in datasets.items() :
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for k in range(eval_iters):
            X, Y = get_batch(dataset, block_size, batch_size, device)
            logits, loss = model(X, Y)

            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()

            # accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == Y).sum().item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        acc = total_correct / total_tokens
        bpc = avg_loss / math.log(2)

        out[ty] = {
            "loss": avg_loss,
            "perplexity": ppl,
            "accuracy": acc,
            "bpc": bpc
        }

    model.train()
    return out
