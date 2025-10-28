# check_model.py
import argparse
import torch
from my_transformer import DecoderOnlyTransformer
from train_utils import TransformerDataset
import os
import time

def estimate_train_time(model, loader, device, n_trials=5):
    times = []
    for i, batch in enumerate(loader):
        if i >= n_trials: break
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        start = time.time()
        logits, loss = model(x, y)
        t = time.time() - start
        times.append(t)
        del batch, logits, loss, x, y
        torch.cuda.empty_cache()
    avg = sum(times) / len(times)
    
    return avg


from torch.utils.data import DataLoader


# def check():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train-shard-dir', type=str, required=True)
#     parser.add_argument('--context-length', type=int, default=2048)
#     parser.add_argument('--vocab-size', type=int, default=100000)
#     parser.add_argument('--model-dim', type=int, default=2048)
#     parser.add_argument('--n-heads', type=int, default=16)
#     parser.add_argument('--n-layers', type=int, default=24)
#     parser.add_argument('--device', type=str, default='cuda')
#     args = parser.parse_args()

#     model = DecoderOnlyTransformer(
#         vocab_size=args.vocab_size,
#         context_length=args.context_length,
#         model_dimension=args.model_dim,
#         n_heads=args.n_heads,
#         Nx=args.n_layers,
#         device=args.device
#     )
#     loader = TransformerDataset(args.train_shard_dir, args.context_length)
#     print_model_stats(args, model, loader)

def check():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-shard-dir', type=str, required=True)
    parser.add_argument('--context-length', type=int, default=2048)
    parser.add_argument('--vocab-size', type=int, default=100000)
    parser.add_argument('--model-dim', type=int, default=2048)
    parser.add_argument('--n-heads', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        model_dimension=args.model_dim,
        n_heads=args.n_heads,
        Nx=args.n_layers,
        device=args.device
    )
    dataset = TransformerDataset(args.train_shard_dir, args.context_length)
    data_loader = DataLoader(dataset, batch_size=4, num_workers=0)
    print_model_stats(args, model, data_loader)

def print_model_stats(args, model, data_loader):
    batch = next(iter(data_loader))
    vram = torch.cuda.max_memory_allocated(args.device) // (1024 ** 2)
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_cnt:,}")
    print(f"Context size: {args.context_length}")
    print(f"Shards: {len(os.listdir(args.train_shard_dir))}")
    print(f"Batch shape: {batch.shape}")
    print(f"Max batch VRAM use: {vram} MB")
    print(f"Estimated batch tokens: {batch.numel()}")
    print(f"Settings: {args}")

if __name__ == '__main__':
    check()
