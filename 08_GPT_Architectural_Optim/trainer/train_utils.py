# train_utils.py
import os
import mmap
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import tiktoken
from tqdm import tqdm
import multiprocessing

# === SHARD CREATOR ===
def create_shards_from_hf_txt(hf_txt_path, shard_dir, tokenizer_name, shard_size=10**6, min_length=32):
    import tiktoken
    os.makedirs(shard_dir, exist_ok=True)
    encoding = tiktoken.get_encoding(tokenizer_name)
    current_shard = []
    shard_idx = 0
    with open(hf_txt_path, "r", encoding="utf-8") as src:
        pbar = tqdm(src, desc="Tokenizing and Sharding")
        for line in pbar:
            tokens = encoding.encode(line.strip())
            if len(tokens) < min_length:
                continue
            current_shard.extend(tokens)
            while len(current_shard) >= shard_size:
                out = np.array(current_shard[:shard_size], dtype=np.uint16)
                np.save(os.path.join(shard_dir, f"shard_{shard_idx:04d}.npy"), out)
                current_shard = current_shard[shard_size:]
                shard_idx += 1
    if current_shard:
        out = np.array(current_shard, dtype=np.uint16)
        np.save(os.path.join(shard_dir, f"shard_{shard_idx:04d}.npy"), out)

# # === STREAMING DATASET LOADER ===
# class TransformerDataset(IterableDataset):
#     def __init__(self, shard_directory, context_length, shuffle=True, buffer_size=4):
#         self.shard_files = sorted(glob.glob(os.path.join(shard_directory, "*.npy")))
#         self.context_length = context_length
#         self.shuffle = shuffle
#         self.buffer_size = buffer_size
#         self.shard_buffers = [None] * len(self.shard_files)

#     def __len__(self):
#         # Returns total number of possible contexts *approximated*
#         all_count = 0
#         for f in self.shard_files:
#             arr = np.load(f, mmap_mode="r")
#             all_count += int((arr.shape[0] - self.context_length) // self.context_length)
#         return all_count

#     def __iter__(self):
#         order = np.arange(len(self.shard_files))
#         rng = np.random.default_rng()
#         if self.shuffle:
#             rng.shuffle(order)
#         for idx in order:
#             arr = np.load(self.shard_files[idx], mmap_mode="r")
#             arr_len = arr.shape[0]
#             ix = np.arange(0, arr_len - self.context_length, self.context_length)
#             if self.shuffle:
#                 rng.shuffle(ix)
#             for start in ix:
#                 yield torch.from_numpy(arr[start: start + self.context_length]).long()



# === STREAMING DATASET LOADER ===
class TransformerDataset(IterableDataset):
    def __init__(self, shard_directory: str, context_length: int, shuffle: bool = True, buffer_size: int = 4):
        """
        Args:
          shard_directory: directory containing shard_xxxxx.pt files
          context_length: length of each output token window
          shuffle: whether to shuffle shards and windows
          buffer_size: number of shards to buffer in memory (not implemented fully here, placeholder)
        """
        self.shard_files = sorted(glob.glob(os.path.join(shard_directory, "*.pt")))
        self.context_length = context_length
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        # Optionally, you can implement an LRU cache of loaded shards, but here it's simple
        # self.shard_buffers = {}

    def __len__(self):
        # Warning: for large datasets this is expensive (loads every shard)
        total_windows = 0
        for f in self.shard_files:
            tok = torch.load(f)
            n = tok.numel()
            if n >= self.context_length:
                # number of non-overlapping windows
                total_windows += (n - self.context_length + 1) // self.context_length
        return total_windows

    def __iter__(self):
        # Optionally shuffle order of shards
        order = list(range(len(self.shard_files)))
        if self.shuffle:
            np.random.shuffle(order)

        for idx in order:
            path = self.shard_files[idx]
            tok = torch.load(path)  # 1D LongTensor
            n = tok.numel()
            if n < self.context_length:
                continue

            # Compute starting indices (non-overlapping windows)
            # Alternatively, you could use overlapping windows by setting stride < context_length
            starts = np.arange(0, n - self.context_length + 1, self.context_length)
            if self.shuffle:
                np.random.shuffle(starts)

            for st in starts:
                yield tok[st : st + self.context_length]



# === EVALUATOR AND METRICS ===
import math
from collections import defaultdict

class TextEvaluator:
    def __init__(self, tokenizer_name="cl100k_base", output_csv_path=None, eval_sample_size=1000, device='cuda'):
        self.encoding = tiktoken.get_encoding(tokenizer_name)
        self.output_csv_path = output_csv_path
        self.eval_sample_size = eval_sample_size
        self.device = device
        if output_csv_path:
            import csv
            with open(output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "input_text", "generated", "perplexity", "neg_loss", "BPC", "bleu", "topk_acc"])
    
    @staticmethod
    def calc_perplexity(loss):
        return math.exp(loss)
    
    @staticmethod
    def calc_bpc(loss):
        return loss / math.log(2)
    
    def calc_bleu(self, ref, hyp):
        import sacrebleu
        return sacrebleu.corpus_bleu([hyp], [[ref]]).score
    
    def topk_acc(self, logits, targets, k=5):
        with torch.no_grad():
            vals, idx = logits.topk(k, dim=-1)
            targets_exp = targets.unsqueeze(-1)
            return (idx == targets_exp).any(dim=-1).float().mean().item()
    
    @torch.no_grad()
    def evaluate_model(self, model, step=None, val_data=None):
        """ Produces metrics and generated samples.
        """
        if val_data is None:
            raise ValueError("Must provide validation data")
        model.eval()
        loss_list = []
        bleu_list = []
        bpc_list = []
        tok_list = []
        kacc_list = []
        for idx, batch in enumerate(val_data):
            if idx > self.eval_sample_size:
                break
            batch = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits, loss = model(x, y)
            loss_list.append(loss.item())
            perplexity = self.calc_perplexity(loss.item())
            bpc = self.calc_bpc(loss.item())
            kacc = self.topk_acc(logits.cpu(), y.cpu())
            bpc_list.append(bpc)
            kacc_list.append(kacc)
            if idx == 0:  # Generate sample only for first example
                y_pred = torch.argmax(logits, dim=-1)
                gen_tokens = y_pred[0].tolist()
                input_tokens = x[0].tolist()
                bleu = self.calc_bleu(self.encoding.decode(input_tokens), self.encoding.decode(gen_tokens))
                bleu_list.append(bleu)
                generated = self.encoding.decode(gen_tokens)
                ref = self.encoding.decode(input_tokens)
            else:
                bleu_list.append(0)
        ret = {
            "loss": np.mean(loss_list),
            "perplexity": np.mean([self.calc_perplexity(l) for l in loss_list]),
            "BPC": np.mean(bpc_list),
            "topk_acc": np.mean(kacc_list),
            "bleu": np.mean(bleu_list)
        }
        if self.output_csv_path:
            import csv
            with open(self.output_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, ref, generated, ret["perplexity"], ret["loss"], ret["BPC"], ret["bleu"], ret["topk_acc"]])
        return ret
