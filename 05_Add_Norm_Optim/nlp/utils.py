import torch.nn.functional as F
import numpy as np
import math
import torch
import csv
import os


def get_batch(X, block_size, batch_size, device):
    start = np.random.randint(0, len(X) - block_size - 1, (batch_size,))
    X_batch = torch.stack([X[s:s + block_size] for s in start]).to(device)
    y_batch = torch.stack([X[s+1:s + block_size+1] for s in start]).to(device)
    return X_batch.to(device), y_batch.to(device)


@torch.no_grad()
def get_metrics(model, datasets, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for ty, dataset in datasets.items():
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


def log_results_to_csv(
    csv_path,
    model_name,
    model_parameters,
    train_time,
    args,
    metrics
):
    args_dict = vars(args)
    remove = [
        'pred_char_len',
        'models_folder',
        'data_file',
        'chars_file',
        'encoding_file',
        'model_file',
        'n_gram',
        'csv_file',
        'generate',
    ]
    # Prepare a row dictionary with model and args as prefix columns

    def convert_to_KMB(model_params):
        if model_params//1000 <= 0:
            return f'{model_params} B'
        elif model_params//1000000 <= 0:
            return f'{model_params/1000:.1f} K'
        elif model_params//1000000000 <= 0:
            return f'{model_params/1000000:.1f} M'
        elif model_params//1000000000000 <= 0:
            return f'{model_params/1000000000:.1f} G'
        else:
            return f'{model_params/1000000000000:.1f} T'

    row = {
        "model_name": model_name,
        'params': convert_to_KMB(model_parameters),
        'param_count': model_parameters,
        'train_time': train_time
    }
    row.update(args_dict)

    for k in remove:
        row.pop(k, None)
    # Add metrics for each split (train/val/test) and stat
    for split, val in metrics.items():
        for key, value in val.items():
            # Example: train_loss, val_acc, test_bpc, etc.
            row[f"{split}_{key}"] = value
    # Define the order of columns (optional: read header if file exists)
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
    else:
        fieldnames = list(row.keys())
    # Append row to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.path.getsize(csv_path) == 0:  # File is empty, write header
            writer.writeheader()
        writer.writerow(row)

    # # Add metrics for each split (train/val/test) and stat
    # for split, val in metrics.items():
    #     for key, value in val.items():
    #         # Example: train_loss, val_acc, test_bpc, etc.
    #         row[f"{split}_{key}"] = value

    # # Define the order of columns (optional: read header if file exists)
    # if os.path.isfile(csv_path):
    #     with open(csv_path, newline='') as f:
    #         reader = csv.DictReader(f)
    #         fieldnames = reader.fieldnames or []
    # else:
    #     fieldnames = []

    # # Add any new keys from row to fieldnames
    # new_keys = [k for k in row.keys() if k not in fieldnames]
    # fieldnames.extend(new_keys)

    # # Prepare row aligned with full fieldnames, fill blanks for missing keys
    # aligned_row = {key: row.get(key, "") for key in fieldnames}

    # # Check if file empty to write header
    # write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0

    # # Append row
    # with open(csv_path, 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     if write_header:
    #         writer.writeheader()
    #     writer.writerow(aligned_row)
