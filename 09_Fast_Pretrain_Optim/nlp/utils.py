import torch.nn.functional as F
import math
import numpy as np
import torch
import os
import csv

def get_batch( X, context_length, batch_size,device):
    start = np.random.randint(0, len(X) - context_length - 1, (batch_size,))
    X_batch = torch.stack([X[s:s + context_length] for s in start]).to(device)
    y_batch = torch.stack([X[s+1:s + context_length+1] for s in start]).to(device)
    return X_batch.to(device), y_batch.to(device)

@torch.no_grad()
def get_metrics(model,datasets, context_length, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for ty, dataset in datasets.items() :
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for k in range(eval_iters):
            X, Y = get_batch(dataset, context_length, batch_size, device)
            logits, loss = model(X, Y)

            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == Y).sum().item()
        
        avg_loss = (total_loss / total_tokens) if total_tokens != 0 else  total_loss
        ppl = math.exp(avg_loss)
        acc = total_correct / total_tokens if total_tokens != 0 else  total_correct
        bpc = avg_loss / math.log(2)

        out[ty] = {
            "loss": avg_loss,
            "perplexity": ppl,
            "accuracy": acc,
            "bpc": bpc
        }

    model.train()
    return out

def convert_to_hr(model_params):
    for size in ('B', 'K', 'M', 'B', 'T', 'P'):
        if model_params//1000 <= 0:
            return f'{model_trail/1000:.2f} {size}'
        model_trail = model_params
        model_params = model_params//1000
    return f'{model_trail/1000:.2f} Z'


def log_results_to_csv(
    csv_path,
    model_name,
    model_parameters,
    train_time,
    args,
    metrics
):
    args_dict = vars(args)

    
    row = {
        "model_name": model_name,
        'params': convert_to_hr(model_parameters),
        'param_count': model_parameters,
        'train_time': train_time
    }
    row.update(args_dict)

    remove = [
        'models_folder',
        'data_file',
        'token_model_file',
        'chars_file',
        'encoding_file',
        'model_file',
        'csv_file',
        'generate',
        'pred_char_len',
        'log_level',
    ]
    for k in remove:
        row.pop(k, None)
    
    for split, val in metrics.items():
        for key, value in val.items():
            row[f"{split}_{key}"] = value
    try:
        if os.path.isfile(csv_path):
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
        else:
            fieldnames = list(row.keys())
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error logging results to Given CSV file: {e}")
        print(f"Attempting to write to temporary csv file: {csv_path}")
        temp_csv = f'{args.model_name}_{args.version}_temp_results.csv'
        print(f"writing it to temporary csv file: {temp_csv}")
        with open(temp_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
            if os.path.getsize(temp_csv) == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"Temporary results written to {temp_csv}. Please check and merge manually.")
    finally:
        print("CSV Logging finished.")

        