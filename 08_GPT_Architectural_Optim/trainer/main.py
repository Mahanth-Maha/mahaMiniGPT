# main.py
import argparse
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from trainer import TransformerTrainer
from train_utils import TransformerDataset, TextEvaluator

from my_transformer import DecoderOnlyTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help="Path to config or set parameters below")
    parser.add_argument('--train-shard-dir', type=str, required=True)
    parser.add_argument('--val-shard-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--context-length', type=int, default=128)
    parser.add_argument('--vocab-size', type=int, default=100277)
    parser.add_argument('--model-dim', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--ffn-hid-dim', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--mixed-precision', action='store_true')
    return parser.parse_args()

def create_optimizer_and_scheduler(model, args, total_steps):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / args.warmup_steps, 1.0)
    )
    return optimizer, scheduler

def main():
    args = parse_args()
    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        model_dimension=args.model_dim,
        n_heads=args.n_heads,
        Nx=args.n_layers,
        ffn_hid_dim=args.ffn_hid_dim,
        non_linearity='gelu',
        dropout=0.1,
        device=args.device
    )
    train_dataset = TransformerDataset(args.train_shard_dir, args.context_length)
    val_dataset = TransformerDataset(args.val_shard_dir, args.context_length)
    total_steps = len(train_dataset) // args.batch_size * args.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, total_steps)
    evaluator = TextEvaluator(tokenizer_name='cl100k_base', output_csv_path=os.path.join(args.output_dir, 'eval_metrics.csv'), device=args.device)

    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        scheduler=scheduler,
        device=args.device,
        log_interval=100,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        grad_clip=args.gradient_clip,
        mixed_precision=args.mixed_precision,
        evaluator=evaluator
    )
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
