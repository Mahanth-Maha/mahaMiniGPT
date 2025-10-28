# trainer.py
import os
import time
import logging
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil
import GPUtil
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
from collections import deque

class TransformerTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataset,
        val_dataset,
        output_dir,
        scheduler=None,
        device='cuda',
        max_checkpoints=5,
        log_interval=20,
        eval_interval=500,
        save_interval=2000,
        milestone_intervals=None,
        grad_clip=1.0,
        mixed_precision=True,
        evaluator=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.max_checkpoints = max_checkpoints
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.milestone_intervals = milestone_intervals or [10000, 50000, 100000]
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.evaluator = evaluator

        # Prepare directories
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.milestones_dir = self.output_dir / "milestones"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.milestones_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        self.logger.info(f"Trainer initialized. Output directory: {self.output_dir}")

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()

        # for plotting
        self.train_losses = deque(maxlen=10000)
        self.val_losses = deque(maxlen=1000)
        self.learning_rates = deque(maxlen=10000)
        self.grad_norms = deque(maxlen=10000)
        self.tokens_per_sec = deque(maxlen=1000)

        # mixed precision
        # if self.mixed_precision and torch.cuda.is_available():
        #     self.scaler = torch.cuda.amp.GradScaler()

        # log CSV
        self._setup_csv_logging()

        # Model to device
        self.model.to(self.device)

        # Resume if checkpoint exists
        self._auto_resume()

    def _setup_logging(self):
        self.logger = logging.getLogger('transformer_trainer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        log_file = self.logs_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s','%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_csv_logging(self):
        self.metrics_csv = self.logs_dir / "metrics.csv"
        self.csv_fieldnames = [
            'step', 'epoch', 'train_loss', 'val_loss', 'learning_rate',
            'grad_norm', 'tokens_per_sec', 'time_elapsed', 'eta',
            'gpu_memory_used', 'gpu_utilization'
        ]
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()

    def _log_metrics_to_csv(self, metrics):
        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(metrics)

    def _get_system_stats(self):
        stats = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats['gpu_memory_used'] = gpu.memoryUsed
                stats['gpu_utilization'] = gpu.load * 100
            else:
                stats['gpu_memory_used'] = 0
                stats['gpu_utilization'] = 0
        except Exception:
            stats['gpu_memory_used'] = 0
            stats['gpu_utilization'] = 0
        stats['cpu_percent'] = psutil.cpu_percent()
        stats['ram_percent'] = psutil.virtual_memory().percent
        return stats

    def _calculate_eta(self, current_step, total_steps):
        if current_step == 0:
            return "Unknown"
        elapsed = time.time() - self.start_time
        steps_per_second = current_step / elapsed
        remaining_steps = total_steps - current_step
        if steps_per_second > 0:
            eta_seconds = remaining_steps / steps_per_second
            eta_hours = eta_seconds // 3600
            eta_minutes = (eta_seconds % 3600) // 60
            return f"{int(eta_hours)}h {int(eta_minutes)}m"
        return "Unknown"

    def _save_checkpoint(self, step, is_best=False, is_milestone=False):
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': list(self.train_losses),
            'val_losses': list(self.val_losses),
            'learning_rates': list(self.learning_rates),
            'grad_norms': list(self.grad_norms),
            'tokens_per_sec': list(self.tokens_per_sec),
        }
        if self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.mixed_precision:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        if not is_milestone:
            checkpoint_path = self.checkpoints_dir / f"checkpoint_step_{step}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            self._cleanup_checkpoints()
        if is_milestone or is_best:
            milestone_path = self.milestones_dir / f"milestone_step_{step}.pt"
            torch.save(checkpoint_data, milestone_path)
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
        self.logger.info(f"Checkpoint saved at step {step}")

    def _cleanup_checkpoints(self):
        checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_step_*.pt"), key=lambda x: int(x.stem.split('_')[-1]))
        if len(checkpoints) > self.max_checkpoints:
            for path in checkpoints[:-self.max_checkpoints]:
                path.unlink()
                self.logger.info(f"Removed old checkpoint: {path.name}")

    def _auto_resume(self):
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            self.logger.info("No checkpoint found. Starting training from scratch.")
            return
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        self.load_checkpoint(latest_checkpoint)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = deque(checkpoint.get('train_losses', []), maxlen=10000)
        self.val_losses = deque(checkpoint.get('val_losses', []), maxlen=1000)
        self.learning_rates = deque(checkpoint.get('learning_rates', []), maxlen=10000)
        self.grad_norms = deque(checkpoint.get('grad_norms', []), maxlen=10000)
        self.tokens_per_sec = deque(checkpoint.get('tokens_per_sec', []), maxlen=1000)
        self.logger.info(f"Resumed training from step {self.global_step}")

    def _train_step(self, batch):
        self.model.train()
        batch = batch.to(self.device, non_blocking=True)
        x = batch[:, :-1]
        y = batch[:, 1:]
        self.optimizer.zero_grad()
        if self.mixed_precision and torch.cuda.is_available():
            with torch.cuda.autocast():
                logits, loss = self.model(x, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, loss = self.model(x, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        del batch, x, y, logits
        torch.cuda.empty_cache()
        return loss.item(), grad_norm.item()

    def _validate(self):
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device, non_blocking=True)
                x = batch[:, :-1]
                y = batch[:, 1:]
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.autocast():
                        logits, loss = self.model(x, y)
                else:
                    logits, loss = self.model(x, y)
                total_loss += loss.item() * batch.size(0)
                del batch, x, y, logits
        torch.cuda.empty_cache()
        return total_loss / len(self.val_dataset)

    def _update_plots(self):
        if len(self.train_losses) < 10:
            return
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        plt.plot(list(self.train_losses), label='Train Loss', alpha=0.7)
        if self.val_losses:
            val_steps = [i * self.eval_interval for i in range(len(self.val_losses))]
            plt.plot(val_steps, list(self.val_losses), label='Val Loss', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(list(self.learning_rates), label='LR')
        plt.xlabel('Steps')
        plt.title('Learning Rate')
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(list(self.grad_norms), label='Grad norm')
        plt.xlabel('Steps')
        plt.title('Grad Norm')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(list(self.tokens_per_sec), label='Tokens/sec')
        plt.xlabel('Steps')
        plt.title('Throughput')
        plt.grid(True)
        plt.tight_layout()
        plot_path = self.plots_dir / f"training_progress_step_{self.global_step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, num_epochs=1, batch_size=32, num_workers=4):
        self.logger.info("Starting training loop")
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
        total_steps = len(train_loader) * num_epochs
        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training", unit="step")
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            for batch_idx, batch in enumerate(train_loader):
                if self.global_step < self.global_step:
                    self.global_step += 1
                    pbar.update(1)
                    continue
                step_start_time = time.time()
                train_loss, grad_norm = self._train_step(batch)
                self.train_losses.append(train_loss)
                self.grad_norms.append(grad_norm)
                if self.scheduler:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                tokens_processed = batch.size(0) * batch.size(1)
                step_time = time.time() - step_start_time
                tokens_per_sec = tokens_processed / step_time
                self.tokens_per_sec.append(tokens_per_sec)
                pbar.set_postfix({
                    'loss': f"{train_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'tok/s': f"{tokens_per_sec:.0f}",
                    'grad': f"{grad_norm:.3f}"
                })
                pbar.update(1)
                # Log
                if self.global_step % self.log_interval == 0:
                    system_stats = self._get_system_stats()
                    eta = self._calculate_eta(self.global_step, total_steps)
                    elapsed = time.time() - self.start_time
                    metrics = {
                        'step': self.global_step,
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': self.val_losses[-1] if self.val_losses else None,
                        'learning_rate': current_lr,
                        'grad_norm': grad_norm,
                        'tokens_per_sec': tokens_per_sec,
                        'time_elapsed': elapsed,
                        'eta': eta,
                        'gpu_memory_used': system_stats['gpu_memory_used'],
                        'gpu_utilization': system_stats['gpu_utilization']
                    }
                    self._log_metrics_to_csv(metrics)
                    self.logger.info(f"{metrics}")
                # Validation/eval
                if self.global_step % self.eval_interval == 0:
                    val_loss = self._validate()
                    self.val_losses.append(val_loss)
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    self.logger.info(f"Validation Loss: {val_loss:.4f} {'(Best!)' if is_best else ''}")
                    if self.evaluator:
                        try:
                            eval_results = self.evaluator.evaluate_model(self.model, step=self.global_step)
                            self.logger.info(f"Evaluation results: {eval_results}")
                        except Exception as e:
                            self.logger.error(f"Evaluation failed: {e}")
                    self._save_checkpoint(self.global_step, is_best=is_best)
                # Save checkpoint regular
                if self.global_step % self.save_interval == 0:
                    self._save_checkpoint(self.global_step)
                # Milestones
                if self.global_step in self.milestone_intervals:
                    self._save_checkpoint(self.global_step, is_milestone=True)
                    self.logger.info(f"Milestone reached at step {self.global_step}!")
                # Plots update occasionally
                if self.global_step % (self.eval_interval * 2) == 0:
                    self._update_plots()
                self.global_step += 1
        self._save_checkpoint(self.global_step)
        self._update_plots()
        pbar.close()
        self.logger.info("Training completed.")
        total_time = time.time() - self.start_time
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
