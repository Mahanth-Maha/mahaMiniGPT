import os
import json
import math
import time
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import shutil

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from train_utils import DataLoaderLite, ModelEvaluator

# Add these imports at the top
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set matplotlib to use non-interactive backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend
import matplotlib.pyplot as plt

# Configure matplotlib to reduce font searching
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # Use a specific font to avoid searching
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})


# # Setup logging
# def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
#     """Create a logger with file and console handlers"""
#     formatter = logging.Formatter(
#         '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     logger = logging.getLogger(name)
    
#     # FIX: Prevent duplicate handlers
#     if logger.hasHandlers():
#         return logger  # Return existing logger if already configured
    
#     logger.setLevel(level)
    
#     # File handler
#     file_handler = logging.FileHandler(log_file, mode='a')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
    
#     # Console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)
#     logger.addHandler(console_handler)
    
#     # FIX: Prevent propagation to root logger
#     logger.propagate = False
    
#     return logger


class TransformerTrainer:
    """
    Professional Transformer Training System with Production-Ready Features
    
    Features:
    - Memory Efficient: Optimized GPU memory usage with gradient accumulation
    - Scalable: Supports streaming data and large datasets  
    - Robust: Comprehensive checkpointing and error handling
    - Observable: Detailed logging, metrics tracking, and visualization
    - Research Friendly: Full metric suite and generation sampling
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        tokenizer,
        logger,
        train_dataloader,
        val_dataloader,
        output_dir: str = "./training_output",
        max_checkpoints: int = 5,
        checkpoint_every: int = 10000,
        eval_every: int = 5000,
        log_every: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        generation_config: Optional[Dict] = None,
        warmup_steps: int = 0,
        save_total_limit: int = None,
        max_steps: int = None,
        vocab_size: int = None,
        batch_size: int = None,
        val_steps: int = 50,
        # train_steps: int = 1000,
    ):
        """
        Initialize the trainer with comprehensive configuration
        
        Args:
            model: Transformer model to train
            optimizer: PyTorch optimizer 
            scheduler: Learning rate scheduler (optional)
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            output_dir: Directory to save outputs
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_every: Save checkpoint every N steps
            eval_every: Run evaluation every N steps
            log_every: Log metrics every N steps
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Gradient clipping threshold
            device: Training device
            mixed_precision: Use automatic mixed precision
            gradient_checkpointing: Use gradient checkpointing to save memory
            compile_model: Use torch.compile for optimization
            generation_config: Configuration for text generation sampling
            warmup_steps: Learning rate warmup steps
            save_total_limit: Total checkpoints to keep (overrides max_checkpoints if set)
        """
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Training configuration
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.warmup_steps = warmup_steps
        
        # Checkpoint and evaluation configuration
        self.max_checkpoints = save_total_limit if save_total_limit else max_checkpoints
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        
        self.val_steps = val_steps
        # self.train_steps = train_steps
        
        # Directory setup
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.milestones_dir = self.output_dir / "milestones"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.checkpoints_dir, self.milestones_dir, 
                        self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps if max_steps else float('inf')
        # Setup logging
        # self.logger = setup_logger(
        #     "trainer", 
        #     self.logs_dir / "training.log",
        #     logging.INFO
        # )
        self.logger = logger
        
        # Setup detailed metrics logger (CSV)
        self.metrics_file = self.logs_dir / "training_metrics.csv"
        self.setup_metrics_logger()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.start_time = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': [],
            'grad_norm': [],
            'tokens_per_second': [],
            'steps': []
        }
        
        # Model optimization
        if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.logger.info("Enabled gradient checkpointing")
        
        if compile_model and torch.__version__ >= "2.0":
            try:
                self.model = torch.compile(model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Mixed precision setup
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        
        # Generation configuration
        self.generation_config = generation_config or {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True
        }
        self.model_evaluator = ModelEvaluator(
            logger = self.logger,
            model=self.model, 
            tokenizer = self.tokenizer, 
            device = self.device, 
            generation_config = self.generation_config,
            mixed_precision=False,
        )
        # Move model to device
        self.model = self.model.to(self.device)
        
        self.logger.info("✅ Trainer initialized successfully")
        self.logger.info(f"\t👉 Model parameters: {self.count_parameters():,}")
        self.logger.info(f"\t👉 Training device: {self.device}")
        self.logger.info(f"\t👉 Mixed precision: {mixed_precision}")
        self.logger.info(f"\t👉 Gradient checkpointing: {gradient_checkpointing}")
    
    def setup_metrics_logger(self):
        """Setup CSV logger for detailed training metrics"""
        headers = [
            'step', 'epoch', 'train_loss', 'val_loss', 'perplexity', 
            'learning_rate', 'grad_norm', 'tokens_per_sec', 
            'time_elapsed', 'eta', 'gpu_memory_gb'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to CSV file"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('step', ''),
                metrics.get('epoch', ''),
                metrics.get('train_loss', ''),
                metrics.get('val_loss', ''),
                metrics.get('perplexity', ''),
                metrics.get('learning_rate', ''),
                metrics.get('grad_norm', ''),
                metrics.get('tokens_per_sec', ''),
                metrics.get('time_elapsed', ''),
                metrics.get('eta', ''),
                metrics.get('gpu_memory_gb', '')
            ])
    
    def count_parameters(self) -> int:
        """Count trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**3
        return 0.0
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from cross-entropy loss"""
        return math.exp(min(loss, 20))  # Cap to prevent overflow
    
    def compute_tokens_per_second(self, num_tokens: int, elapsed_time: float) -> float:
        """Compute processing speed in tokens per second"""
        if elapsed_time > 0:
            return num_tokens / elapsed_time
        return 0.0

    def train_step(self, batch) -> Tuple[float, Dict]:
        """Execute a single training step"""
        self.model.train()
        
        # Move batch to device efficiently
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
        
        step_start_time = time.time()

        with torch.amp.autocast(device_type="cuda", enabled=self.mixed_precision):
            # output = model(input)
            # loss = loss_fn(output, target)
            logits, loss = self.model(input_ids, target_ids)
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Calculate metrics
        step_time = time.time() - step_start_time
        num_tokens = input_ids.numel()
        tokens_per_sec = num_tokens / step_time if step_time > 0 else 0.0
        
        # # Clear tensors to free GPU memory
        del input_ids, target_ids, logits
        # torch.cuda.empty_cache()
        
        return loss.item() * self.gradient_accumulation_steps, {
            'tokens_per_sec': tokens_per_sec,
            'step_time': step_time
        }
        

    def optimizer_step(self) -> float:
        """Execute optimizer step with gradient clipping"""
        grad_norm = 0.0
        
        if self.mixed_precision:
            # Gradient clipping
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                ).item()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Gradient clipping  
            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                ).item()
            
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        return grad_norm
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        self.logger.info("📝 Starting evaluation...")
        # self.val_steps
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            target_ids = batch.get('target_ids', input_ids).to(self.device, non_blocking=True)
            
            with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                logits, loss = self.model(input_ids, target_ids)
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
            num_batches += 1
            
            # Clear GPU memory
            # del input_ids, target_ids, logits, loss
            # torch.cuda.empty_cache()
            
            if num_batches >= self.val_steps:
                break
            
        val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # test_batch = next(iter(self.train_dataloader))
        # B = test_batch['input_ids'].shape[0]
        # T = test_batch['input_ids'].shape[1]
        # data_dir = self.train_dataloader.dataset.shard_dir.parent.parent
        # val_steps = len(self.val_dataloader) // self.batch_size 
        # bacth_iter = iter(self.val_dataloader)
        # self.logger.info(f'{val_steps = }')
        # with torch.no_grad():
        #     val_loss_accum = 0.0
        #     for _ in range(val_steps):
        #         batch_ = next(bacth_iter)
        #         x = batch_['input_ids']
        #         y = batch_['target_ids']
        #         x, y = x.to(self.device), y.to(self.device)
                
        #         with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
        #             logits, loss = self.model(x, y)
                
        #         loss = loss / val_steps
        #         val_loss_accum += loss.detach()
                
        #         # Clean up immediately
        #         # del x, y, logits, loss
        
        # val_loss = val_loss_accum.item()
        
        perplexity = self.calculate_perplexity(val_loss)
        
        metrics = {
            'eval_loss': val_loss,
            'eval_perplexity': perplexity,
            'eval_tokens': total_tokens
        }
        
        self.logger.info(f"Evaluation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "The future of AI is", max_length: int = 100) -> str:
        """Generate text sample for monitoring training progress"""
        self.model.eval()
        
        out = self.model_evaluator.generate_and_evaluate_text(prompt)
        
        try:
            completed = out['generated_texts']
            # Basic text generation logic
            generated_text = f"{prompt} {completed}"
            return generated_text
        except Exception as e:
            self.logger.warning(f"Text generation failed: {e}")
            return f"{prompt} [Generation failed]"
    
    def save_checkpoint(self, step: int, is_best: bool = False, is_milestone: bool = False) -> str:
        """Save model checkpoint with comprehensive state"""
        
        checkpoint_name = f"checkpoint-{step}"
        if is_milestone:
            checkpoint_dir = self.milestones_dir / checkpoint_name
        else:
            checkpoint_dir = self.checkpoints_dir / checkpoint_name
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint state
        checkpoint_state = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': {
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_grad_norm': self.max_grad_norm,
                'mixed_precision': self.mixed_precision,
            }
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        torch.save(checkpoint_state, checkpoint_path)
        
        # Save training history as JSON
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save model config if available
        if hasattr(self.model, 'config'):
            config_path = checkpoint_dir / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(vars(self.model.config) if hasattr(self.model.config, '__dict__') else {}, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints (only for regular checkpoints, not milestones)
        if not is_milestone:
            self.cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the limit"""
        checkpoints = []
        
        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (ValueError, IndexError):
                    continue
        
        checkpoints.sort(key=lambda x: x[0])
        
        while len(checkpoints) > self.max_checkpoints:
            _, old_checkpoint = checkpoints.pop(0)
            shutil.rmtree(old_checkpoint)
            self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load checkpoint and resume training state"""
        try:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.is_dir():
                checkpoint_path = checkpoint_path / "checkpoint.pt"
            
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if checkpoint.get('scaler_state_dict') and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore training state
            self.global_step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', {
                'train_loss': [], 'val_loss': [], 'learning_rate': [],
                'perplexity': [], 'grad_norm': [], 'tokens_per_second': [], 'steps': []
            })
            
            self.logger.info(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def auto_resume(self) -> bool:
        """Automatically find and load the latest checkpoint"""
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            return self.load_checkpoint(latest_checkpoint)
        return False
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in checkpoints directory"""
        checkpoints = []
        
        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (ValueError, IndexError):
                    continue
        
        if checkpoints:
            latest_step, latest_dir = max(checkpoints, key=lambda x: x[0])
            return str(latest_dir)
        
        return None
    
    def update_plots(self):
        """Update training plots with optimized performance"""
        if not self.training_history['steps']:
            return
        
        try:
            # Use fast plotting settings
            with plt.style.context('fast'):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
                
                steps = self.training_history['steps']
                
                # Loss plot
                if self.training_history['train_loss']:
                    ax1.plot(steps, self.training_history['train_loss'], label='Train Loss', alpha=0.8, linewidth=1, color='blue')
                if self.training_history['val_loss']:
                    # Align validation loss with corresponding steps
                    val_steps = [s for s in steps if s % self.eval_every == 0][:len(self.training_history['val_loss'])]
                    ax1.plot(val_steps, self.training_history['val_loss'], label='Val Loss', alpha=0.8, linewidth=1.5, color='red')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Perplexity plot
                if self.training_history['perplexity']:
                    val_steps = [s for s in steps if s % self.eval_every == 0][:len(self.training_history['perplexity'])]
                    ax2.plot(val_steps, self.training_history['perplexity'],  color='orange', linewidth=1.5)
                    ax2.set_xlabel('Steps')
                    ax2.set_ylabel('Perplexity')
                    ax2.set_title('Perplexity')
                    ax2.grid(True, alpha=0.3)
                
                # Learning rate plot
                if self.training_history['learning_rate']:
                    ax3.plot(steps, self.training_history['learning_rate'],  color='green', linewidth=1)
                    ax3.set_xlabel('Steps')
                    ax3.set_ylabel('Learning Rate')
                    ax3.set_title('Learning Rate')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_yscale('log')  # Log scale for LR
                
                # Gradient norm plot
                if self.training_history['grad_norm']:
                    ax4.plot(steps, self.training_history['grad_norm'],  color='red', alpha=0.7, linewidth=1)
                    ax4.set_xlabel('Steps')
                    ax4.set_ylabel('Gradient Norm')
                    ax4.set_title('Gradient Norm')
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout(pad=2.0)
                
                # Save with optimized settings
                plot_path = self.plots_dir / f"training_plots_step_{self.global_step}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight', 
                        facecolor='white', edgecolor='none', 
                        format='png', optimize=True)
                plt.close(fig)  # Important: close the figure to free memory
                        
            self.logger.debug(f"Training plots updated: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update plots: {e}")
            # Don't let plotting errors crash training
            pass

        
    def validate_setup(self) -> bool:
        """
        Validate training setup before starting full training
        Tests model forward pass, data loading, optimizer step, etc.
        """
        self.logger.info("Running training setup validation...")
        
        try:
            # Test model forward pass
            self.logger.info("❓ Testing model forward pass...")
            test_batch = next(iter(self.train_dataloader))
            input_ids = test_batch['input_ids'][:1].to(self.device)  # Single sample
            
            self.model.train()
            with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                logits, loss = self.model(input_ids, input_ids)
            expected_loss = f'[Expected loss: -log(likelihood) = {-np.log(1/self.vocab_size).item():.4f}' if self.vocab_size is not None else ""
            self.logger.info(f"✅ Model forward pass successful - Loss: {loss.item():.4f} {expected_loss}")
            
            # Test backward pass
            self.logger.info("❓ Testing backward pass...")
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.logger.info("✅ Backward pass and optimizer step successful")
            
            # Test data loading
            self.logger.info("❓ Testing data loading...")
            batch_count = 0
            for batch in self.train_dataloader:
                batch_count += 1
                if batch_count >= 3:  # Test a few batches
                    break
            
            self.logger.info(f"✅ Data loading successful - Tested {batch_count} batches")
            
            # Test evaluation
            self.logger.info("❓ Testing evaluation...")
            eval_metrics = self.evaluate()
            self.logger.info(f"✅ Evaluation successful - Val Loss: {eval_metrics['eval_loss']:.4f}")
            
            # Test checkpointing
            self.logger.info("❓ Testing checkpointing...")
            test_checkpoint = self.save_checkpoint(0, is_milestone=True)
            self.logger.info(f"✅ Checkpointing successful - Saved: {test_checkpoint}")
            
            self.logger.info("🎉 All validation checks passed! Ready to train.")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return False
        
        finally:
            # Cleanup test tensors
            if 'input_ids' in locals():
                del input_ids
            if 'logits' in locals():
                del logits
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache()
            
    def train(
        self,
        num_epochs: int,
        train_steps: int = None,
        resume_from_checkpoint: Optional[str] = None,
        auto_resume: bool = True
    ):
        """
        Main training loop with comprehensive monitoring and checkpointing
        
        Args:
            num_epochs: Number of training epochs
            resume_from_checkpoint: Path to specific checkpoint to resume from
            auto_resume: Automatically resume from latest checkpoint if found
        """
        self.train_steps = train_steps
        # Resume training if requested
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        elif auto_resume:
            self.auto_resume()
        
        self.logger.info(f"Starting training for {num_epochs} epochs from step {self.global_step}")
        
        # Calculate total steps
        try:
            steps_per_epoch = len(self.train_dataloader)
        except:
            # Fallback for IterableDataset
            steps_per_epoch = 1000  # Conservative estimate
        
        # FIXED: Calculate total steps as minimum of (epochs * steps_per_epoch, max_steps)
        total_steps_from_epochs = steps_per_epoch * num_epochs
        total_steps = min(total_steps_from_epochs, self.max_steps) if self.train_steps is None else self.train_steps
        remaining_steps = total_steps - self.global_step
        
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"Total steps from epochs: {total_steps_from_epochs}")
        self.logger.info(f"Max steps limit: {self.max_steps}")
        self.logger.info(f"Final total steps: {total_steps}")
        self.logger.info(f"Current step: {self.global_step}")
        self.logger.info(f"Remaining steps: {remaining_steps}")
        
        if remaining_steps <= 0:
            self.logger.info("Training already completed!")
            return
        
        self.start_time = time.time()
        
        # Setup progress bar for total training steps
        pbar = tqdm(
            total=total_steps,
            desc="Training",
            initial=0,
            unit="steps",
            dynamic_ncols=True,
            leave=True
        )
        
        # FAST-FORWARD: If resuming, quickly update progress bar to current position
        if self.global_step > 0:
            pbar.write(f"Resuming from step {self.global_step}, fast-forwarding progress bar...")
            # Update progress bar quickly in chunks to avoid spam
            chunk_size = max(1, self.global_step // 100)  # Update in 1% chunks
            for i in range(0, self.global_step, chunk_size):
                update_amount = min(chunk_size, self.global_step - i)
                pbar.update(update_amount)
                if i % (chunk_size * 10) == 0:  # Show progress every 10%
                    pbar.set_postfix({'status': f'Resuming... {i}/{self.global_step}'}, refresh=True)
            
            # Final adjustment to exact position
            if pbar.n < self.global_step:
                pbar.update(self.global_step - pbar.n)
            
            pbar.set_postfix({}, refresh=True)  # Clear resume status
            pbar.write(f"Resumed at step {self.global_step}")
        
        try:
            # Calculate starting epoch based on global step
            current_epoch_start = self.global_step // steps_per_epoch
            
            for epoch in range(current_epoch_start, num_epochs):
                self.epoch = epoch
                epoch_start_time = time.time()
                epoch_loss = 0.0
                epoch_steps = 0
                steps_completed_in_epoch = 0
                
                self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Calculate how many steps already completed in this epoch
                steps_already_done_in_epoch = self.global_step - (epoch * steps_per_epoch)
                steps_already_done_in_epoch = max(0, steps_already_done_in_epoch)
                
                batch_accumulator = 0
                batch_iter = iter(self.train_dataloader)
                
                # FIXED: Use strict step counting within epoch
                while steps_completed_in_epoch < steps_per_epoch:
                    # Check if we've reached total step limit
                    if self.global_step >= total_steps:
                        pbar.write(f"Reached total step limit: {total_steps}")
                        break
                    
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        # Dataset exhausted, break out of epoch
                        pbar.write("Dataset exhausted in epoch")
                        break
                    
                    # Skip batches if we've already processed them (resume logic)
                    if steps_completed_in_epoch < steps_already_done_in_epoch:
                        steps_completed_in_epoch += 1
                        continue
                    
                    # Training step
                    step_loss, step_metrics = self.train_step(batch)
                    epoch_loss += step_loss
                    epoch_steps += 1
                    batch_accumulator += 1
                    
                    # Accumulate gradients
                    if batch_accumulator >= self.gradient_accumulation_steps:
                        grad_norm = self.optimizer_step()
                        batch_accumulator = 0
                        
                        # Update counters
                        self.global_step += 1
                        steps_completed_in_epoch += 1
                        
                        # Get current learning rate
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # Update training history
                        self.training_history['steps'].append(self.global_step)
                        self.training_history['train_loss'].append(step_loss)
                        self.training_history['learning_rate'].append(current_lr)
                        self.training_history['grad_norm'].append(grad_norm)
                        self.training_history['tokens_per_second'].append(step_metrics['tokens_per_sec'])
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': f"{step_loss:.4f}",
                            'lr': f"{current_lr:.2e}",
                            'grad_norm': f"{grad_norm:.3f}",
                            'tok/s': f"{step_metrics['tokens_per_sec']:.0f}"
                        }, refresh=False)
                        
                        # Detailed logging
                        if self.global_step % self.log_every == 0:
                            elapsed_time = time.time() - self.start_time
                            steps_per_second = self.global_step / elapsed_time if elapsed_time > 0 else 0
                            eta_seconds = (total_steps - self.global_step) / steps_per_second if steps_per_second > 0 else 0
                            
                            metrics = {
                                'step': self.global_step,
                                'epoch': epoch + 1,
                                'train_loss': step_loss,
                                'learning_rate': current_lr,
                                'grad_norm': grad_norm,
                                'tokens_per_sec': step_metrics['tokens_per_sec'],
                                'time_elapsed': str(timedelta(seconds=int(elapsed_time))),
                                'eta': str(timedelta(seconds=int(eta_seconds))),
                                'gpu_memory_gb': f"{self.get_memory_usage():.2f}"
                            }
                            
                            self.log_metrics(metrics)
                            
                            self.logger.info(
                                f"Step {self.global_step} | "
                                f"Loss: {step_loss:.4f} | "
                                f"LR: {current_lr:.2e} | "
                                f"Grad: {grad_norm:.3f} | "
                                f"Tok/s: {step_metrics['tokens_per_sec']:.0f} | "
                                f"GPU: {metrics['gpu_memory_gb']}GB"
                            )
                        
                        # Evaluation
                        if self.global_step % self.eval_every == 0:
                            pbar.write(f"Running evaluation at step {self.global_step}...")
                            eval_metrics = self.evaluate()
                            val_loss = eval_metrics['eval_loss']
                            perplexity = eval_metrics['eval_perplexity']
                            
                            # Update history
                            self.training_history['val_loss'].append(val_loss)
                            self.training_history['perplexity'].append(perplexity)
                            
                            # Check if this is the best model
                            is_best = val_loss < self.best_val_loss
                            if is_best:
                                self.best_val_loss = val_loss
                                pbar.write(f"New best model! Validation loss: {val_loss:.4f}")
                            
                            # Log evaluation metrics
                            elapsed_time = time.time() - self.start_time
                            steps_per_second = self.global_step / elapsed_time if elapsed_time > 0 else 0
                            eta_seconds = (total_steps - self.global_step) / steps_per_second if steps_per_second > 0 else 0
                            
                            eval_log_metrics = {
                                'step': self.global_step,
                                'epoch': epoch + 1,
                                'val_loss': val_loss,
                                'perplexity': perplexity,
                                'time_elapsed': str(timedelta(seconds=int(elapsed_time))),
                                'eta': str(timedelta(seconds=int(eta_seconds))),
                                'gpu_memory_gb': f"{self.get_memory_usage():.2f}"
                            }
                            self.log_metrics(eval_log_metrics)
                            
                            # Generate sample text
                            sample_text = self.generate_sample()
                            pbar.write(f"Sample: {sample_text}")
                        
                        # Checkpointing
                        if self.global_step % self.checkpoint_every == 0:
                            is_milestone = self.global_step % (self.checkpoint_every * 10) == 0
                            checkpoint_path = self.save_checkpoint(self.global_step, is_milestone=is_milestone)
                            pbar.write(f"Checkpoint saved: {checkpoint_path}")
                            
                            # Update plots
                            self.update_plots()
                        
                        # Break if we've reached total step limit
                        if self.global_step >= total_steps:
                            pbar.write(f"Reached total step limit: {total_steps}")
                            break
                
                # End of epoch summary
                avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
                epoch_time = time.time() - epoch_start_time
                
                pbar.write(
                    f"Epoch {epoch + 1} completed | "
                    f"Avg Loss: {avg_epoch_loss:.4f} | "
                    f"Time: {timedelta(seconds=int(epoch_time))} | "
                    f"Steps in epoch: {steps_completed_in_epoch}"
                )
                
                # Break if we've reached total step limit
                if self.global_step >= total_steps:
                    break
        
        except KeyboardInterrupt:
            pbar.write("\nTraining interrupted by user")
            self.save_checkpoint(self.global_step, is_milestone=True)
            self.update_plots()
            
        except Exception as e:
            pbar.write(f"\nTraining failed with error: {e}")
            self.save_checkpoint(self.global_step, is_milestone=True)
            raise
            
        finally:
            pbar.close()
            
            # Final checkpoint and plots if not already saved
            if self.global_step % self.checkpoint_every != 0:
                self.save_checkpoint(self.global_step, is_milestone=True)
            
            self.update_plots()
            
            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
            self.logger.info(f"Final step: {self.global_step}")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def train_lite(
        self,
        num_steps: int,
        val_steps: int = 20,
        resume_from_checkpoint: Optional[str] = None,
        auto_resume: bool = True
    ):
        """
        Simplified training loop matching the legacy pattern
        """
        
        # Resume training if requested
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        elif auto_resume:
            self.auto_resume()
        
        # Create simple data loaders
        
        
        # Determine batch size from existing dataloader
        test_batch = next(iter(self.train_dataloader))
        B = test_batch['input_ids'].shape[0]
        T = test_batch['input_ids'].shape[1]
        
        # Get data directory from dataloader
        data_dir = self.train_dataloader.dataset.shard_dir.parent
        
        train_loader = DataLoaderLite(str(data_dir), B, T, 'train')
        val_loader = DataLoaderLite(str(data_dir), B, T, 'val')
        
        self.logger.info(f"Starting training for {num_steps} steps from step {self.global_step}")
        self.logger.info(f"Batch size: {B}, Sequence length: {T}")
        self.logger.info(f"Effective batch size: {B * self.gradient_accumulation_steps}")
        
        self.start_time = time.time()
        
        # Setup progress bar
        pbar = tqdm(
            total=num_steps,
            desc="Training",
            initial=self.global_step,
            unit="steps",
            dynamic_ncols=True,
            leave=True
        )
        
        try:
            # Main training loop - EXACTLY like the legacy code
            for step in range(self.global_step, num_steps):
                t0 = time.time()
                last_step = (step == num_steps - 1)
                
                # Validation evaluation
                if step % self.eval_every == 0 or last_step:
                    self.model.eval()
                    val_loader.reset()
                    with torch.no_grad():
                        val_loss_accum = 0.0
                        for _ in range(val_steps):
                            x, y = val_loader.next_batch()
                            x, y = x.to(self.device), y.to(self.device)
                            
                            with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                                logits, loss = self.model(x, y)
                            
                            loss = loss / val_steps
                            val_loss_accum += loss.detach()
                            
                            # Clean up immediately
                            # del x, y, logits, loss
                    
                    # Log validation results
                    val_loss = val_loss_accum.item()
                    perplexity = self.calculate_perplexity(val_loss)
                    
                    self.logger.info(f"validation loss: {val_loss:.4f}, perplexity: {perplexity:.2f}")
                    
                    # Update history
                    self.training_history['val_loss'].append(val_loss)
                    self.training_history['perplexity'].append(perplexity)
                    
                    # Check if best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        pbar.write(f"New best model! Validation loss: {val_loss:.4f}")
                    
                    # Log to CSV
                    metrics = {
                        'step': step,
                        'val_loss': val_loss,
                        'perplexity': perplexity,
                        'gpu_memory_gb': f"{self.get_memory_usage():.2f}"
                    }
                    self.log_metrics(metrics)
                    
                    # Checkpointing
                    if step > 0 and (step % self.checkpoint_every == 0 or last_step):
                        is_milestone = step % (self.checkpoint_every * 5) == 0
                        checkpoint_path = self.save_checkpoint(step, is_milestone=is_milestone)
                        pbar.write(f"Checkpoint saved: {checkpoint_path}")
                        self.update_plots()
                
                # Training step
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0
                
                # Gradient accumulation micro-steps
                for micro_step in range(self.gradient_accumulation_steps):
                    x, y = train_loader.next_batch()
                    x, y = x.to(self.device), y.to(self.device)
                    
                    with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                        logits, loss = self.model(x, y)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    loss_accum += loss.detach()
                    
                    # Backward pass
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Clean up immediately after backward pass
                    # del x, y, logits, loss
                
                # Optimizer step
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Synchronize GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU to finish
                
                # Calculate metrics
                t1 = time.time()
                dt = t1 - t0  # Time difference in seconds
                tokens_processed = B * T * self.gradient_accumulation_steps
                tokens_per_sec = tokens_processed / dt
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update global step
                self.global_step = step + 1
                
                # Update history
                self.training_history['steps'].append(self.global_step)
                self.training_history['train_loss'].append(loss_accum.item())
                self.training_history['learning_rate'].append(current_lr)
                self.training_history['grad_norm'].append(grad_norm.item())
                self.training_history['tokens_per_second'].append(tokens_per_sec)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss_accum.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'grad_norm': f"{grad_norm.item():.3f}",
                    'tok/s': f"{tokens_per_sec:.0f}"
                }, refresh=False)
                
                # Detailed logging
                if step % self.log_every == 0:
                    elapsed_time = time.time() - self.start_time
                    gpu_memory_gb = self.get_memory_usage()
                    
                    self.logger.info(
                        f"step {step:5d} | loss: {loss_accum.item():.6f} | "
                        f"lr {current_lr:.4e} | norm: {grad_norm.item():.4f} | "
                        f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | "
                        f"GPU: {gpu_memory_gb:.2f}GB"
                    )
                    
                    # Log to CSV
                    metrics = {
                        'step': step,
                        'train_loss': loss_accum.item(),
                        'learning_rate': current_lr,
                        'grad_norm': grad_norm.item(),
                        'tokens_per_sec': tokens_per_sec,
                        'time_elapsed': str(timedelta(seconds=int(elapsed_time))),
                        'gpu_memory_gb': f"{gpu_memory_gb:.2f}"
                    }
                    self.log_metrics(metrics)
                
                # # Periodic GPU cleanup (much less aggressive)
                # if step % 100 == 0:
                #     torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            pbar.write("\nTraining interrupted by user")
            self.save_checkpoint(self.global_step, is_milestone=True)
            self.update_plots()
            
        except Exception as e:
            pbar.write(f"\nTraining failed with error: {e}")
            self.save_checkpoint(self.global_step, is_milestone=True)
            raise
            
        finally:
            pbar.close()
            
            # Final checkpoint
            if self.global_step % self.checkpoint_every != 0:
                self.save_checkpoint(self.global_step, is_milestone=True)
            
            self.update_plots()
            
            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
            self.logger.info(f"Final step: {self.global_step}")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
