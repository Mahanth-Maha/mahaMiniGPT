import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import math
import numpy as np
import time
import psutil
import gc
from typing import List


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    SequentialLR,
    OneCycleLR
)


from Transformer import DecoderOnlyTransformer

from trainer import TransformerTrainer

from train_utils import (
    TikTokenizer, 
    DatasetShardCreator, 
    DatasetLoader, 
    DataLoaderLite,
    ModelEvaluator,
    TextGenerator,
    validate_data_shards,
    estimate_dataset_size,
    find_best_checkpoint,
    load_prompts_from_file,
    save_generation_results
)


logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine",
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8
) -> tuple:
    """
    Create optimizer and learning rate scheduler with SOTA configurations
    """
    
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # No weight decay for biases, layer norms, and embeddings
            if any(nd in name for nd in ['bias', 'norm', 'embedding']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Create scheduler
    if scheduler_type.lower() == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=learning_rate * 0.1  # 10% of initial LR
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
    elif scheduler_type.lower() == "onecycle":
        # One cycle LR (good for shorter training runs)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=max_steps,
            pct_start=warmup_steps / max_steps,
            div_factor=25,
            final_div_factor=10000
        )
        
    elif scheduler_type.lower() == "linear":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=max_steps - warmup_steps
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
        
    elif scheduler_type.lower() == "none":
        scheduler = None
        
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    return optimizer, scheduler


def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration based on arguments"""
    
    model_configs = {
        "tiny": {
            "vocab_size": 50257,
            "context_length": 512,
            "n_embeddings": 128,
            "n_heads": 8,
            "Nx": 4,
            "ffn_hid_dim": 512
        },
        "small": {
            "vocab_size": 50257,
            "context_length": 1024,
            "n_embeddings": 768,
            "n_heads": 12,
            "Nx": 12,
            "ffn_hid_dim": 3072
        },
        "medium": {
            "vocab_size": 50257,
            "context_length": 1024,
            "n_embeddings": 1024,
            "n_heads": 16,
            "Nx": 24,
            "ffn_hid_dim": 4096
        },
        "large": {
            "vocab_size": 50257,
            "context_length": 1024,
            "n_embeddings": 1280,
            "n_heads": 20,
            "Nx": 36,
            "ffn_hid_dim": 5120
        }
    }
    
    if args.model_size in model_configs:
        config = model_configs[args.model_size].copy()
    else:
        # Custom configuration
        config = {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "n_embeddings": args.n_embeddings,
            "n_heads": args.n_heads,
            "Nx": args.n_layers,
            "ffn_hid_dim": args.ffn_hid_dim or (4 * args.n_embeddings)
        }
    
    # Override with any provided arguments
    if args.vocab_size != 50257:
        config["vocab_size"] = args.vocab_size
    if args.context_length != 1024:
        config["context_length"] = args.context_length
    
    # Add training-specific config
    config.update({
        "non_linearity": args.activation,
        "dropout": args.dropout,
        "device": args.device
    })
    
    return config


def validate_training_setup(
    args,
    model: nn.Module,
    tokenizer: TikTokenizer,
    data_dir: str,
    device: str,
    logger: logging.Logger
) -> bool:
    """
    Comprehensive validation of training setup
    Tests all components before starting full training
    """
    
    logger.info("🔍 Starting comprehensive training setup validation...")
    
    try:
        # 1. Validate data shards
        logger.info("1. Validating data shards...")
        if not validate_data_shards(logger, data_dir):
            logger.error("❌ Data shard validation failed")
            return False
        logger.info("✅ Data shards validation passed")
        
        # 2. Test tokenizer
        logger.info("2. Testing tokenizer...")
        test_text = "The quick brown fox jumps over the lazy dog."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        if not encoded or not decoded:
            logger.error("❌ Tokenizer encoding/decoding failed")
            return False
        
        logger.info(f"✅ Tokenizer test passed - {len(encoded)} tokens")
        logger.info(f"   Original: {test_text}")
        logger.info(f"   Decoded:  {decoded}")
        
        # 3. Test data loading
        logger.info("3. Testing data loading...")
        try:
            data_loader = DatasetLoader(
                logger = logger,
                data_dir=data_dir,
                batch_size=2,  # Small batch for testing
                max_length=512,
                num_workers=1
            )
            # data_loader = DataLoaderLite(
            #     data_dir=data_dir,
            #     batch_size=2,  # Small batch for testing
            #     max_length=512,
            #     num_workers=1
            # )
            
            train_dataloader = data_loader.get_train_dataloader()
            val_dataloader = data_loader.get_val_dataloader()
            
            # Test loading a few batches
            train_batch = next(iter(train_dataloader))
            val_batch = next(iter(val_dataloader))
            
            logger.info(f"✅ Data loading test passed")
            logger.info(f"   Train batch shape: {train_batch['input_ids'].shape}")
            logger.info(f"   Val batch shape: {val_batch['input_ids'].shape}")
            
        except Exception as e:
            logger.error(f"❌ Data loading test failed: {e}")
            return False
        
        # 4. Test model forward pass
        logger.info("4. Testing model forward pass...")
        model = model.to(device)
        model.train()
        
        # Use actual batch from data loader
        input_ids = train_batch['input_ids'].to(device)
        target_ids = train_batch['target_ids'].to(device)
        
        try:
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                logits, loss = model(input_ids, target_ids)
            
            logger.info(f"✅ Model forward pass test passed")
            logger.info(f"   Input shape: {input_ids.shape}")
            logger.info(f"   Logits shape: {logits.shape}")
            logger.info(f"   Loss: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Model forward pass test failed: {e}")
            return False
        
        # 5. Test backward pass and optimizer
        logger.info("5. Testing backward pass and optimizer...")
        try:
            # Create dummy optimizer for testing
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            logger.info(f"✅ Backward pass and optimizer test passed")
            logger.info(f"   Gradient norm: {grad_norm.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Backward pass test failed: {e}")
            return False
        
        # 6. Test model evaluation
        logger.info("6. Testing model evaluation...")
        try:
            evaluator = ModelEvaluator(
                logger,
                model, 
                tokenizer, 
                device, 
                mixed_precision=args.mixed_precision
            )
            
            # Test perplexity calculation on small subset
            val_dataloader_small = DatasetLoader(
                logger = logger,
                data_dir=data_dir,
                batch_size=2,
                max_length=512,
                num_workers=1
            ).get_val_dataloader()
            
            perplexity_metrics = evaluator.evaluate_perplexity(
                val_dataloader_small, 
                max_batches=2
            )
            
            logger.info(f"✅ Model evaluation test passed")
            logger.info(f"   Perplexity: {perplexity_metrics['eval_perplexity']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Model evaluation test failed: {e}")
            return False
        
        # 7. Test text generation
        logger.info("7. Testing text generation...")
        try:
            max_new_tokens=25
            sample_text = evaluator.generate_and_evaluate_text(
                prompts=["The future of AI"],
                max_new_tokens=max_new_tokens
            )
            logger.info(f"   evaluated: Generated: {sample_text['generated_texts'][0]}")
            
            test_text = "Virat Kohli is a Indian Cricket"
            encoded_tokens = torch.tensor(tokenizer.encode(test_text), dtype=torch.long,device=device).unsqueeze(0)
            inference_tokens = model.generate(encoded_tokens,max_new_tokens )[0].tolist()
            decoded_text = tokenizer.decode(inference_tokens)
            
            logger.info(f"   model: Generated: {decoded_text}")
            logger.info(f"✅ Text generation test passed")
            
        except Exception as e:
            logger.error(f"❌ Text generation test failed: {e}")
            return False
        
        # 8. Test GPU memory usage (if using CUDA)
        if device == 'cuda' and torch.cuda.is_available():
            logger.info("8. Testing GPU memory usage...")
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            logger.info(f"✅ GPU memory test passed")
            logger.info(f"   Allocated: {memory_allocated:.2f} GB")
            logger.info(f"   Reserved: {memory_reserved:.2f} GB")
        
        logger.info("🎉 All validation tests passed! Ready for training.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed with unexpected error: {e}")
        return False
    
    finally:
        # Cleanup
        if 'input_ids' in locals():
            del input_ids, target_ids, logits, loss
        torch.cuda.empty_cache()


def comprehensive_system_check(
    model: nn.Module,
    tokenizer: TikTokenizer,
    data_dir: str,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    args,
    device: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Comprehensive system check with detailed model, dataset, and training information
    """
    
    logger.info("🔍 Running comprehensive system check...")
    
    # Initialize results dictionary
    results = {}
    
    # 1. Model Information
    logger.info("📊 Analyzing model...")
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Model memory estimation (FP32)
    model_memory_fp32 = total_params * 4 / (1024**3)  # GB
    model_memory_fp16 = total_params * 2 / (1024**3)  # GB
    
    # Optimizer memory (AdamW stores 2 additional copies: momentum + variance)
    optimizer_memory = trainable_params * 8 / (1024**3) if args.optimizer.lower() == 'adamw' else trainable_params * 4 / (1024**3)
    
    model_info = {
        'architecture': f"{args.n_layers}L-{args.n_embeddings}H-{args.n_heads}A",
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_fp32_gb': model_memory_fp32,
        'model_size_fp16_gb': model_memory_fp16,
        'optimizer_memory_gb': optimizer_memory,
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'embedding_dim': args.n_embeddings,
        'attention_heads': args.n_heads,
        'layers': args.n_layers,
        'ffn_hidden_dim': args.ffn_hid_dim or (4 * args.n_embeddings),
        'activation': args.activation,
        'dropout': args.dropout
    }
    results['model'] = model_info
    
    # 2. Dataset Information (UPDATED)
    logger.info("📁 Analyzing dataset...")

    # Use the estimate_dataset_size function
    dataset_stats = estimate_dataset_size(logger, data_dir)

    # Load metadata for more detailed info
    metadata_path = Path(data_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        train_stats = metadata['statistics']['train']
        val_stats = metadata['statistics']['validation']
        
        dataset_info = {
            'train_shards': train_stats['num_shards'],
            'val_shards': val_stats['num_shards'],
            'train_sequences': train_stats['total_sequences'],
            'val_sequences': val_stats['total_sequences'],
            'train_tokens': train_stats['total_tokens'],
            'val_tokens': val_stats['total_tokens'],
            'total_tokens': train_stats['total_tokens'] + val_stats['total_tokens'],
            'avg_train_seq_length': train_stats['avg_sequence_length'],
            'avg_val_seq_length': val_stats['avg_sequence_length'],
            'shard_size': metadata['data_config']['shard_size'],
            'validation_split': metadata['data_config']['validation_split'],
            'max_sequence_length': metadata['tokenizer_config']['max_length'],
            'input_file': metadata['data_config'].get('input_file', 'Unknown'),
            'creation_date': metadata.get('creation_date', 'Unknown')
        }
    else:
        # Fallback to estimation
        logger.warning("No metadata found, using estimation...")
        dataset_info = dataset_stats
        dataset_info['estimated_only'] = True
        dataset_info['max_sequence_length'] = args.context_length

    results['dataset'] = dataset_info
    
    # 3. Training Configuration
    logger.info("⚙️ Analyzing training configuration...")
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = min(args.max_steps, steps_per_epoch * args.num_epochs)
    
    training_info = {
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': effective_batch_size,
        'num_epochs': args.num_epochs,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'mixed_precision': args.mixed_precision,
        'gradient_checkpointing': args.gradient_checkpointing,
        'compile_model': args.compile_model
    }
    results['training'] = training_info
    
    # 4. Memory Analysis
    logger.info("💾 Analyzing memory requirements...")
    
    # Batch memory estimation
    batch_tokens = args.batch_size * args.context_length
    batch_memory_fp32 = batch_tokens * 4 / (1024**3)  # Input + targets
    batch_memory_fp16 = batch_tokens * 2 / (1024**3)
    
    # Activation memory (rough estimate)
    activation_memory = batch_tokens * args.n_embeddings * args.n_layers * 4 / (1024**3)
    if args.gradient_checkpointing:
        activation_memory *= 0.3  # Significant reduction with gradient checkpointing
    
    # Total VRAM estimation
    precision_multiplier = 2 if args.mixed_precision else 4
    total_model_memory = model_memory_fp16 if args.mixed_precision else model_memory_fp32
    
    peak_vram = (
        total_model_memory +  # Model weights
        optimizer_memory +    # Optimizer states
        batch_memory_fp16 if args.mixed_precision else batch_memory_fp32 +  # Batch data
        activation_memory +   # Activations
        1.0  # Buffer for other operations
    )
    
    memory_info = {
        'batch_memory_gb': batch_memory_fp16 if args.mixed_precision else batch_memory_fp32,
        'activation_memory_gb': activation_memory,
        'peak_vram_gb': peak_vram,
        'gradient_checkpointing_enabled': args.gradient_checkpointing,
        'mixed_precision_enabled': args.mixed_precision
    }
    results['memory'] = memory_info
    
    # 5. Performance Estimation
    logger.info("⏱️ Estimating training performance...")
    
    # Run 5 test iterations to estimate timing
    model.train()
    torch.cuda.empty_cache()
    
    test_times = []
    test_batch = next(iter(train_dataloader))
    
    # Warmup
    for _ in range(2):
        input_ids = test_batch['input_ids'][:min(2, args.batch_size)].to(device)
        target_ids = test_batch['target_ids'][:min(2, args.batch_size)].to(device)
        
        with torch.amp.autocast('cuda', enabled=args.mixed_precision):
            logits, loss = model(input_ids, target_ids)
        
        if args.mixed_precision:
            from torch.amp import GradScaler
            scaler = GradScaler()
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        optimizer.zero_grad()
        del input_ids, target_ids, logits, loss
        torch.cuda.empty_cache()
    
    # Actual timing tests
    for i in range(5):
        input_ids = test_batch['input_ids'][:min(args.batch_size, test_batch['input_ids'].size(0))].to(device)
        target_ids = test_batch['target_ids'][:min(args.batch_size, test_batch['target_ids'].size(0))].to(device)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.amp.autocast('cuda', enabled=args.mixed_precision):
            logits, loss = model(input_ids, target_ids)
        
        if args.mixed_precision:
            scaler = torch.amp.GradScaler('cuda')
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        optimizer.zero_grad()
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        test_times.append(end_time - start_time)
        
        del input_ids, target_ids, logits, loss
        torch.cuda.empty_cache()
    
    avg_step_time = sum(test_times) / len(test_times)
    tokens_per_second = (args.batch_size * args.context_length) / avg_step_time
    
    # Time estimations
    total_time_seconds = total_steps * avg_step_time
    total_time_hours = total_time_seconds / 3600
    total_time_days = total_time_hours / 24
    
    performance_info = {
        'avg_step_time_seconds': avg_step_time,
        'tokens_per_second': tokens_per_second,
        'steps_per_hour': 3600 / avg_step_time,
        'estimated_total_time_hours': total_time_hours,
        'estimated_total_time_days': total_time_days,
        'test_iterations': len(test_times),
        'step_time_std': np.std(test_times) if len(test_times) > 1 else 0
    }
    results['performance'] = performance_info
    
    # 6. System Information
    logger.info("💻 Gathering system information...")
    
    # CPU info
    cpu_info = {
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    # GPU info
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count()
        }
    
    system_info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'device': str(device),
        **cpu_info,
        **gpu_info
    }
    results['system'] = system_info
    
    # 7. Warnings and Recommendations
    logger.info("⚠️ Analyzing potential issues...")
    
    warnings = []
    recommendations = []
    
    # Memory warnings
    if gpu_info and peak_vram > gpu_info['gpu_memory_total_gb'] * 0.9:
        warnings.append(f"Peak VRAM ({peak_vram:.1f}GB) may exceed GPU memory ({gpu_info['gpu_memory_total_gb']:.1f}GB)")
        recommendations.append("Consider reducing batch size, enabling gradient checkpointing, or using mixed precision")
    
    # Performance warnings
    if tokens_per_second < 1000:
        warnings.append(f"Low throughput: {tokens_per_second:.0f} tokens/sec")
        recommendations.append("Consider enabling mixed precision, gradient checkpointing, or model compilation")
    
    # Time warnings
    if total_time_days > 7:
        warnings.append(f"Training will take {total_time_days:.1f} days")
        recommendations.append("Consider using a larger GPU, reducing model size, or fewer training steps")
    
    # Dataset warnings
    if 'total_tokens' in dataset_info and dataset_info['total_tokens'] < total_params * 20:
        warnings.append("Dataset size may be too small for model size (rule of thumb: 20+ tokens per parameter)")
        recommendations.append("Consider using a larger dataset or smaller model")
    
    results['warnings'] = warnings
    results['recommendations'] = recommendations
    
    # Print comprehensive report
    print_system_report(results, logger)
    
    return results


def print_system_report(results: Dict[str, Any], logger: logging.Logger):
    """Print a comprehensive system report"""
    
    logger.info("=" * 80)
    logger.info("🚀 COMPREHENSIVE SYSTEM CHECK REPORT")
    logger.info("=" * 80)
    
    # Model Information
    model = results['model']
    logger.info("\n📊 MODEL INFORMATION:")
    logger.info(f"  Architecture: {model['architecture']}")
    logger.info(f"  Total Parameters: {model['total_parameters']:,}")
    logger.info(f"  Trainable Parameters: {model['trainable_parameters']:,}")
    logger.info(f"  Model Size (FP32): {model['model_size_fp32_gb']:.2f} GB")
    logger.info(f"  Model Size (FP16): {model['model_size_fp16_gb']:.2f} GB")
    logger.info(f"  Optimizer Memory: {model['optimizer_memory_gb']:.2f} GB")
    logger.info(f"  Context Length: {model['context_length']:,}")
    logger.info(f"  Vocabulary Size: {model['vocab_size']:,}")
    
    # Dataset Information
    dataset = results['dataset']
    logger.info("\n📁 DATASET INFORMATION:")
    logger.info(f"  Train Shards: {dataset['train_shards']}")
    logger.info(f"  Validation Shards: {dataset['val_shards']}")
    if 'total_tokens' in dataset:
        logger.info(f"  Total Tokens: {dataset['total_tokens']:,}")
        logger.info(f"  Train Tokens: {dataset['train_tokens']:,}")
        logger.info(f"  Validation Tokens: {dataset['val_tokens']:,}")
        logger.info(f"  Train Sequences: {dataset['train_sequences']:,}")
        logger.info(f"  Validation Sequences: {dataset['val_sequences']:,}")
        logger.info(f"  Avg Sequence Length: {dataset['avg_train_seq_length']:.1f}")
    
    # Training Configuration
    training = results['training']
    logger.info("\n⚙️ TRAINING CONFIGURATION:")
    logger.info(f"  Batch Size: {training['batch_size']}")
    logger.info(f"  Gradient Accumulation: {training['gradient_accumulation_steps']}")
    logger.info(f"  Effective Batch Size: {training['effective_batch_size']}")
    logger.info(f"  Learning Rate: {training['learning_rate']:.2e}")
    logger.info(f"  Weight Decay: {training['weight_decay']}")
    logger.info(f"  Warmup Steps: {training['warmup_steps']:,}")
    logger.info(f"  Total Steps: {training['total_steps']:,}")
    logger.info(f"  Optimizer: {training['optimizer'].upper()}")
    logger.info(f"  Scheduler: {training['scheduler'].upper()}")
    logger.info(f"  Mixed Precision: {training['mixed_precision']}")
    logger.info(f"  Gradient Checkpointing: {training['gradient_checkpointing']}")
    
    # Memory Analysis
    memory = results['memory']
    logger.info("\n💾 MEMORY ANALYSIS:")
    logger.info(f"  Batch Memory: {memory['batch_memory_gb']:.2f} GB")
    logger.info(f"  Activation Memory: {memory['activation_memory_gb']:.2f} GB")
    logger.info(f"  Peak VRAM Required: {memory['peak_vram_gb']:.2f} GB")
    
    # Performance Estimation
    perf = results['performance']
    logger.info("\n⏱️ PERFORMANCE ESTIMATION:")
    logger.info(f"  Avg Step Time: {perf['avg_step_time_seconds']:.3f} seconds")
    logger.info(f"  Tokens/Second: {perf['tokens_per_second']:.0f}")
    logger.info(f"  Steps/Hour: {perf['steps_per_hour']:.1f}")
    logger.info(f"  Estimated Total Time: {perf['estimated_total_time_hours']:.1f} hours ({perf['estimated_total_time_days']:.1f} days)")
    
    # System Information
    system = results['system']
    logger.info("\n💻 SYSTEM INFORMATION:")
    logger.info(f"  Device: {system['device']}")
    logger.info(f"  CPU Cores: {system['cpu_count']} physical, {system['cpu_count_logical']} logical")
    logger.info(f"  System RAM: {system['memory_total_gb']:.1f} GB total, {system['memory_available_gb']:.1f} GB available")
    logger.info(f"  PyTorch Version: {system['torch_version']}")
    
    if 'gpu_name' in system:
        logger.info(f"  GPU: {system['gpu_name']}")
        logger.info(f"  GPU Memory: {system['gpu_memory_total_gb']:.1f} GB total")
        logger.info(f"  CUDA Version: {system['cuda_version']}")
    
    # Warnings and Recommendations
    if results['warnings']:
        logger.info("\n⚠️ WARNINGS:")
        for warning in results['warnings']:
            logger.warning(f"  • {warning}")
    
    if results['recommendations']:
        logger.info("\n💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            logger.info(f"  • {rec}")
    
    logger.info("\n" + "=" * 80)


def prepare_data(args, tokenizer: TikTokenizer, logger: logging.Logger) -> str:
    """Prepare data shards if they don't exist"""
    
    data_dir = Path(args.data_dir)
    
    # Check if shards already exist
    if data_dir.exists() and validate_data_shards(logger, str(data_dir)):
        logger.info(f"Using existing data shards in {data_dir}")
        return str(data_dir)
    
    # Create shards from raw text file
    if not args.text_file:
        raise ValueError("Text file must be provided for data preparation")
    
    if not Path(args.text_file).exists():
        raise FileNotFoundError(f"Text file not found: {args.text_file}")
    
    logger.info(f"Creating data shards from {args.text_file}")
    
    shard_creator = DatasetShardCreator(
        logger=logger,
        tokenizer=tokenizer,
        shard_size=args.shard_size,
        max_length=args.context_length,
        num_processes=args.num_workers
    )
    
    metadata = shard_creator.create_shards(
        input_file=args.text_file,
        output_dir=str(data_dir),
        validation_split=args.val_split
    )
    
    logger.info("Data preparation completed")
    logger.info(f"Train shards: {metadata['statistics']['train']['num_shards']}")
    logger.info(f"Val shards: {metadata['statistics']['validation']['num_shards']}")
    
    return str(data_dir)


def main():

    parser = argparse.ArgumentParser(description="Train Transformer LLM")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["tiny", "small", "medium", "large", "custom"],
                       help=f"Predefined model size")
    parser.add_argument("--vocab-size", type=int, default=50257,
                       help=f"Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512,
                       help=f"Maximum context length")
    parser.add_argument("--n-embeddings", type=int, default=128,
                       help=f"Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=8,
                       help=f"Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4,
                       help=f"Number of transformer layers")
    parser.add_argument("--ffn-hid-dim", type=int, default=None,
                       help=f"FFN hidden dimension (default: 4 * n_embeddings)")
    parser.add_argument("--activation", type=str, default="gelu",
                       choices=["relu", "gelu", "silu", "leakyrelu", "softplus"],
                       help=f"Activation function")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help=f"Dropout rate")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32,
                       help=f"Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help=f"Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help=f"Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                       help=f"Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help=f"Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                       help=f"Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8,
                       help=f"Adam epsilon")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help=f"Gradient clipping threshold")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                       help=f"Warmup steps")
    parser.add_argument("--max-steps", type=int, default=100000,
                       help=f"Maximum training steps")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help=f"Number of training epochs")
    parser.add_argument("--num-steps", type=int, default=10000,
                       help=f"Number of training steps")
    
    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adamw", "adam", "sgd"],
                       help=f"Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "linear", "onecycle", "none"],
                       help=f"Learning rate scheduler")
    
    # Data configuration
    parser.add_argument("--text-file", type=str, default=None,
                       help=f"Input text file for training")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help=f"Directory containing or to store data shards")
    parser.add_argument("--shard-size", type=int, default=1048576,
                       help=f"Number of tokens per shard")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help=f"Validation split ratio")
    parser.add_argument("--num-workers", type=int, default=1,
                       help=f"Number of data loading workers")

    # Training control
    parser.add_argument("--output-dir", type=str, default="./training_output",
                       help=f"Output directory for checkpoints and logs")
    parser.add_argument("--checkpoint-every", type=int, default=10000,
                       help=f"Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=5000,
                       help=f"Evaluate every N steps")
    parser.add_argument("--log-every", type=int, default=1000,
                       help=f"Log every N steps")
    parser.add_argument("--max-checkpoints", type=int, default=5,
                       help=f"Maximum number of checkpoints to keep")
    
    # System configuration
    parser.add_argument("--device", type=str, default="cuda",
                       help=f"Training device")
    parser.add_argument("--mixed-precision", action="store_true", default=False,
                       help=f"Use mixed precision training")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                       help=f"Use gradient checkpointing")
    parser.add_argument("--compile-model", action="store_true", default=False,
                       help=f"Use torch.compile for optimization")
    
    # Validation and testing
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--check", action="store_true",
                        help=f"Run comprehensive system check and analysis")
    mode_group.add_argument("--train", action="store_true",
                        help=f"Start training")
    mode_group.add_argument("--validate-only", action="store_true",
                        help=f"Only run validation checks, don`t train")
    parser.add_argument("--resume-from", type=str, default=None,
                       help=f"Resume training from checkpoint")
    parser.add_argument("--auto-resume", action="store_true", default=True,
                       help=f"Automatically resume from latest checkpoint")
    
    # GPU optimization arguments
    mode_group.add_argument("--get-optimal-bs", action="store_true",
                    help=f"Find optimal batch size for GPU")
    mode_group.add_argument("--test-batch-sizes", type=int, nargs='+', default=None,
                    help=f"Specific batch sizes to test (e.g., --test-batch-sizes 8 16 32)")
    mode_group.add_argument("--target-batch-size", type=int, default=None,
                    help=f"Test specific batch size")
    mode_group.add_argument("--gpu-safety-margin", type=float, default=0.05,
                    help=f"GPU memory safety margin (default: 0.05 = 5%)")
    mode_group.add_argument("--skip-train", action="store_true")

    # Generation mode arguments
    generation_group = parser.add_argument_group('Generation Mode')
    generation_group.add_argument("--generate", action="store_true",
                                help="Run text generation mode (load best model and generate)")
    generation_group.add_argument("--checkpoint-path", type=str, default=None,
                                help="Specific checkpoint path to load for generation")
    generation_group.add_argument("--prompts", type=str, nargs='+', 
                                default=["The future of AI is", "Machine learning will", "Deep learning has"],
                                help="List of prompts for generation")
    generation_group.add_argument("--prompts-file", type=str, default=None,
                                help="File containing prompts (one per line)")
    generation_group.add_argument("--max-new-tokens", type=int, default=100,
                                help="Maximum tokens to generate per prompt")
    generation_group.add_argument("--generation-batch-size", type=int, default=8,
                                help="Batch size for generation")
    generation_group.add_argument("--temperature", type=float, default=1.0,
                                help="Temperature for generation")
    generation_group.add_argument("--top-k", type=int, default=10000,
                                help="Top-k sampling parameter")
    generation_group.add_argument("--top-p", type=float, default=0.0,
                                help="Top-p (nucleus) sampling parameter")
    generation_group.add_argument("--do-sample", action="store_true", default=True,
                                help="Use sampling for generation")
    generation_group.add_argument("--num-return-sequences", type=int, default=1,
                                help="Number of sequences to generate per prompt")
    generation_group.add_argument("--output-file", type=str, default=None,
                                help="File to save generation results")

    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help=f"Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Print arguments
    logger.info("🚀 Starting Transformer Training")
    logger.info("Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        if args.generate:
            logger.info("🎯 Running in generation mode")
            
            # Initialize tokenizer
            tokenizer = TikTokenizer(logger, encoding_name="gpt2", vocab_size=args.vocab_size)
            
            # Create model
            model_config = create_model_config(args)
            model = DecoderOnlyTransformer(**model_config)
            
            # Find and load best checkpoint
            if args.checkpoint_path:
                checkpoint_path = args.checkpoint_path
            else:
                checkpoint_path = find_best_checkpoint(logger, args.output_dir)
            
            if not checkpoint_path:
                logger.error("No checkpoint found for generation!")
                return 1
            
            # Load checkpoint
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ Checkpoint loaded successfully (step: {checkpoint.get('step', 'unknown')})")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return 1
            
            # Get prompts
            if args.prompts_file:
                prompts = load_prompts_from_file(logger, args.prompts_file)
                if not prompts:
                    logger.error("No prompts loaded from file!")
                    return 1
            else:
                prompts = args.prompts
            
            logger.info(f"Loaded {len(prompts)} prompts for generation")
            
            # Create generation config
            generation_config = {
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'top_k': args.top_k,
                'top_p': args.top_p,
                'do_sample': args.do_sample
            }
            
            # Create text generator
            
            generator = TextGenerator(logger, model, tokenizer, args.device, generation_config)
            
            # Interactive mode or batch generation
            if not args.output_file and len(prompts) <= 5:
                # Interactive mode for small number of prompts
                print(f"\n📝 Sample generations:")
                results = generator.generate_batch(prompts, args.generation_batch_size, args.num_return_sequences)
                
                for i, (prompt, generated_text) in enumerate(zip(results['prompts'], results['generated_texts'])):
                    print(f"\n🔸 Prompt {i+1}: {prompt}")
                    print(f"🔹 Generated: {generated_text}")
                
                print(f"\n📊 Stats: {results['throughput']:.2f} texts/sec, avg time: {results['avg_generation_time']:.3f}s")
                
                # Ask if user wants interactive mode
                while True:
                    try:
                        response = input("\n🤖 Enter interactive mode? (y/n): ").strip().lower()
                        if response == 'y':
                            generator.generate_interactive()
                        break
                    except KeyboardInterrupt:
                        break
            else:
                # Batch generation mode
                logger.info("🚀 Starting batch generation...")
                results = generator.generate_batch(prompts, args.generation_batch_size, args.num_return_sequences)
                
                logger.info(f"✅ Generation completed!")
                logger.info(f"📊 Generated {len(results['generated_texts'])} texts in {results['total_generation_time']:.2f}s")
                logger.info(f"📈 Throughput: {results['throughput']:.2f} texts/sec")
                
                # Save results
                if args.output_file:
                    save_generation_results(logger, results, args.output_file)
                else:
                    # Print first few results
                    for i, (prompt, generated_text) in enumerate(zip(results['prompts'][:3], results['generated_texts'][:3])):
                        print(f"\n🔸 Prompt {i+1}: {prompt}")
                        print(f"🔹 Generated: {generated_text}")
                    
                    if len(results['generated_texts']) > 3:
                        print(f"\n... and {len(results['generated_texts']) - 3} more results")
            
            logger.info("🎉 Generation mode completed!")
            return 0

        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = TikTokenizer( logger, encoding_name="gpt2", vocab_size=args.vocab_size)
        
        # Prepare data
        logger.info("Preparing data...")
        data_dir = prepare_data(args, tokenizer, logger)
        
        # Create model
        logger.info("Creating model...")
        model_config = create_model_config(args)
        logger.info(f"Model configuration: {json.dumps(model_config, indent=2)}")
        
        model = DecoderOnlyTransformer(**model_config)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {param_count:,} trainable parameters")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loader = DatasetLoader(
            logger = logger,
            data_dir=data_dir,
            batch_size=args.batch_size,
            max_length=args.context_length,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda")
        )
        
        train_dataloader = data_loader.get_train_dataloader()
        val_dataloader = data_loader.get_val_dataloader()
        
        # Calculate effective batch size and steps
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        total_steps = min(args.max_steps, steps_per_epoch * args.num_epochs)
        
        # Create optimizer and scheduler
        logger.info("Creating optimizer and scheduler...")
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=total_steps,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps
        )
        
        # Handle GPU optimization mode
        if args.get_optimal_bs:
            logger.info("🔍 Running GPU optimization and batch size analysis...")
            
            # Import the function
            from train_utils import find_optimal_batch_size, print_batch_size_analysis
            
            # Run batch size optimization
            optimization_results = find_optimal_batch_size(
                logger = logger,
                model_config=model_config,
                tokenizer=tokenizer,
                data_dir=data_dir,
                device=args.device,
                mixed_precision=args.mixed_precision,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                context_length=args.context_length,
                test_batch_sizes=args.test_batch_sizes,
                target_batch_size=args.target_batch_size,
                safety_margin=args.gpu_safety_margin
            )
            
            # Print detailed analysis
            print_batch_size_analysis(optimization_results)
            
            # Save results to file
            results_file = Path(args.output_dir) / "gpu_optimization_results.json"
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            
            logger.info(f"Optimization results saved to: {results_file}")
            
            # Suggest optimal batch size for current run
            optimal_bs = optimization_results.get('optimal_batch_size')
            if optimal_bs and optimal_bs != args.batch_size:
                logger.info(f"💡 Suggestion: Use --batch-size {optimal_bs} for optimal performance")
            
            return 0  # Exit after optimization

        # Handle different modes
        elif args.check:
            # Run comprehensive system check
            logger.info("Running comprehensive system check...")
            comprehensive_system_check(
                model=model,
                tokenizer=tokenizer,
                data_dir=data_dir,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                device=args.device,
                logger=logger
            )
            logger.info("✅ System check completed!")
            return 0
            
        elif args.validate_only:
            # Run validation checks only
            logger.info("Running validation checks only...")
            success = validate_training_setup(args, model, tokenizer, data_dir, args.device, logger)
            if success:
                logger.info("✅ All validation checks passed!")
                return 0
            else:
                logger.error("❌ Validation failed!")
                return 1
                
        elif args.train:
            # Create trainer and start training
            logger.info("Creating trainer...")
            trainer = TransformerTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                logger=logger,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                output_dir=args.output_dir,
                max_checkpoints=args.max_checkpoints,
                checkpoint_every=args.checkpoint_every,
                eval_every=args.eval_every,
                log_every=args.log_every,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                device=args.device,
                mixed_precision=args.mixed_precision,
                gradient_checkpointing=args.gradient_checkpointing,
                compile_model=args.compile_model,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                vocab_size = args.vocab_size,
                batch_size = args.batch_size,
            )
            
            # Validate setup before training
            logger.info("Validating training setup...")
            if not trainer.validate_setup():
                logger.error("❌ Training setup validation failed!")
                return 1
            
            # Start training
            logger.info("🤞 Starting training...")
            trainer.train(
                num_epochs=args.num_epochs,
                train_steps=args.num_steps,
                resume_from_checkpoint=args.resume_from,
                auto_resume=args.auto_resume
            )
            # trainer.train_lite(
            #     num_steps=args.num_steps,
            #     val_steps=20,
            #     auto_resume=args.auto_resume
            # )
            
            logger.info("🎉 Training completed successfully!")
            return 0

    
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    code_ex = main()
    if isinstance(code_ex,int):
        print(f'👍 [DONE] Scipt exited with CODE-{code_ex}')
        exit(code_ex)