usage: main.py [-h] [--model-size {tiny,small,medium,large,custom}] [--vocab-size VOCAB_SIZE] [--context-length CONTEXT_LENGTH]
               [--n-embeddings N_EMBEDDINGS] [--n-heads N_HEADS] [--n-layers N_LAYERS] [--ffn-hid-dim FFN_HID_DIM]
               [--activation {relu,gelu,silu,leakyrelu,softplus}] [--dropout DROPOUT] [--batch-size BATCH_SIZE]
               [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY] [--beta1 BETA1]
               [--beta2 BETA2] [--eps EPS] [--max-grad-norm MAX_GRAD_NORM] [--warmup-steps WARMUP_STEPS] [--max-steps MAX_STEPS] [--num-epochs NUM_EPOCHS]
               [--optimizer {adamw,adam,sgd}] [--scheduler {cosine,linear,onecycle,none}] [--text-file TEXT_FILE] [--data-dir DATA_DIR]
               [--shard-size SHARD_SIZE] [--val-split VAL_SPLIT] [--num-workers NUM_WORKERS] [--output-dir OUTPUT_DIR]
               [--checkpoint-every CHECKPOINT_EVERY] [--eval-every EVAL_EVERY] [--log-every LOG_EVERY] [--max-checkpoints MAX_CHECKPOINTS]
               [--device DEVICE] [--mixed-precision] [--gradient-checkpointing] [--compile-model] [--check] [--train] [--validate-only]
               [--resume-from RESUME_FROM] [--auto-resume] [--get-optimal-bs] [--test-batch-sizes TEST_BATCH_SIZES [TEST_BATCH_SIZES ...]]
               [--target-batch-size TARGET_BATCH_SIZE] [--gpu-safety-margin GPU_SAFETY_MARGIN] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Train Transformer LLM

options:
  -h, --help            show this help message and exit
  
  --model-size {tiny,small,medium,large,custom} Predefined model size
  --vocab-size VOCAB_SIZE Vocabulary size
  --context-length CONTEXT_LENGTH Maximum context length
  --n-embeddings N_EMBEDDINGS Embedding dimension
  --n-heads N_HEADS     Number of attention heads 
  --n-layers N_LAYERS   Number of transformer layers 
  --ffn-hid-dim FFN_HID_DIM FFN hidden dimension (default: 4 * n_embeddings)
  --activation {relu,gelu,silu,leakyrelu,softplus} Activation function
  --dropout DROPOUT     Dropout rate 
  --batch-size BATCH_SIZE Batch size
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS Gradient accumulation steps
  --learning-rate LEARNING_RATE Learning rate
  --weight-decay WEIGHT_DECAY Weight decay
  --beta1 BETA1         Adam beta1 
  --beta2 BETA2         Adam beta2 
  --eps EPS             Adam epsilon 
  --max-grad-norm MAX_GRAD_NORM Gradient clipping threshold
  --warmup-steps WARMUP_STEPS Warmup steps
  --max-steps MAX_STEPS Maximum training steps
  --num-epochs NUM_EPOCHS Number of training epochs
  --optimizer {adamw,adam,sgd} Optimizer type
  --scheduler {cosine,linear,onecycle,none} Learning rate scheduler
  --text-file TEXT_FILE Input text file for training
  --data-dir DATA_DIR   Directory containing or to store data shards 
  --shard-size SHARD_SIZE Number of tokens per shard
  --val-split VAL_SPLIT Validation split ratio
  --num-workers NUM_WORKERS Number of data loading workers
  --output-dir OUTPUT_DIR Output directory for checkpoints and logs
  --checkpoint-every CHECKPOINT_EVERY Save checkpoint every N steps
  --eval-every EVAL_EVERY Evaluate every N steps
  --log-every LOG_EVERY Log every N steps
  --max-checkpoints MAX_CHECKPOINTS Maximum number of checkpoints to keep
  --device DEVICE       Training device 
  --mixed-precision     Use mixed precision training 
  --gradient-checkpointing Use gradient checkpointing
  --compile-model       Use torch.compile for optimization 
  --check               Run comprehensive system check and analysis 
  --train               Start training 
  --validate-only       Only run validation checks, don`t train 
  --resume-from RESUME_FROM Resume training from checkpoint
  --auto-resume         Automatically resume from latest checkpoint 
  --get-optimal-bs      Find optimal batch size for GPU 
  --test-batch-sizes TEST_BATCH_SIZES [TEST_BATCH_SIZES ...] Specific batch sizes to test (e.g., test-batch-sizes 8 16 32)
  --target-batch-size TARGET_BATCH_SIZE Test specific batch size
  --gpu-safety-margin GPU_SAFETY_MARGIN GPU memory safety margin (default: 0.15 = 15%)
  --log-level {DEBUG,INFO,WARNING,ERROR} Logging level