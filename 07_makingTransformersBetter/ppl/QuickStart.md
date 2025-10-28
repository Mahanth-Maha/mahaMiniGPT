## Complete System Flow & Quick Start Guide:

### 1. **File Structure Setup:**
```
your_project/
├── transformer_model.py  # Your existing transformer code
├── trainer.py           # Training module (provided)
├── train_utils.py       # Dataset utilities (provided)
├── main.py             # Main script (provided)
├── data/               # Will be created automatically
└── training_output/    # Will be created automatically
```

### 2. **Dependencies Installation:**
```bash
pip install torch torchvision torchaudio
pip install tiktoken
pip install matplotlib
pip install nltk
pip install tqdm
pip install psutil
pip install numpy
```

### 3. **Data Preparation:**
Create a text file with your training data:
```bash
# Example: create a sample text file
echo "This is sample text for training. The quick brown fox jumps over the lazy dog. 
Artificial intelligence is transforming the world. 
Machine learning models require large datasets for training." > sample_data.txt
```

### 4. **Quick Start Commands:**

#### **Step 1: System Check (RECOMMENDED FIRST)**
```bash
python main.py --check \
    --text-file sample_data.txt \
    --model-size small \
    --batch-size 8 \
    --gradient-accumulation-steps 2
```

**What this does:**
- Analyzes your system capabilities
- Creates/validates data shards
- Tests model forward/backward pass 5 times
- Estimates memory requirements
- Calculates training time
- Provides warnings and recommendations

#### **Step 2: Validation Only (Optional)**
```bash
python main.py --validate-only \
    --text-file sample_data.txt \
    --model-size small \
    --batch-size 8
```

**What this does:**
- Tests all components work together
- Validates data loading
- Tests model functionality
- No performance benchmarking

#### **Step 3: Start Training**
```bash
python main.py --train \
    --text-file sample_data.txt \
    --model-size small \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --num-epochs 1 \
    --learning-rate 3e-4 \
    --mixed-precision \
    --gradient-checkpointing
```

### 5. **Detailed System Flow:**

```
1. INITIALIZATION PHASE
   ├── Parse arguments
   ├── Setup logging
   ├── Check device availability (CUDA/CPU)
   └── Initialize tokenizer (tiktoken GPT-2)

2. DATA PREPARATION PHASE
   ├── Check if shards exist in --data-dir
   ├── If not exist: Create shards from --text-file
   │   ├── Read text file
   │   ├── Split into train/validation
   │   ├── Tokenize in parallel processes
   │   ├── Create memory-mapped shards
   │   └── Save metadata.json
   └── Validate shard accessibility

3. MODEL CREATION PHASE
   ├── Create model configuration (predefined or custom)
   ├── Instantiate DecoderOnlyTransformer
   ├── Count parameters
   └── Move to device

4. TRAINING SETUP PHASE
   ├── Create data loaders (train/validation)
   ├── Setup optimizer (AdamW with parameter groups)
   ├── Setup scheduler (cosine with warmup)
   └── Calculate training steps

5. MODE EXECUTION PHASE
   ├── --check: Comprehensive analysis + benchmarking
   ├── --validate-only: Component testing
   └── --train: Full training loop

6. TRAINING LOOP (if --train)
   ├── Load/resume checkpoints
   ├── For each epoch:
   │   ├── For each batch:
   │   │   ├── Forward pass (with mixed precision)
   │   │   ├── Backward pass
   │   │   ├── Gradient accumulation
   │   │   ├── Optimizer step (with clipping)
   │   │   ├── Scheduler step
   │   │   └── Clear GPU memory
   │   ├── Periodic evaluation
   │   ├── Checkpoint saving
   │   ├── Progress logging
   │   └── Plot generation
   └── Final checkpoint and cleanup
```

### 6. **Configuration Options:**

#### **Model Sizes (Predefined):**
```python
"tiny": 41M parameters   (768 hidden, 12 layers, 12 heads)
"small": 117M parameters (768 hidden, 12 layers, 12 heads, 1024 context)
"medium": 345M parameters (1024 hidden, 24 layers, 16 heads)
"large": 774M parameters (1280 hidden, 36 layers, 20 heads)
```

#### **Memory Optimization Settings:**
```bash
# For low VRAM (4-8GB):
--batch-size 4 --gradient-accumulation-steps 8 --mixed-precision --gradient-checkpointing

# For medium VRAM (8-16GB):
--batch-size 16 --gradient-accumulation-steps 2 --mixed-precision --gradient-checkpointing

# For high VRAM (16GB+):
--batch-size 32 --gradient-accumulation-steps 1 --mixed-precision
```

### 7. **Expected Output from --check:**

```
🚀 COMPREHENSIVE SYSTEM CHECK REPORT
================================================================================

📊 MODEL INFORMATION:
  Architecture: 12L-768H-12A
  Total Parameters: 117,477,632
  Trainable Parameters: 117,477,632
  Model Size (FP32): 0.44 GB
  Model Size (FP16): 0.22 GB
  Optimizer Memory: 0.88 GB
  Context Length: 1,024
  Vocabulary Size: 50,257

📁 DATASET INFORMATION:
  Train Shards: 3
  Validation Shards: 1
  Total Tokens: 1,250,000
  Train Tokens: 1,125,000
  Validation Tokens: 125,000

⚙️ TRAINING CONFIGURATION:
  Batch Size: 8
  Effective Batch Size: 16
  Total Steps: 5,000
  Learning Rate: 3.00e-04
  Mixed Precision: True

💾 MEMORY ANALYSIS:
  Peak VRAM Required: 2.45 GB

⏱️ PERFORMANCE ESTIMATION:
  Tokens/Second: 2,450
  Estimated Total Time: 3.2 hours (0.1 days)

💻 SYSTEM INFORMATION:
  GPU: NVIDIA RTX 4090
  GPU Memory: 24.0 GB total

✅ No warnings detected!
```


### 8. GPU Batch Size 


* Find optimal batch size automatically
```bash
python main.py --get-optimal-bs --text-file data.txt --model-size small
```
* Test specific batch sizes
```bash
python main.py --get-optimal-bs --test-batch-sizes 8 16 32 64 --text-file data.txt --model-size medium
```
* Test single batch size
```bash
python main.py --get-optimal-bs --target-batch-size 32 --text-file data.txt --model-size large
```
* Use custom safety margin (10% instead of default 15%)
```bash
python main.py --get-optimal-bs --gpu-safety-margin 0.10 --text-file data.txt --model-size small
```

---

## Generate

### Basic generation with default prompts
```bash
python main.py --generate --model-size small
```
### Generate from specific checkpoint
```bash
python main.py --generate --checkpoint-path "./training_output/milestones/checkpoint-5000/checkpoint.pt"
```
### Generate from prompts file
```bash
python main.py --generate --prompts-file prompts.txt --output-file results.txt
```
### Custom generation settings
```bash
python main.py --generate \
    --prompts "The future of AI" "Machine learning is" \
    --max-new-tokens 150 \
    --temperature 0.7 \
    --top-k 40 \
    --top-p 0.95 \
    --generation-batch-size 4 \
    --output-file generation_results.json
```
### Interactive generation mode
```bash
python main.py --generate --prompts "Hello world" --model-size small
```
### (Will automatically enter interactive mode for small bashprompt lists)
```bash
echo -e "The future of artificial intelligence is\nMachine learning will revolutionize\nDeep learning has enabled\nNatural language processing can\nComputer vision allows us to\nReinforcement learning helps agents\nNeural networks are capable of\nTransformers have changed how we" > prompts.txt
```


### 99. **Troubleshooting:**

**Common Issues:**
1. **CUDA OOM:** Reduce batch size or enable gradient checkpointing
2. **Slow tokenization:** Increase `--num-workers` for data preparation  
3. **No text file:** Provide `--text-file path/to/your/data.txt`
4. **Import errors:** Check if transformer_model.py exists and imports work

**Quick Test:**
```bash
# Minimal test with tiny model
python main.py --check --text-file sample_data.txt --model-size tiny --batch-size 2
```
