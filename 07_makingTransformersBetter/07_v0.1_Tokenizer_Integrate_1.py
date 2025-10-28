""" 
Current Model: 

optimizations DONE:
    0. [NONE] just copied into one file
    0.1 [Tokenizer] use new tokenizer and tokens 


optimizations TODO:
    1. parallelize the MHSA block (Effect: Time-- )
        Solutions 
        - do some kind of parallel programming : x not possible in GPU, increase time due to loading
        - [WORKS] Combine all the matrices of each W_i into single W, but preserve head wise (ie attention only in the head)
            This method is called GROUPED QUERY ATTENTION -> even combines QKV into single matrix (3*d_k^2 into one matrix)
    
    2 Embeddings
    
    2.1 Token Embeddings (Effect: Params-- + Time--  : reuse embedding at final layer)
        - decreases params 
    
    2.2 Positional Embeddings (Effect: Quality++ -> embedding highers ctx)
        - Sinusoidal like the Transformer paper
        - RoPE
        
    3. Better Normalization (Effect: Quality++ -> stabilize training )
        - LayerNorm -> placement (Pre vs. Post)
        - RMSNorm
        
    4. non linearity (Effect: Quality++)
        - GELU
        - Swish
        - GLU
        - SwiGLU (Swish + GLU ?)
        - Time++  :: Fused SwiGLU ( faster SwiGLU )
        
    6. Fully Sharded Data Parallelism (Effect: Time++ Space-- )
        (useful in multi-GPU multi-NODE)
        - FSDP 
    
    7. Attention Type (Effect: Quality++ )
        - Flash Attention-2 (fused layernorm, Rope, CELoss)
        - LogN scaling 
        - Windowed Attn
    
    8. Decoding Strategies:
    
    8.1 Topk
    8.2 topp
    8.3 beam search
    ...
    
    9. Fine-Tuning Modules 
    9.1 LoRA
    9.2 LoRA+
    
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import argparse 

logger_transformer = None
args = None

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, n_embeddings, head_size, context_length,  dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-SHSA] {self.variable_name = } , type({type(self.variable_name)})')
        self.n_embeddings = n_embeddings
        self.head_size = head_size
        self.context_length = context_length
        self.device = device
        
        self.attn_norm = self.head_size ** -0.5
        
        self.q = nn.Linear(self.n_embeddings, self.head_size , bias=False, device=self.device)
        self.k = nn.Linear(self.n_embeddings, self.head_size , bias=False, device=self.device)
        self.v = nn.Linear(self.n_embeddings, self.head_size , bias=False, device=self.device)

        self.register_buffer('tril', torch.tril(torch.ones(self.context_length,self.context_length, device=self.device)))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(B, T, self.head_size)
        K = K.view(B, T, self.head_size)
        V = V.view(B, T, self.head_size)

        attn = (Q @ K.transpose(-2,-1) ) * self.attn_norm
        masked_attn = attn.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        scores = F.softmax(masked_attn, dim=-1)

        scores = self.dropout_layer(scores)

        out = scores @ V
        out = out.view(B, T, self.head_size)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, context_length, n_embeddings, n_heads, dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-MHSA] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.context_length = context_length
        self.n_embeddings = n_embeddings
        self.n_heads = n_heads
        self.each_head_size = self.n_embeddings // self.n_heads
        self.dropout = dropout
        self.device = device
        
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(
                self.n_embeddings,
                self.each_head_size,
                self.context_length,
                self.dropout,
                self.device
            )  for _ in range(self.n_heads)
        ])
        
        self.proj_attn = nn.Linear(self.n_embeddings,self.n_embeddings, device=self.device)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self,x):
        ### not a parallelized code !!! need to fix ASAP !
        x = torch.cat(
            [
                shs_attn(x) for shs_attn in self.heads
            ], 
            dim = -1
        )
        return self.dropout_layer(self.proj_attn(x))


class FeedForwardNet(nn.Module):
    def __init__(self, n_embeddings, ffn_hid_dim = None, non_linearity = 'gelu', dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-FFN] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.n_embeddings = n_embeddings
        self.ffn_hid_dim = ffn_hid_dim if ffn_hid_dim is not None else 4 * n_embeddings
        self.dropout = dropout 
        self.device = device
        
        self.ln_ffn = nn.Linear(self.n_embeddings, self.ffn_hid_dim).to(self.device)
        # self.ffn_hid_dim = ffn_hid_dim if ffn_hid_dim is not None else int((8 * n_embeddings )//3)
        
        if non_linearity == 'leakyrelu':
            self.non_linearity = nn.LeakyReLU(0.01)
        elif non_linearity == 'silu':
            self.non_linearity = nn.SiLU()
        elif non_linearity == 'gelu':
            self.non_linearity = nn.GELU()
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        else:
            # print(f'[WARNING] Specified ({non_linearity}) is not supported, falling back to `ReLU` non-linearity')
            # logging.warning(f' Specified ({non_linearity}) is not supported, falling back to `ReLU` non-linearity')
            self.non_linearity = nn.ReLU()
        
        self.proj_ffn = nn.Linear(self.ffn_hid_dim, self.n_embeddings, device=self.device)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        x = self.non_linearity(self.ln_ffn(x))
        x = self.dropout_layer(self.non_linearity(self.proj_ffn(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, context_length, n_embeddings, n_heads, ffn_hid_dim = None,  non_linearity ='gelu', dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-DECODER] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.context_length = context_length
        self.n_embeddings = n_embeddings
        self.n_heads = n_heads
        self.ffn_hid_dim = ffn_hid_dim
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.device = device
        self.layer_norm1 = nn.LayerNorm(self.n_embeddings, device=self.device)
        self.attention = MultiHeadSelfAttention(self.context_length, self.n_embeddings, self.n_heads, self.dropout, self.device)
        self.layer_norm2 = nn.LayerNorm(self.n_embeddings, device=self.device)
        self.ffn = FeedForwardNet(self.n_embeddings, self.ffn_hid_dim, self.non_linearity, self.dropout, self.device)
        
        
    def forward(self, x, y=None):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size, context_length, n_embeddings, n_heads, Nx, ffn_hid_dim = None, non_linearity ='gelu', dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-Transformer] {self.variable_name = } , type({type(self.variable_name)})')

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_embeddings = n_embeddings
        self.n_heads = n_heads
        self.Nx = Nx
        self.ffn_hid_dim = ffn_hid_dim
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.device = device
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.n_embeddings, device=self.device)
        self.position_embeddings = nn.Embedding(self.context_length, self.n_embeddings, device=self.device)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    context_length=context_length,
                    ffn_hid_dim=ffn_hid_dim,
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    dropout=dropout,
                    device=device,
                    non_linearity=non_linearity
                ) for _ in range(self.Nx)
            ]
        )
        self.lm_head = nn.Linear(self.n_embeddings, self.vocab_size, device=self.device)

    def forward(self, x, y=None):
        # print(f'[INFO] | Transformer | [dim] {variable_name.shape = }')
        xB , xT = x.shape
        # print(f'[INFO] | Transformer | [dim] {x.shape = }')
        x_tokens = self.token_embeddings(x)
        # print(f'[INFO] | Transformer | [dim] {x_tokens.shape = }')
        x_positions = self.position_embeddings(torch.arange(xT).to(self.device))
        # print(f'[INFO] | Transformer | [dim] {x_positions.shape = }')
        embeds = x_tokens + x_positions
        # print(f'[INFO] | Transformer | [dim] {embeds.shape = }')

        for block in self.blocks:
            embeds = block(embeds)
        
        # print(f'[INFO] | Transformer | [dim] {embeds.shape = }')
        logits = self.lm_head(embeds)
        # print(f'[INFO] | Transformer | [dim] {logits.shape = }')

        if y is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        # print(f'[INFO] | Transformer | [dim] {loss.shape = }')
        return logits, loss


    def generate(self, x, n_pred):
        for _ in range(n_pred):
            logits, loss = self(x[:,-self.context_length:])
            logits = logits[:, -1, :]
            prob_dist = F.softmax(logits, -1)
            x = torch.cat([x, torch.multinomial(prob_dist, 1)], -1).to(self.device)
        return x













def get_args():
    parser = argparse.ArgumentParser(description="Train a character-level LM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (ignore saved models)")
    parser.add_argument("--check", action="store_true", help="Just checks model works and exits")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--context-length", type=int, default=16, help="Context length")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocab Size")
    parser.add_argument("--max-iters", type=int, default=10000, help="Maximum training iterations")
    parser.add_argument("--pred-char-len", type=int, default=100, help="Prediction length for generation")
    parser.add_argument("--Nx-blocks", type=int, default=4, help="No of Transformer Blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Heads in MHSA")
    parser.add_argument("--n-embeddings", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--ffn-hid-dim", type=int, default=None, help="Hidden dimension of FFN")
    parser.add_argument("--non-linearity", type=str, default='gelu', help="non linearity in FFN")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropouts")
    parser.add_argument("--models-folder", default="../saved_models", help="Folder for saving models")
    parser.add_argument("--data-file", default="../data/wikitext/processed.txt", help="Processed text file")
    # parser.add_argument("--chars-file", default="../data/wikitext/chars.txt", help="Characters file")
    parser.add_argument("--token-model-file", default="../data/wikitext/bpe1024.model", help="Characters file")
    # parser.add_argument("--encoding-file", default='../data/encodings/encoding.pth', help="Encoded data file")
    parser.add_argument("--encoding-file", default='../data/encodings/bpe1024_encoding.pth', help="Encoded data file")
    parser.add_argument("--model-file", type=str, default=None, help="Model checkpoint file")
    parser.add_argument("--model-name", type=str, default=f'Transformer_dev', help="Model Name")
    parser.add_argument("--csv-file", type=str, default=f'results.csv', help="csv")
    parser.add_argument("--generate", action="store_true", help="Toggle on to generate from model")
    parser.add_argument("--version", type=str, default=None, help="Version")
    parser.add_argument("--log", default='debug', help="set : debug, info, error, warning ")
    
    return parser.parse_args()

def main():
    args = get_args()
    import os
    import logging
    
    import math
    import numpy as np
    from tqdm import tqdm
    from datetime import datetime
    
    from nlp.tokenizer import CharTokenizer, myTokenizer_faster
    from nlp.utils import get_metrics, log_results_to_csv, convert_to_hr
    
    started_script = datetime.now()
    if args.log == 'info':
        log_level = logging.INFO
    elif args.log == 'debug':
        log_level = logging.DEBUG
    elif args.log == 'warning':
        log_level = logging.WARNING
    elif args.log == 'error':
        log_level = logging.ERROR
    else:
        log_level = logging.NOTSET
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    logger_transformer = logging.getLogger(__name__)
    
    # logging.error('Test info')
    # logging.warning('Test info')
    # logging.info('Test info')
    # logging.debug('Test info')
    if args.check:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger_transformer.info(f"{DEVICE = }")
        
        
        options = dict(
            vocab_size =args.vocab_size,
            context_length = args.context_length,
            n_embeddings =args.n_embeddings,
            n_heads =args.n_heads,
            Nx =args.Nx_blocks,
            ffn_hid_dim =args.ffn_hid_dim,
            non_linearity =args.non_linearity,
            dropout =args.dropout,
            device =args.device,
        )

        gpt = DecoderOnlyTransformer(**options)
        
        batch_size = args.batch_size
        time_dim = args.context_length
        
        data_X = np.random.randint(0, 50257, (batch_size, time_dim))
        data_y = np.random.randint(0, 50257, (batch_size * time_dim))
        random_X = torch.tensor(data_X, dtype=torch.long, device=DEVICE)
        random_y = torch.tensor(data_y, dtype=torch.long, device=DEVICE)
        
        logger_transformer.info(f"Random inputs: {random_X.shape = } {random_y.shape = }")
        
        # logits = gpt(random_X)
        # logger_transformer.info(f"{logits.shape = }")
        
        logits, loss = gpt(random_X, random_y)
        
        logger_transformer.info(f"✅ Forward Pass Successful\n\tOutputs: {logits.shape = }\n\tModel Loss : {loss.item():.4f}\n\tExpected Loss: {-np.log(1/args.vocab_size):.4f}")
        
        logger_transformer.info(f"Trying to overfit on a single Batch")
        
        optimizer = torch.optim.AdamW(
            gpt.parameters(),
            lr=1e-3,  # Learning rate
            betas=(0.9, 0.999),  # Beta parameters for momentum
            eps=1e-8,  # Small constant for numerical stability
            weight_decay=1e-2,  # L2 regularization
            amsgrad=False  # Whether to use AMSGrad variant
        )
        
        num_epochs = 100
        target_loss = 0.05
        log_interval = 5
        
        logger_transformer.info(f"Starting overfitting training for {num_epochs} epochs")
        logger_transformer.info(f"Target loss: {target_loss:.4f}")
        
        # Training loop for overfitting on single batch
        gpt.train()  # Set model to training mode
        
        for epoch in range(num_epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            logits, loss = gpt(random_X, random_y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                logger_transformer.info(f"Epoch [{epoch+1:4d}/{num_epochs}] | Loss: {loss.item():.6f}")
            
            # Early stopping if target loss is reached
            if loss.item() < target_loss:
                logger_transformer.info(f"🎯 Target loss reached at epoch {epoch+1}")
                logger_transformer.info(f"Final loss: {loss.item():.6f}")
                break
        
        gpt.eval() 
        with torch.no_grad():
            final_logits, final_loss = gpt(random_X, random_y)
        
        logger_transformer.info(f"✅ Overfitting Complete!")
        logger_transformer.info(f"Final training loss: {final_loss.item():.6f}")
        logger_transformer.info(f"Loss reduction: {(loss.item() - final_loss.item()):.6f}")
        with torch.no_grad():
            predictions = torch.argmax(final_logits, dim=-1)
            targets_reshaped = random_y.view(batch_size, time_dim)
            accuracy = (predictions == targets_reshaped).float().mean()
            logger_transformer.info(f"Training accuracy: {accuracy.item()*100:.2f}%")
        
        return 
    
    model_name = args.model_name
    DEVICE = args.device
    CONTEXT_LENGTH = args.context_length
    BATCH_SIZE = args.batch_size

    ALPHA = args.alpha
    MAX_ITERS = args.max_iters
    EVAL_ITERS = MAX_ITERS // 100
    MAX_CONTEXT_LEN = args.pred_char_len


    if args.version:
        model_name += '_' + args.version


    logger_transformer.info(f"Device : {DEVICE}")

    os.makedirs(args.models_folder, exist_ok=True)
    if args.model_file is not None:
        MODEL_FILE = args.model_file
    else :
        MODEL_FILE = f'{args.models_folder}/{model_name}_{DEVICE}.pth'

    model_exists = os.path.exists(MODEL_FILE)

    data_exists = os.path.exists(args.data_file)
    token_file_exists = os.path.exists(args.token_model_file)

    if not data_exists:
        logger_transformer.info("⚠️ Data not found!")
        exit(0)

    tokenizer = myTokenizer_faster(
        token_model_file=args.token_model_file,
        special_tokens={
            '<|startofseq|>':args.vocab_size - 4,
            '<|endofseq|>':args.vocab_size - 3,
            '<|padding|>':args.vocab_size - 2,
            '<|endoftext|>':args.vocab_size - 1
        }
    )
    if not token_file_exists:
        logger_transformer.info("🔷 Loading Data & Training tokenizer...")
        vocab = tokenizer.train(
            args.data_file,
            args.vocab_size
            )
        logger_transformer.info("✅ Data Loaded and chars saved!")
    else:
        vocab = tokenizer.load()

    if os.path.exists(args.encoding_file):
        encoded_data = torch.load(args.encoding_file, map_location=DEVICE)
        logger_transformer.info("✅ Encoding Loaded!")
    else:
        logger_transformer.info("Encoding not found! Loading Data...")
        data = open(args.data_file, "r").read()
        logger_transformer.info("✅ Data Loaded!")
        logger_transformer.info("🔷 Encoding Data...")
        st = datetime.now()
        encoded_data = torch.tensor(
            tokenizer.encode(data[:10000000], progress=True),
            dtype=torch.long, 
            device=DEVICE
        )
        logger_transformer.info(f"✅ Data Encoded in {datetime.now() - st}!")
        torch.save(encoded_data.cpu(), args.encoding_file)
        logger_transformer.info("✅ Encoding saved and Loaded!")

    VOCAB_SIZE = tokenizer._vocab_size()
    logger_transformer.info(f"Number of Tokens (vocab size) : {VOCAB_SIZE}")

    n = len(encoded_data)
    n1, n2 = int(0.8 * n), int(0.9 * n)
    Xtr, Xval, Xte = encoded_data[:n1], encoded_data[n1:n2], encoded_data[n2:]

    logger_transformer.info(f"Dataset split into train:{len(Xtr)} tkns, val:{len(Xval)} tkns, test:{len(Xte)} tkns")

    Xtr.to(DEVICE)
    Xval.to(DEVICE)
    Xte.to(DEVICE)

    def get_batch(split):
        if split == 'train':
            X = Xtr
        elif split == 'val':
            X = Xval
        else:
            X = Xte
        start = np.random.randint(0, len(X) - CONTEXT_LENGTH - 1, (BATCH_SIZE,))
        X_batch = torch.stack([X[s:s + CONTEXT_LENGTH] for s in start]).to(DEVICE)
        y_batch = torch.stack([X[s+1:s + CONTEXT_LENGTH+1] for s in start]).to(DEVICE)
        return X_batch.to(DEVICE), y_batch.to(DEVICE)


    @torch.no_grad()
    def estimate_batch_loss(model):
        X_batch, y_batch = get_batch('dev')
        logits, loss = model(X_batch, y_batch)
        return loss.item()


    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val', 'test']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # bi-gram model
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE, 
        context_length=CONTEXT_LENGTH,
        n_embeddings=args.n_embeddings,
        n_heads = args.n_heads,
        Nx=args.Nx_blocks,
        ffn_hid_dim = args.ffn_hid_dim,
        non_linearity=args.non_linearity,
        dropout = args.dropout,
        device=DEVICE
        )
    model = model.to(DEVICE)
    logger_transformer.info('Model Created\n')
    logger_transformer.info('-'*50)
    model_parameters = sum(p.numel() for p in model.parameters())
    logger_transformer.info(f"Total parameters:{model_parameters:,} \t({convert_to_hr(model_parameters)})") 
    logger_transformer.info('-'*50 + '\n')
    
    train_time = None
    if args.train:
        logger_transformer.info(f'🔷 Training ...')
        if not model_exists or args.scratch:
            logger_transformer.info(f'⚠️ Training from scratch ...')
        else:
            if model_exists:
                model.load_state_dict(torch.load(MODEL_FILE))
                logger_transformer.info('✅ Model loaded! Resuming training ...')

        st = datetime.now()
        optimiser = optim.AdamW(model.parameters(), lr=ALPHA)
        
        # Initialize progress bar with custom format
        pbar = tqdm(range(MAX_ITERS), desc='Training', unit='iter')
        
        # Variables for tracking metrics
        eval_interval = MAX_ITERS // 10
        total_tokens_processed = 0
        val_perplexity = float('-inf')
        val_loss = float('-inf')
        for iter in pbar:
            iter_start_time = datetime.now()
            
            # Training step
            x, y = get_batch('train')
            logits, loss = model(x, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            
            # Calculate tokens processed this iteration
            iter_tokens = y.numel()  # Number of tokens in current batch
            total_tokens_processed += iter_tokens
            
            # Calculate iteration time and tokens per second
            iter_end_time = datetime.now()
            iter_duration = (iter_end_time - iter_start_time).total_seconds()
            current_tps = iter_tokens / iter_duration if iter_duration > 0 else 0
            
            # Calculate metrics at evaluation intervals
            if iter % eval_interval == 0 or iter == 0:
                model.eval()
                with torch.no_grad():
                    # Quick evaluation for train loss
                    # train_x, train_y = get_batch('train')
                    # train_logits, train_loss = model(train_x, train_y)
                    
                    # Quick evaluation for validation loss and perplexity
                    val_x, val_y = get_batch('val')
                    val_logits, val_loss = model(val_x, val_y)
                    val_perplexity = math.exp(val_loss.item())
                model.train()
                
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                # 'train_loss': f'{train_loss.item():.4f}',
                'val_loss': f'{val_loss.item():.4f}',
                'val_ppl': f'{val_perplexity:.2f}',
                'tps': f'{current_tps:.0f}'
            })
                
        pbar.close()
        
        et = datetime.now()
        train_time = et - st
        
        # Calculate overall training statistics
        total_duration = train_time.total_seconds()
        overall_tps = total_tokens_processed / total_duration if total_duration > 0 else 0
        
        logger_transformer.info(f'✅ Training Done in {train_time}!')
        logger_transformer.info(f'📊 Training Statistics:')
        logger_transformer.info(f'   Total Tokens Processed: {total_tokens_processed:,}')
        logger_transformer.info(f'   Overall Tokens/Second: {overall_tps:.2f}')
        
        torch.save(model.state_dict(), MODEL_FILE)
        logger_transformer.info('✅ Model saved!')
        
        # Final comprehensive evaluation using your existing function
        logger_transformer.info(f'\n[>] Final Evaluation ...')
        out = get_metrics(model,
                        {
                            'train': Xtr,
                            'val': Xval,
                            'test': Xte,
                        },
                        CONTEXT_LENGTH,
                        BATCH_SIZE,
                        EVAL_ITERS,
                        DEVICE
                        )
        for split, metrics in out.items():
            logger_transformer.info(f"\t{split:5} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"PPL: {metrics['perplexity']:.2f} | "
                f"Acc: {metrics['accuracy']*100:.2f}% | "
                f"BPC: {metrics['bpc']:.4f}")


    if args.generate:
        logger_transformer.info(f'🔷  Generating ...')

        logger_transformer.info('Infernece:\n' +'-'*100 + f'\n\tExample 1 : Starting with no context' + '-'*100 +'\nGenerated: ' + tokenizer.decode(
            model.generate(torch.zeros((1, 1), dtype=torch.long,device=DEVICE), n_pred=MAX_CONTEXT_LEN)[0].tolist()) + '\n'
        )

        example_prompts = [
            "Padma Vibhush",
            "Virat Koh",
            "Virat Koh",
            "Virat Koh",
            "Virat Koh",
            "Virat Koh",
            "As with previous Valkyira Chronic",
        ]
        for it , starts_with in enumerate(example_prompts):
            contxt = torch.tensor( 
                tokenizer.encode(starts_with), 
                dtype=torch.long).unsqueeze(0).to(DEVICE)
            inference = tokenizer.decode(
                model.generate(contxt, n_pred=MAX_CONTEXT_LEN)[0].tolist()
                )
            logger_transformer.info('Infernece:\n' +'-'*100 + f'\n\tExample {it + 2} : Starting with {repr(starts_with)}\n' + '-'*100 + f'\nGenerated:{inference}\n' )
        
    logger_transformer.info(f'🔷 Writing results to csv {args.csv_file} !')
    log_results_to_csv(args.csv_file, model_name, model_parameters, train_time, args, out)
    logger_transformer.info('✅ Done !')
    
    logger_transformer.info(f'⌚ Script Time: {datetime.now() - started_script}')



if __name__ == '__main__':
    main()