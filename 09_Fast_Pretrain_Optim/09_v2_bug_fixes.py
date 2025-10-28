""" 
Current Model: 

optimizations DONE:
    0. [NONE] just copied into one file
    0.1 [Tokenizer] use new tokenizer and tokens 

    1. parallelize the MHSA block (Effect: Time-- )
        Solutions 
        - do some kind of parallel programming : x not possible in GPU, increase time due to loading
        - [WORKS] Combine all the matrices of each W_i into single W, but preserve head wise (ie attention only in the head)
            This method is called GROUPED QUERY ATTENTION -> even combines QKV into single matrix (3*d_k^2 into one matrix)
    
    2 Embeddings
    
    2.1.1 Tying Embeddings (Effect: Params-- + Time--  : reuse embedding at final layer)
        - decreases params 
    
    Repercussion :: Exploded loss (Less capacity, Coupled optimization problem)

    Solution ::  initialize embeddings carefully (e.g. GPT2 std=0.02) 
                 apply a scaling factor in the forward pass (e.g. GPT2 div by sqrt(model_dimension)) 
                    
                    - I will do residual normalization = 2 residues per block
                    = 1/sqrt ( Nx * 2 )
    
    2.1.2 Token Embeddings (Effect: Params++ + Time-- )
        - changing tokenizer to the 100352 vocab extended cl100k_base tiktoken model
        

    5. Hidden Dimensions (Effect: Time-- Params-- Quality?)
        - 4 x , 3/8 x, 2 x,   which is better ? 
        - 1x, 2x, 2.5x, 3x, 3.5x, 4x, 4.5x, 6x, 8x
    
    Result : seems unclear may be 3.5 (?)

    3. Better Normalization (Effect: Quality++ -> stabilize training )
        - LayerNorm -> placement (Pre vs. Post)
        - RMSNorm

    Result : prefering RMS Norm

        
    4. non linearity (Effect: Quality++)
        - GELU
        - Swish
        - GLU
        - SwiGLU (Swish + GLU ?)
        - Time++  :: Fused SwiGLU ( faster SwiGLU )
        
        Winner : SwiGLU
    
    6. Fully Sharded Data Parallelism (Effect: Time++ Space-- )
        (useful in multi-GPU multi-NODE)
        - FSDP 
    Reason : No multi gpus
    
        
    7. Attention Type (Effect: Quality++ )
        - Flash Attention-1/2   :: scaled_dot_product_attention()
            - https://pytorch.org/blog/pytorch2-2/
        - Flash Attention-3     :: No current implementations for AMD :(
    
---------------------------------           ONGOING            ---------------------------------
    
    OPTIMIZATIONS :
        - BUG fix : Dropout in inference
        - Cleaned up FFN code -> swiglu (Faster version) or GeLU , fall back = ReLU
        - implemented - RMS Norm 
        - BUG Fix : introduced final RMS Norm Layer
    8. Decoding Strategies:
        8.1 Topk
        8.3 temp
---------------------------------             TODO            ---------------------------------

optimizations TODO:

    2 Embeddings
    
    2.2 Positional Embeddings (Effect: Quality++ -> embedding highers ctx)
        - Sinusoidal like the Transformer paper
        - RoPE
        
===> Inference::    
    8. Decoding Strategies:
    
        8.2 topp
        8.4 beam search
        8.5 spculative
    ...
    
    9. Fine-Tuning Modules 
        9.1 LoRA
        9.2 LoRA+
        9.2 QLoRA
    
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

import argparse 

# logger_transformer = None
args = None


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, context_length, model_dimension, n_heads, dropout = 0.3, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-MHSA] {self.variable_name.shape = } \ttype({type(self.variable_name)})')
        
        self.context_length = context_length
        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.each_head_size = self.model_dimension // self.n_heads
        self.dropout = dropout
        self.device = device
        self.W_qkv = nn.Linear(self.model_dimension, 3 * self.model_dimension , bias=False, device=self.device)
        self.proj_attn = nn.Linear(self.model_dimension,self.model_dimension, device=self.device)
        self._init_weights()
    
    def forward(self,x):
        B,T,C = x.shape

        Q,K,V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q,K,V,attn_mask = None, dropout_p=self.dropout, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj_attn(out)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.xavier_uniform_(self.proj_attn.weight)

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc_a = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_b = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        a = self.fc_a(x) 
        b = self.fc_b(x)
        return F.silu(a) * b

class SwiGLUFast(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2 * dim_out)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return F.silu(a) * b

class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ffn_hid_dim = None, non_linearity = 'swiglu', dropout = 0.3, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-FFN] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.model_dimension = model_dimension
        ffn_factor = 8/3 # 4 is Original , 8/3 is PALM
        self.ffn_hid_dim = ffn_hid_dim if ffn_hid_dim is not None else int(ffn_factor * model_dimension)
        self.non_linearity = non_linearity
        self.dropout = dropout 
        self.device = device
        
        if non_linearity == 'swiglu'or non_linearity == 'swiglufast' :
            self.proj_ffn = nn.Linear(self.ffn_hid_dim, self.model_dimension, device=self.device)
            self.activation = SwiGLU(self.model_dimension, self.ffn_hid_dim).to(self.device) if non_linearity == 'swiglu' else SwiGLUFast(self.model_dimension, self.ffn_hid_dim).to(self.device)
        else :
            self.ln_ffn = nn.Linear(model_dimension, self.ffn_hid_dim, bias=False)
            self.proj_ffn = nn.Linear(self.ffn_hid_dim, model_dimension, bias=False)
            self.activation = nn.GELU() if non_linearity == 'gelu' else nn.ReLU()
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        if  self.non_linearity.startswith('swiglu'):
            x = self.activation(x)
        else:
            x = self.activation(self.ln_ffn(x))
        return self.dropout_layer(self.proj_ffn(x))

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, device = 'cpu', eps = 1e-6):
        super().__init__()
        self.device = device
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(self.device)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

class DecoderBlock(nn.Module):
    def __init__(self, context_length, model_dimension, n_heads, ffn_hid_dim = None,  non_linearity ='gelu', dropout = 0.3, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-DECODER] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.context_length = context_length
        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.ffn_hid_dim = ffn_hid_dim
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.device = device
        self.layer_norm1 = RMSNorm(self.model_dimension, device=self.device)
        self.attention = MultiHeadSelfAttention(self.context_length, self.model_dimension, self.n_heads, self.dropout, self.device)
        self.layer_norm2 = RMSNorm(self.model_dimension, device=self.device)
        self.ffn = FeedForwardNet(self.model_dimension, self.ffn_hid_dim, self.non_linearity, self.dropout, self.device)
        
        
    def forward(self, x, y=None):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size, context_length, model_dimension, n_heads, Nx, ffn_hid_dim = None, training = False, tie_weights = True, non_linearity ='gelu', dropout = 0.3, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-Transformer] {self.variable_name = } , type({type(self.variable_name)})')

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.Nx = Nx
        self.training = training
        self.ffn_hid_dim = ffn_hid_dim
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.device = device
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.model_dimension,device=self.device)
        self.position_embeddings = nn.Embedding(self.context_length, self.model_dimension, device=self.device)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    context_length=context_length,
                    ffn_hid_dim=ffn_hid_dim,
                    n_heads=n_heads,
                    model_dimension=model_dimension,
                    dropout=self.dropout if self.training else 0.0,
                    device=device,
                    non_linearity=non_linearity
                ) for _ in range(self.Nx)
            ]
        )
        self.layer_normfinal = RMSNorm(self.model_dimension, device=self.device)
        self.lm_head = nn.Linear(self.model_dimension, self.vocab_size, bias= False,device=self.device)
        
        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight
                
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * ((2 * self.Nx) ** -0.5))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x, y=None):
        xB , xT = x.shape
        x_tokens = self.token_embeddings(x)
        x_positions = self.position_embeddings(torch.arange(xT,device = self.device))
        embeds = x_tokens + x_positions

        for block in self.blocks:
            embeds = block(embeds)
        
        embeds = self.layer_normfinal(embeds)
        
        logits = self.lm_head(embeds)

        if y is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_pred_tokens,temp = 1.0, top_k = None ):
        self.eval()
        for _ in range(max_pred_tokens):
            logits, loss = self(x[:,-self.context_length:])
            logits = logits[:, -1, :] / temp
            if top_k is not None:
                top_k_vals, top_k_idxs = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = -float('Inf')
            prob_dist = F.softmax(logits, -1)
            x = torch.cat([x, torch.multinomial(prob_dist, 1)], -1).to(self.device)
        self.train()
        return x













def get_args():
    parser = argparse.ArgumentParser(description="Train a character-level LM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (ignore saved models)")
    parser.add_argument("--check", action="store_true", help="Just checks model works and exits")
    parser.add_argument("--tie-weights", action="store_true", help="Tie Weights of embed and last layer?")
    parser.add_argument("--alpha", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--context-length", type=int, default=128, help="Context length")
    parser.add_argument("--vocab-size", type=int, default=100352, help="Vocab Size")
    parser.add_argument("--max-iters", type=int, default=None, help="Maximum training iterations")
    parser.add_argument("--Nx-blocks", type=int, default=4, help="No of Transformer Blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Heads in MHSA")
    parser.add_argument("--model-dimension", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--ffn-hid-dim", type=int, default=None, help="Hidden dimension of FFN")
    parser.add_argument("--non-linearity", type=str, default='gelu', help="non linearity in FFN")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropouts")
    parser.add_argument("--models-folder", default="../saved_models", help="Folder for saving models")
    parser.add_argument("--data-file", default="../data/wikitext/processed.txt", help="Processed text file")
    parser.add_argument("--token-model-file", default="../data/wikitext/cl100k_maha.model", help="Model file")
    parser.add_argument("--encoding-file", default='/mnt/volume5tb/data/english/WanJuan_cc/OpenDataLab___WanJuanCC/raw/old_shards/part-659f9e2120e3-000622/shard_00000.pt', help="Encoded data file")
    parser.add_argument("--model-file", type=str, default=None, help="Model checkpoint file")
    parser.add_argument("--model-name", type=str, default=f'Transformer_dev', help="Model Name")
    parser.add_argument("--description", type=str, default=f'no optimizations applied', help="Model Description")
    parser.add_argument("--version", type=str, default=None, help="Version")
    parser.add_argument("--csv-file", type=str, default=f'results.csv', help="csv to push results to")
    parser.add_argument("--generate", action="store_true", help="Toggle on to generate from model")
    parser.add_argument("--pred-char-len", type=int, default=100, help="Prediction length for generation")
    parser.add_argument("--log-level", default='debug', help="set : debug, info, error, warning ")
    
    return parser.parse_args()

def main():
    args = get_args()
    import os
    import logging
    
    import math
    import numpy as np
    from tqdm import tqdm
    from datetime import datetime
    
    import tiktoken
    from nlp.utils import get_metrics, log_results_to_csv, convert_to_hr

    MAHAFILESEP = "<|maha_sep|>"
    ENDOFTEXT = "<|endoftext|>"
    ENDOFTEXT_IDX = 100257
    MAHAFILESEP_IDX = 100264

    def get_encoder():
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.Encoding(
            name="cl100k_myt",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens, 
                MAHAFILESEP: 100264
            }
        )
        return enc
    
    started_script = datetime.now()
    if args.log_level == 'info':
        log_level = logging.INFO
    elif args.log_level == 'debug':
        log_level = logging.DEBUG
    elif args.log_level == 'warning':
        log_level = logging.WARNING
    elif args.log_level == 'error':
        log_level = logging.ERROR
    else:
        log_level = logging.NOTSET
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    logger_transformer = logging.getLogger(__name__)
    
    if args.check:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger_transformer.info(f"{DEVICE = }")
        
        
        options = dict(
            vocab_size =args.vocab_size,
            context_length = args.context_length,
            model_dimension =args.model_dimension,
            n_heads =args.n_heads,
            Nx =args.Nx_blocks,
            training = True,
            tie_weights = args.tie_weights,
            ffn_hid_dim =args.ffn_hid_dim,
            non_linearity =args.non_linearity,
            dropout =args.dropout,
            device =args.device,
        )

        gpt = DecoderOnlyTransformer(**options)
        model_parameters = sum(p.numel() for p in gpt.parameters())
        logger_transformer.info( '\n' + '-' * 100 + '\n' + \
                                f"\t\tTotal parameters:{model_parameters:,} \t({convert_to_hr(model_parameters)})" + \
                                '\n' + '-' * 100 + '\n'
                                )
        batch_size = args.batch_size
        time_dim = args.context_length
        
        data_X = np.random.randint(0, args.vocab_size, (batch_size, time_dim))
        data_y = np.random.randint(0, args.vocab_size, (batch_size * time_dim))
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
        
        num_epochs = 1000
        target_loss = 0.1
        log_interval = num_epochs // 25
        
        logger_transformer.info(f"Starting overfitting training for {num_epochs} epochs")
        logger_transformer.info(f"Target loss: {target_loss:.4f}")
        
        gpt.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            logits, loss = gpt(random_X, random_y)
            loss.backward()
            
            optimizer.step()
            
            if (epoch + 1) % log_interval == 0:
                logger_transformer.info(f"Epoch [{epoch+1:4d}/{num_epochs}] | Loss: {loss.item():.6f}")
            
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
        
        logger_transformer.info(f'⌚ Script Time: {datetime.now() - started_script}')
        
        return 
    
    model_name = args.model_name
    DEVICE = args.device
    CONTEXT_LENGTH = args.context_length
    BATCH_SIZE = args.batch_size

    ALPHA = args.alpha
    
    MAX_CONTEXT_LEN = args.pred_char_len

    torch.set_float32_matmul_precision('high')

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
        logger_transformer.error("⚠️ Data not found!")
        exit(0)

    # tokenizer = myTokenizer_faster(
    #     token_model_file=args.token_model_file,
    #     special_tokens={
    #         '<|startofseq|>':args.vocab_size - 4,
    #         '<|endofseq|>':args.vocab_size - 3,
    #         '<|padding|>':args.vocab_size - 2,
    #         '<|endoftext|>':args.vocab_size - 1
    #     }
    # )
    # if not token_file_exists:
    #     logger_transformer.warning("⚠️ Tokenizer not found  (training the tokenizer using BPE)")
    #     logger_transformer.info("🔹 Loading Data & Training tokenizer...")
    #     vocab = tokenizer.train(
    #         args.data_file,
    #         args.vocab_size
    #         )
    #     logger_transformer.info("✅ Data Loaded and tokenization encodings saved!")
    # else:
    #     vocab = tokenizer.load()
    
    tokenizer = get_encoder()
    
    if os.path.exists(args.encoding_file):
        encoded_data = torch.load(args.encoding_file, map_location=DEVICE)
        logger_transformer.info("✅ Encoding Loaded!")
    else:
        logger_transformer.info("Encoding not found! Loading Data...")
        data = open(args.data_file, "r").read()
        logger_transformer.info(f"✅ Data Loaded! [{len(data)} characters]")
        logger_transformer.info("🔹 Encoding Data...")
        st = datetime.now()
        encoded_data = torch.tensor(
            tokenizer.encode(data,
                             allowed_special={
                                 MAHAFILESEP, 
                                 ENDOFTEXT
                                }
                            ),
            dtype=torch.long, 
            device=DEVICE
        )
        logger_transformer.info(f"✅ Data Encoded in {datetime.now() - st}!")
        torch.save(encoded_data.cpu(), args.encoding_file)
        logger_transformer.info("✅ Encoding saved and Loaded!")

    VOCAB_SIZE = tokenizer.n_vocab
    logger_transformer.info(f"Tokenizer Size (vocab size) : {VOCAB_SIZE}")
    logger_transformer.info(f"Actual Size (padded vocab size) : {args.vocab_size}")
    n = len(encoded_data)
    train_end = 0.94
    val_end = train_end + ( (1-train_end)/2)
    n1, n2 = int(train_end * n), int(val_end * n)
    Xtr, Xval, Xte = encoded_data[:n1], encoded_data[n1:n2], encoded_data[n2:]

    logger_transformer.info(f"Dataset split into train:{len(Xtr)} tkns, val:{len(Xval)} tkns, test:{len(Xte)} tkns")

    # Xtr.to(DEVICE)
    # Xval.to(DEVICE)
    # Xte.to(DEVICE)

    Xtr_pointer, Xval_pointer, Xte_pointer = 0, 0, 0

    def get_batch(split, Xtr_pointer = Xtr_pointer, Xval_pointer =  Xval_pointer, Xte_pointer =  Xte_pointer):
        if split == 'train':
            X = Xtr
            ptr = Xtr_pointer
        elif split == 'val':
            X = Xval
            ptr = Xval_pointer
        else:
            X = Xte
            ptr = Xte_pointer
        
        X_batch = X[ptr : ptr + BATCH_SIZE * CONTEXT_LENGTH]

        if X_batch.numel() < BATCH_SIZE * CONTEXT_LENGTH:
            pad_len = BATCH_SIZE * CONTEXT_LENGTH - X_batch.numel()
            pad = torch.full((pad_len,), ENDOFTEXT_IDX, dtype=X.dtype, device=DEVICE)
            X_batch = torch.cat([X_batch, pad], dim=0)

        X_batch = X_batch.reshape(BATCH_SIZE, CONTEXT_LENGTH).to(DEVICE)

        y_batch = X[ptr + 1 : ptr + BATCH_SIZE * CONTEXT_LENGTH + 1]

        if y_batch.numel() < BATCH_SIZE * CONTEXT_LENGTH:
            pad_len = BATCH_SIZE * CONTEXT_LENGTH - y_batch.numel()
            pad = torch.full((pad_len,), ENDOFTEXT_IDX, dtype=y_batch.dtype , device=DEVICE)
            y_batch = torch.cat([y_batch, pad], dim=0)

        y_batch = y_batch.reshape(BATCH_SIZE, CONTEXT_LENGTH).to(DEVICE)
        
        if split == 'train':
            Xtr_pointer += BATCH_SIZE * CONTEXT_LENGTH
        elif split == 'val':
            Xval_pointer += BATCH_SIZE * CONTEXT_LENGTH
        else:
            Xte_pointer += BATCH_SIZE * CONTEXT_LENGTH

        return X_batch, y_batch,Xtr_pointer, Xval_pointer, Xte_pointer


    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE, 
        context_length=CONTEXT_LENGTH,
        model_dimension=args.model_dimension,
        n_heads = args.n_heads,
        Nx=args.Nx_blocks,
        training=args.train,
        tie_weights=args.tie_weights,
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
    
    logger_transformer.info('Compiling...')
    model = torch.compile(model, mode="max-autotune", dynamic=True)
    logger_transformer.info('✅ Compiling Done')
    
    data_len = len(encoded_data)
    chinchila_rule = 20
    logger_transformer.info(f"Rule : To train we need {chinchila_rule} tkns/parm -> Required tokens in dataset = {model_parameters * chinchila_rule:,} \t({convert_to_hr(model_parameters * chinchila_rule)})") 
    logger_transformer.info(f"Dataset have {data_len} ({convert_to_hr(data_len)}) tokens out of which train tokens are {int(data_len*train_end)} \t({convert_to_hr(int(data_len*train_end))})") 
    delta = model_parameters * chinchila_rule - int(data_len*train_end)
    if delta > 0:
        logger_transformer.warning(f"❓ Needed more tokens ! ( short off : {delta }\t[{convert_to_hr(delta)}] )") 
    if delta < 0:
        logger_transformer.info(f"🤞 Good enough data ! ( surplus data : {-delta }\t[{convert_to_hr(-delta)}] )") 
        logger_transformer.info(f"Cutting down dataset to {chinchila_rule} tokens/param for faster experiments") 
        logger_transformer.info(f">>> Comment this code section to use all data <<<") 
        new_xtr_len = model_parameters * (chinchila_rule)
        Xtr = Xtr[:new_xtr_len]
        logger_transformer.info(f"🤞 Current data = {len(Xtr) }\t[{convert_to_hr(len(Xtr))}] )") 

    if args.max_iters is not None:
        MAX_ITERS = args.max_iters
    else:
        MAX_ITERS = (len(Xtr) - CONTEXT_LENGTH)// BATCH_SIZE
    EVAL_ITERS = MAX_ITERS // 1000 
    if EVAL_ITERS == 0:
        EVAL_ITERS = MAX_ITERS

    train_time = None
    if args.train:
        logger_transformer.info(f'🔹 Training ...')
        if not model_exists or args.scratch:
            logger_transformer.info(f'⚠️ Training from scratch ...')
        else:
            if model_exists:
                model.load_state_dict(torch.load(MODEL_FILE))
                logger_transformer.info('✅ Model loaded! Resuming training ...')

        st = datetime.now()
        optimiser = optim.AdamW(model.parameters(), lr=ALPHA)
        
        pbar = tqdm(range(MAX_ITERS), desc='Training', unit='iter')
        total_tokens_processed = 0
        val_perplexity = float('inf')
        val_loss = float('-inf')
        for iter in pbar:
            iter_start_time = datetime.now()
            
            x, y,Xtr_pointer, Xval_pointer, Xte_pointer = get_batch('train',Xtr_pointer, Xval_pointer, Xte_pointer)
            logits, loss = model(x, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            
            iter_tokens = y.numel()
            total_tokens_processed += iter_tokens
            
            iter_end_time = datetime.now()
            iter_duration = (iter_end_time - iter_start_time).total_seconds()
            current_tps = iter_tokens / iter_duration if iter_duration > 0 else 0
            
            if iter % EVAL_ITERS == 0 or iter == 0:
                model.eval()
                with torch.no_grad():
                    val_x, val_y,Xtr_pointer, Xval_pointer, Xte_pointer = get_batch('val',Xtr_pointer, Xval_pointer, Xte_pointer)
                    val_logits, val_loss = model(val_x, val_y)
                    if val_loss.item() < 100:
                        val_perplexity = math.exp(val_loss.item())
                    else:
                        val_perplexity = float('inf')
                        
                model.train()
                
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'val_loss': f'{val_loss.item() if val_loss.item() < 1e8 else float("inf") :.4f}',
                'val_ppl': f'{val_perplexity if val_perplexity < 1e3 else float("inf") :.2f}',
                'tps': f'{current_tps:.0f} tkn/s'
            })
                
        pbar.close()
        
        et = datetime.now()
        train_time = et - st
        
        total_duration = train_time.total_seconds()
        overall_tps = total_tokens_processed / total_duration if total_duration > 0 else 0
        
        logger_transformer.info(f'✅ Training Done in {train_time}!')
        logger_transformer.info(f'📊 Training Statistics:')
        logger_transformer.info(f'   Total Tokens Processed: {total_tokens_processed:,}')
        logger_transformer.info(f'   Overall Tokens/Second: {overall_tps:.2f}')
        
        torch.save(model.state_dict(), MODEL_FILE)
        logger_transformer.info('✅ Model saved!')
        
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

        logger_transformer.info(f'🔹 Writing results to csv {args.csv_file} !')
        log_results_to_csv(args.csv_file, model_name, model_parameters, train_time, args, out)

    if args.generate:
        logger_transformer.info(f'🔹  Generating ...')

        logger_transformer.info('Infernece:\n' +'-'*100 + f'\n\tExample 1 : Starting with no context\n' + '-'*100 +'\nGenerated: ' + tokenizer.decode(
            model.generate(torch.zeros((1, 1), dtype=torch.long,device=DEVICE), max_pred_tokens=MAX_CONTEXT_LEN)[0].tolist()) + '\n'
        )

        example_prompts = [
            "Padma Vibhush",
            "Virat Koh",
            # "Virat Koh",
            # "Virat Koh",
            # "Virat Koh",
            # "Virat Koh",
            "As with previous Valkyira Chronic",
        ]
        for it , starts_with in enumerate(example_prompts):
            contxt = torch.tensor( 
                tokenizer.encode(starts_with,allowed_special={
                                    MAHAFILESEP, 
                                    ENDOFTEXT
                                    }
                                 ), 
                dtype=torch.long).unsqueeze(0).to(DEVICE)
            inference = tokenizer.decode(
                model.generate(contxt, max_pred_tokens=MAX_CONTEXT_LEN)[0].tolist()
                )
            logger_transformer.info('Infernece:\n' +'-'*100 + f'\n\tExample {it + 2} : Starting with {repr(starts_with)}\n' + '-'*100 + f'\nGenerated:{inference}\n' )
        
    logger_transformer.info('✅ Done !')
    
    logger_transformer.info(f'⌚ Script Time: {datetime.now() - started_script}')


if __name__ == '__main__':
    main()