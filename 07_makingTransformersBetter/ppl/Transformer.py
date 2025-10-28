""" 
Current Model: 

optimizations DONE:
    0. [NONE] just copied into one file


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
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Batch size")
    parser.add_argument("--context_length", type=int, default=1024, help="Block size (context length)")
    parser.add_argument("--pred_char_len", type=int, default=1000, help="Prediction length for generation")
    parser.add_argument("--Nx_blocks", type=int, default=4, help="No of Transformer Blocks")
    parser.add_argument("--n_heads", type=int, default=8, help="Heads in MHSA")
    parser.add_argument("--n_embeddings", type=int, default=728, help="Embedding dimension")
    parser.add_argument("--ffn_hid_dim", type=int, default=None, help="Hidden dimension of FFN")
    parser.add_argument("--non_linearity", type=str, default='gelu', help="non linearity in FFN")
    parser.add_argument("--dropout", type=int, default=0.5, help="dropouts")
    parser.add_argument("--generate", default=False, help="Toggle on to generate from model")
    parser.add_argument("--log", default='info', help="set : debug, info, error, warning ")
    return parser.parse_args()




if __name__ == '__main__':
    args = get_args()
    import numpy as np
    import logging
    
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
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger_transformer.info(f"{DEVICE = }")
    
    
    options = dict(
        vocab_size =args.vocab_size,
        context_length =args.context_length,
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
    time_dim = 12
    
    data_X = np.random.randint(0, args.vocab_size, (batch_size, time_dim))
    data_y = np.random.randint(0, args.vocab_size, (batch_size * time_dim))
    random_X = torch.tensor(data_X, dtype=torch.long, device=DEVICE)
    random_y = torch.tensor(data_y, dtype=torch.long, device=DEVICE)
    logger_transformer.info(f"{random_X.shape = } {random_y.shape = }")
    
    # logits = gpt(random_X)
    # logger_transformer.info(f"{logits.shape = }")
    
    logits, loss = gpt(random_X, random_y)
    
    logger_transformer.info(f"{logits.shape = } {loss.item() = }")