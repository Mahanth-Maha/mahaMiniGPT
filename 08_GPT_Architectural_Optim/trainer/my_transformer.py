
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, context_length, model_dimension, n_heads, dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-MHSA] {self.variable_name.shape = } \ttype({type(self.variable_name)})')
        
        self.context_length = context_length
        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.each_head_size = self.model_dimension // self.n_heads
        self.dropout = dropout
        self.device = device
        
        self.attn_norm = self.each_head_size ** -0.5
        
        self.W_qkv = nn.Linear(self.model_dimension, 3 * self.model_dimension , bias=False, device=self.device)

        self.register_buffer("mask", torch.tril(torch.ones(self.context_length,self.context_length, device=self.device)).view(1, 1, context_length, context_length))
        
        self.proj_attn = nn.Linear(self.model_dimension,self.model_dimension, device=self.device)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self,x):
        B,T,C = x.shape

  
        Q,K,V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)

        attn = (Q @ K.transpose(-2,-1) ) * self.attn_norm
        
        masked_attn = attn.masked_fill_(self.mask[:, :, :T, :T] == 0, float("-inf"))
        scores = F.softmax(masked_attn, dim=-1)

        scores = self.dropout_layer(scores)

        out = scores @ V
        out = out.transpose(1,2).reshape(B, T, C)
        
        return self.dropout_layer(self.proj_attn(out))


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc_a = nn.Linear(dim_in, dim_out)
        self.fc_b = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        a = self.fc_a(x) 
        b = F.silu(self.fc_b(x))
        return a * b

class SwiGLUFast(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2 * dim_out)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * F.silu(b)

class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ffn_hid_dim = None, non_linearity = 'gelu', dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-FFN] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.model_dimension = model_dimension
        ffn_factor = 8/3 # 4 is Original , 8/3 is PALM
        self.ffn_hid_dim = ffn_hid_dim if ffn_hid_dim is not None else int(ffn_factor * model_dimension)
        self.dropout = dropout 
        self.device = device
        
        
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'leakyrelu':
            self.non_linearity = nn.LeakyReLU(0.01)
        elif non_linearity == 'gelu':
            self.non_linearity = nn.GELU()
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity == 'glu':
            self.non_linearity = nn.GLU()
        elif non_linearity == 'silu' or  non_linearity == 'swish':
            self.non_linearity = nn.SiLU()
        elif non_linearity == 'swiglu':
            self.non_linearity = SwiGLU(self.model_dimension, self.ffn_hid_dim).to(self.device)
        elif non_linearity == 'swiglufast':
            self.non_linearity = SwiGLUFast(self.model_dimension, self.ffn_hid_dim).to(self.device)
        else:
            print(f'[WARNING] Specified ({non_linearity}) is not supported, falling back to `ReLU` non-linearity')
            # logging.warning(f' Specified ({non_linearity}) is not supported, falling back to `ReLU` non-linearity')
            self.non_linearity = nn.ReLU()
            
        if non_linearity == 'swiglu' or non_linearity == 'swiglufast':
            self.ln_ffn = None
        elif non_linearity == 'glu':
            self.ln_ffn = nn.Linear(self.model_dimension, 2 * self.ffn_hid_dim).to(self.device)
        else:
            self.ln_ffn = nn.Linear(self.model_dimension, self.ffn_hid_dim).to(self.device)
        
        self.proj_ffn = nn.Linear(self.ffn_hid_dim, self.model_dimension, device=self.device)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        if isinstance(self.non_linearity, SwiGLUFast) or isinstance(self.non_linearity, SwiGLU) :
            x = self.non_linearity(x)
        else:
            x = self.non_linearity(self.ln_ffn(x))  
        # x = self.non_linearity(self.ln_ffn(x)) 
        return self.dropout_layer(self.proj_ffn(x))


class DecoderBlock(nn.Module):
    def __init__(self, context_length, model_dimension, n_heads, ffn_hid_dim = None,  non_linearity ='gelu', dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-DECODER] {self.variable_name = } , type({type(self.variable_name)})')
        
        self.context_length = context_length
        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.ffn_hid_dim = ffn_hid_dim
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.device = device
        self.layer_norm1 = nn.RMSNorm(self.model_dimension, device=self.device)
        self.attention = MultiHeadSelfAttention(self.context_length, self.model_dimension, self.n_heads, self.dropout, self.device)
        self.layer_norm2 = nn.RMSNorm(self.model_dimension, device=self.device)
        self.ffn = FeedForwardNet(self.model_dimension, self.ffn_hid_dim, self.non_linearity, self.dropout, self.device)
        
        
    def forward(self, x, y=None):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size, context_length, model_dimension, n_heads, Nx, ffn_hid_dim = None, non_linearity ='gelu', dropout = 0.7, device = 'cpu'):
        super().__init__()
        # print(f'[INFO-Transformer] {self.variable_name = } , type({type(self.variable_name)})')

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.Nx = Nx
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
                    dropout=dropout,
                    device=device,
                    non_linearity=non_linearity
                ) for _ in range(self.Nx)
            ]
        )
        
        self.lm_head = nn.Linear(self.model_dimension, self.vocab_size, bias= False,device=self.device)
        
        self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=(2 * self.Nx) ** -0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
