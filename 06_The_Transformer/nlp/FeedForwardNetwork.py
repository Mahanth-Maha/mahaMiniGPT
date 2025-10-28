import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):
    def __init__(self, nonlinearity, in_dim, hidden_dim, out_dim = None, dropout =0.7, device = 'cpu'):
        super(FeedForwardNet, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.device = device 
        
        self.layer1 = nn.Linear(self.in_dim, self.hidden_dim).to(device)
        self.layer2 = nn.Linear(self.hidden_dim, self.out_dim).to(device)
        if nonlinearity == 'leakyrelu':
            self.non_linearity = nn.LeakyReLU(0.01)
        elif nonlinearity == 'silu':
            self.non_linearity = nn.SiLU() # sigmoid
        elif nonlinearity == 'gelu':
            self.non_linearity = nn.GELU() # gaussuian
        elif nonlinearity == 'softplus':
            self.non_linearity = nn.Softplus()
        else:
            self.non_linearity = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.non_linearity(self.layer1(x))
        x = self.dropout_layer(self.non_linearity(self.layer2(x)))
        return x
        