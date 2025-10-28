import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormal1D:
    def __init__(self, dim , epsilon = 1e-5, device='cpu'):
        self.dim = dim 
        self.epsilon = epsilon
        
        self.gamma = torch.ones((1,self.dim)).to(device)
        self.beta = torch.zeros((1,self.dim)).to(device)

    def __call__(self, x):
        xmean = x.mean(dim=1 , keepdim=True)
        xstd = x.std(dim=1 , keepdim=True)
        
        self.out = ( (x - xmean ) / (xstd + self.epsilon) ) * self.gamma + self.beta
        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]
    