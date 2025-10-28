import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormal1D:
    def __init__(self, dim , epsilon = 1e-5, momentum = 0.01 ):
        self.dim = dim 
        self.momentum = momentum
        self.epsilon = epsilon

        self.training = True
        
        self.gamma = torch.ones((1,self.dim))
        self.beta = torch.zeros((1,self.dim))

        self.mean_running = torch.zeros((1,self.dim))
        self.std_running = torch.ones((1,self.dim))

    def __call__(self, x):
        if self.training: 
            xmean = x.mean(dim=0 , keepdim=True)
            xstd = x.std(dim=0 , keepdim=True)
            with torch.no_grad():
                self.mean_running = (1-self.momentum) * self.mean_running + self.momentum * xmean
                self.std_running = (1-self.momentum) * self.std_running + self.momentum * xstd
        else :
            xmean = self.mean_running
            xstd = self.std_running
        
        self.out = ( (x - xmean ) / (xstd + self.epsilon) ) * self.gamma + self.beta
        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]


# only change from batch Norm is change of dimension ;D  (we can even delete buffers)
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
    