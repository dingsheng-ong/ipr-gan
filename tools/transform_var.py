import math
import torch
import torch.nn as nn

class TransformVar(nn.Module):
    def __init__(self, config, **kwargs):
        super(TransformVar, self).__init__()
        self.register_buffer('w', torch.ones(1, 128))
        self.register_buffer('a', torch.ones(1, 128))
        self.reset()

    def forward(self, z):
        return z * (1 - self.a) + self.a * self.w

    def reset(self):
        self.w = torch.exp(torch.randn_like(self.w).abs())
        self.a = (torch.rand(1, 128) < 0.25).float()