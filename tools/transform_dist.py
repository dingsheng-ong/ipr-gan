import math
import torch
import torch.nn as nn

class TransformDist(nn.Module):
    def __init__(self, config, **kwargs):
        super(TransformDist, self).__init__()

    def forward(self, z):
        y = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        return y * math.sqrt(2 * math.pi)

    def reset(self): pass