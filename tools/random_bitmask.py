import torch
import torch.nn as nn

class RandomBitMask(nn.Module):
    def __init__(self, config, **kwargs):
        super(RandomBitMask, self).__init__()
        self.n = config.n_bit
        self.c = config.constant
        self.z_dim = config.z_dim
        self.reset()

    def forward(self, z):
        with torch.no_grad():
            mask = self._mask.repeat(z.size(0), 1)
            return z.clone().scatter_(1, mask, self.c)

    def reset(self):
        mask = torch.randperm(self.z_dim)[:self.n].unsqueeze(0)
        if hasattr(self, '_mask'):
            device = self._mask.device
            mask = mask.to(device)
        self.register_buffer('_mask', mask)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask