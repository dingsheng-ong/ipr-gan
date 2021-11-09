from itertools import chain
from models.base import Model
from torch import optim
from torch.nn import DataParallel, functional as F
import networks
import torch
import torch.nn as nn

class VAE(Model):
    def __init__(self, config, device=[torch.device('cpu', )]):
        super(VAE, self).__init__()
        fn_d = getattr(networks, config.D)
        fn_g = getattr(networks, config.G)

        self.device = device
        ids = [k.index for k in device]

        self.G = DataParallel(fn_g().to(device[0]), device_ids=ids)
        self.D = DataParallel(fn_d().to(device[0]), device_ids=ids)
        self.G.train()
        self.D.train()

        opt_fn = getattr(optim, config.opt)
        opt_param = config.opt_param.to_dict()
        self.optG = opt_fn(
            chain(self.G.parameters(), self.D.parameters()),
            **opt_param
        )

        self._modules['G'] = self.G
        self._modules['D'] = self.D
        self._modules['opt'] = self.optG

    def compute_d_loss(self): pass

    def compute_g_loss(self):
        mean   = self.mean
        logvar = self.logvar
        N = mean.size(0)
        self.kl_loss = (
            (mean ** 2 + logvar.exp() - 1 - logvar) / 2
        ).sum() / N
        self.reconstruct = F.binary_cross_entropy(
            (self.fake_sample + 1.) / 2.,
            (self.real_sample + 1.) / 2.,
            reduction='sum'
        ) / N
        self.LossG = self.kl_loss + self.reconstruct

    def forward_d(self, data):
        self.real_sample = data['real_sample']
        self.latent, (self.mean, self.logvar) = self.D(self.real_sample)
        self.fake_sample = self.G(self.latent)
        self.real_sample = self.real_sample.to(self.fake_sample.device)
        self.generated = self.fake_sample

    def forward_g(self, data): pass
    
    def get_metrics(self):
        return {
            'G/KL': self.kl_loss,
            'G/R': self.reconstruct,
            'G/Sum': self.LossG
        }
    
    def update_d(self, data):
        self.forward_d(data)

    def update_g(self, data, update=True):
        self.compute_g_loss()

        if update:
            self.optG.zero_grad()
            self.LossG.backward()
            self.optG.step()