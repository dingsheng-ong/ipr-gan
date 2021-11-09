from models.base import Model
from torch import optim
from torch.nn import DataParallel, functional as F
import networks
import torch

class DCGAN(Model):
    def __init__(self, config, device=[torch.device('cpu'), ]):
        super(DCGAN, self).__init__()
        fn_g = getattr(networks, config.G)
        fn_d = getattr(networks, config.D)
        
        self.device = device
        ids = [k.index for k in device]

        self.G = DataParallel(fn_g().to(device[0]), device_ids=ids)
        self.D = DataParallel(fn_d().to(device[0]), device_ids=ids)
        self.G.train()
        self.D.train()

        opt_fn = getattr(optim, config.opt)
        opt_param = config.opt_param.to_dict()
        self.optG = opt_fn(self.G.parameters(), **opt_param)
        self.optD = opt_fn(self.D.parameters(), **opt_param)

        self._modules['G'] = self.G
        self._modules['D'] = self.D
        self._modules['optG'] = self.optG
        self._modules['optD'] = self.optD

    def compute_d_loss(self):
        # hinge loss
        self.LossR = F.relu(1. - self.real_logits).mean()
        self.LossF = F.relu(1. + self.fake_logits).mean()
        self.LossD = self.LossR + self.LossF

    def compute_g_loss(self):
        # adversarial loss
        self.LossA = - self.gen_logits.mean()
        self.LossG = self.LossA

    def forward_d(self, data):
        
        self.latent      = data['latent']
        self.real_sample = data['real_sample']
        self.fake_sample = self.G(self.latent)
        self.real_logits = self.D(self.real_sample)
        self.fake_logits = self.D(self.fake_sample.detach())

    def forward_g(self, data):
        self.generated  = data['fake_sample']
        self.gen_logits = self.D(self.generated)

    def get_metrics(self):
        return {
            'D/Sum': self.LossD.item(),
            'D/Real': self.LossR.item(),
            'D/Fake': self.LossF.item(),
            'G/Sum': self.LossG.item(),
            'G/Adv': self.LossA.item()
        }

    def update_d(self, data):
        self.forward_d(data)
        self.compute_d_loss()

        self.optD.zero_grad()
        self.LossD.backward()
        self.optD.step()

    def update_g(self, data, update=True):
        self.forward_g(data)
        self.compute_g_loss()
        
        if update:
            self.optG.zero_grad()
            self.LossG.backward()
            self.optG.step()