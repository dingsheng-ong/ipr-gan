from models.base import Model
from torch import optim
from torch.nn import DataParallel, functional as F
import networks
import torch

class SRGAN(Model):
    def __init__(self, config, device=[torch.device('cpu'), ]):
        super(SRGAN, self).__init__()
        fn_g = getattr(networks, config.G)
        fn_d = getattr(networks, config.D)
        fn_v = getattr(networks, config.V)
        
        self.device = device
        ids = [k.index for k in device]

        self.G = DataParallel(fn_g().to(device[0]), device_ids=ids)
        self.D = DataParallel(fn_d().to(device[0]), device_ids=ids)
        self.V = DataParallel(fn_v().to(device[0]), device_ids=ids)
        self.G.train()
        self.D.train()
        self.V.eval()

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
        self.LossR = F.binary_cross_entropy_with_logits(
            self.real_logits,
            torch.ones_like(self.real_logits)
        )
        self.LossF = F.binary_cross_entropy_with_logits(
            self.fake_logits,
            torch.zeros_like(self.fake_logits)
        )
        self.LossD = self.LossR + self.LossF

    def compute_g_loss(self):
        # adversarial loss
        device = self.super_res.device
        if self.pretrain:
            self.LossG = F.mse_loss(self.super_res, self.high_res.to(device))
        else:
            self.LossA = F.binary_cross_entropy_with_logits(
                self.gen_logits,
                torch.ones_like(self.gen_logits)
            )
            sr_feat = self.V(self.super_res)
            hr_feat = self.V(self.high_res).detach()
            self.LossX = F.mse_loss(sr_feat, hr_feat)
            self.LossG = self.LossX + 1e-3 * self.LossA

    def forward_d(self, data):
        self.high_res  = data['high_res']
        self.super_res = data['super_res']
        self.real_logits = self.D(self.high_res)
        self.fake_logits = self.D(self.super_res.detach())

    def forward_g(self, data):
        self.low_res  = data['low_res']
        self.high_res = data['high_res']
        self.pretrain = data['pretrain']
        self.super_res = self.G(self.low_res)

        if not self.pretrain:
            self.gen_logits = self.D(self.super_res)

    def get_metrics(self):
        if self.pretrain:
            return {
                'G/MSE': self.LossG.item(),
                'G/Sum': self.LossG.item(),
            }
        else:
            return {
                'D/Sum': self.LossD.item(),
                'D/Real': self.LossR.item(),
                'D/Fake': self.LossF.item(),
                'G/Sum': self.LossG.item(),
                'G/Adv': self.LossA.item(),
                'G/Con': self.LossX.item(),
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