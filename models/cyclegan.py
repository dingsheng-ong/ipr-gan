from itertools import chain
from models.base import Model
from models.util import ImagePool
from torch import optim
from torch.nn import DataParallel, functional as F
import networks
import torch
import torch.nn as nn

class CycleGAN(Model):
    def __init__(self, config, device=[torch.device('cpu'), ]):
        super(CycleGAN, self).__init__()
        fn_g = getattr(networks, config.G)
        fn_d = getattr(networks, config.D)
        
        self.device = device
        ids = [k.index for k in device]

        self.GA = DataParallel(fn_g().to(device[0]), device_ids=ids)
        self.GB = DataParallel(fn_g().to(device[0]), device_ids=ids)

        self.DA = DataParallel(fn_d().to(device[0]), device_ids=ids)
        self.DB = DataParallel(fn_d().to(device[0]), device_ids=ids)

        self.poolA = ImagePool(config.pool_size)
        self.poolB = ImagePool(config.pool_size)

        self.GA.train()
        self.GB.train()
        self.DA.train()
        self.DB.train()

        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B
        self.lambda_idt = config.lambda_idt

        opt_fn = getattr(optim, config.opt)
        opt_param = config.opt_param.to_dict()

        self.optG = opt_fn(chain(
            self.GA.parameters(),
            self.GB.parameters()
        ), **opt_param)

        self.optD = opt_fn(chain(
            self.DA.parameters(),
            self.DB.parameters()
        ), **opt_param)

        half_epoch = config.epoch // 2
        linear_lr = lambda e: 1.0 - max(0, e - half_epoch) / half_epoch
        self.schedulerG = optim.lr_scheduler.LambdaLR(
            self.optG, lr_lambda=linear_lr
        )
        self.schedulerD = optim.lr_scheduler.LambdaLR(
            self.optD, lr_lambda=linear_lr
        )

        self.MSE = nn.MSELoss()
        self.L1  = nn.L1Loss()

        self._modules['GA'] = self.GA
        self._modules['GB'] = self.GB
        self._modules['DA'] = self.DA
        self._modules['DB'] = self.DB
        self._modules['optG'] = self.optG
        self._modules['optD'] = self.optD
        self._modules['schG'] = self.schedulerG
        self._modules['schD'] = self.schedulerD
        self._modules['poolA'] = self.poolA
        self._modules['poolB'] = self.poolB

    def get_metrics(self):
        return {
            'G/A': self.LossGA.item(),
            'G/B': self.LossGB.item(),
            'G/CycA': self.LossCycA.item(),
            'G/CycB': self.LossCycB.item(),
            'G/IdtA': self.LossIdtA.item(),
            'G/IdtB': self.LossIdtB.item(),
            'G/Sum': self.LossG.item(),
            'D/RealA': self.LossDRA.item(),
            'D/FakeA': self.LossDFA.item(),
            'D/SumA': self.LossDA.item(),
            'D/RealB': self.LossDRB.item(),
            'D/FakeB': self.LossDFB.item(),
            'D/SumB': self.LossDB.item(),
            'LR': self.optG.param_groups[0]['lr'],
        }

    def forward_g(self, data):
        self.real_A = data['real_A']
        self.real_B = data['real_B']

        self.fake_B = self.GA(self.real_A)
        self.fake_A = self.GB(self.real_B)
        
        self.rec_A = self.GB(self.fake_B)
        self.rec_B = self.GA(self.fake_A)
        
        self.idt_A = self.GA(self.real_B)
        self.idt_B = self.GB(self.real_A)

        self.GA_logits = self.DA(self.fake_B)
        self.GB_logits = self.DB(self.fake_A)

    def forward_d(self, data):
        self.real_A = data['real_A']
        self.real_B = data['real_B']
        self.fake_A = self.poolA(data['fake_A'])
        self.fake_B = self.poolB(data['fake_B'])

        self.RA_logits = self.DB(self.real_A)
        self.FA_logits = self.DB(self.fake_A.detach())
        self.RB_logits = self.DA(self.real_B)
        self.FB_logits = self.DA(self.fake_B.detach())

    def compute_g_loss(self):
        self.real_A = self.real_A.to(self.rec_A.device)
        self.real_B = self.real_B.to(self.rec_B.device)
        
        self.LossGA = self.MSE(self.GA_logits, torch.ones_like(self.GA_logits))
        self.LossGB = self.MSE(self.GB_logits, torch.ones_like(self.GB_logits))
        self.LossCycA = self.L1(self.rec_A, self.real_A) * self.lambda_A
        self.LossCycB = self.L1(self.rec_B, self.real_B) * self.lambda_B
        self.LossG = self.LossGA + self.LossGB + self.LossCycA + self.LossCycB

        if self.lambda_idt > 0:
            self.LossIdtA = self.L1(self.idt_A, self.real_B) * self.lambda_B
            self.LossIdtB = self.L1(self.idt_B, self.real_A) * self.lambda_A

            self.LossG += self.lambda_idt * (self.LossIdtA + self.LossIdtB)
        else:
            self.LossIdtA = self.LossIdtB = torch.zeros([])

    def compute_d_loss(self):
        self.LossDRA = self.MSE(self.RB_logits, torch.ones_like(self.RB_logits))
        self.LossDFA = self.MSE(self.FB_logits, torch.zeros_like(self.FB_logits))
        self.LossDA = (self.LossDRA + self.LossDFA) * 0.5

        self.LossDRB = self.MSE(self.RA_logits, torch.ones_like(self.RA_logits))
        self.LossDFB = self.MSE(self.FA_logits, torch.zeros_like(self.FA_logits))
        self.LossDB = (self.LossDRB + self.LossDFB) * 0.5

    def update_lr(self):
        self.schedulerG.step()
        self.schedulerD.step()

    def update_g(self, data, update=True):
        self.forward_g(data)
        self.compute_g_loss()
        
        if update:
            self.optG.zero_grad()
            self.LossG.backward()
            self.optG.step()

    def update_d(self, data):
        self.forward_d(data)
        self.compute_d_loss()

        self.optD.zero_grad()
        self.LossDA.backward()
        self.LossDB.backward()
        self.optD.step()