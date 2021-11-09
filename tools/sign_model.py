import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class BitGenerator:
    def __init__(self, string=None):
        self.random = string is None
        if string:
            assert isinstance(string, str)
            self.string = ''.join([f'{ord(c):08b}' for c in (string + '\t')])
            self.string = list(map(int, self.string))
        self.index = 0

    def __next__(self):
        if self.random:
            return random.randint(0, 1)
        else:
            bit = self.string[self.index % len(self.string)]
            self.index += 1
            return bit

    def get(self, n):
        return [next(self) for _ in range(n)]

class SignLossModel(nn.Module):
    def __init__(self, model, config, **kwargs):
        super(SignLossModel, self).__init__()
        self.gamma_0 = config.gamma_0
        self.bit_gen = BitGenerator(config.string)
        self._create_signs(model)

    def _create_signs(self, model):
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                safe_name = name.replace('.', '_')
                sign = self.bit_gen.get(m.weight.size(0))
                sign = torch.FloatTensor(sign) * 2 - 1
                m.weight.data.abs_().mul_(sign.to(m.weight.data.device))
                self.register_buffer(safe_name, sign)
    
    def forward(self, model):
        loss = 0
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                safe_name = name.replace('.', '_')
                sign = getattr(self, safe_name)
                loss += F.relu(self.gamma_0 - m.weight * sign).mean()
        return loss

    def compute_ber(self, model):
        bit_error  = 0
        bit_length = 0
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                safe_name = name.replace('.', '_')
                sign = getattr(self, safe_name)
                bit_error  += (m.weight.sign() != sign).float().sum()
                bit_length += sign.size(0)
        return bit_error / bit_length