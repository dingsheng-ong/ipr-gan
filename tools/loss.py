from torch.nn import L1Loss, MSELoss
from torchvision.transforms import Normalize
from pytorch_msssim import SSIM, MS_SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['l1', 'mse', 'ms_ssim', 'ssim']

class Loss(object):
    def __init__(self, fn, normalized=False):
        self.fn = fn
        self.denorm = normalized

    def __call__(self, x, y):
        if self.denorm:
            x = (x + 1.) / 2.
            y = (y + 1.) / 2.
        
        return self.fn(x, y)

# class SSIM(nn.Module):
#     def __init__(self, channel=3, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.channel = channel
#         self.padding = window_size // 2
#         self.size_average = size_average
#         self.window = self._create_window(window_size, channel)

#     @staticmethod
#     def _create_window(win_size, channel):
#         gaussian = torch.exp(
#             -(torch.arange(win_size).float() - win_size//2) ** 2 / (2 * 1.5 ** 2)
#         )
#         gaussian = gaussian / gaussian.sum()
#         window = gaussian.ger(gaussian)[None, None, ...]
#         window = window.repeat(channel, 1, 1, 1)
#         return window

#     def forward(self, x, y):
#         _, C, _, _ = x.shape
#         assert C == self.channel
#         assert x.data.type() == x.data.type()

#         if x.is_cuda:
#             self.window = self.window.to(x.get_device())

#         W = self.window
#         P = self.padding

#         mu1 = F.conv2d(x, W, padding=P, groups=C)
#         mu2 = F.conv2d(y, W, padding=P, groups=C)
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2

#         sigma1 = F.conv2d(x*x, W, padding=P, groups=C) - mu1_sq
#         sigma2 = F.conv2d(y*y, W, padding=P, groups=C) - mu2_sq
#         sigma12 = F.conv2d(x*y, W, padding=P, groups=C) - mu1_mu2

#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2

#         ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
#         ssim_map /= (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2)

#         if self.size_average:
#             return ssim_map.mean()
#         else:
#             return ssim_map.mean(dim=[1, 2, 3])

def l1(normalized=False):
    return Loss(L1Loss(), normalized=normalized)

def mse(normalized=False):
    return Loss(MSELoss(), normalized=normalized)

def ms_ssim(normalized=False):
    fn = MS_SSIM(data_range=1)
    return Loss(lambda x, y: 1 - fn(x, y), normalized=normalized)

def ssim(normalized=False):
    fn = SSIM(data_range=1)
    # fn = SSIM()
    return Loss(lambda x, y: 1 - fn(x, y), normalized=normalized)