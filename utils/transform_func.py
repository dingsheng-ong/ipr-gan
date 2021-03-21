import os
import random

import torch

from torchvision import transforms

from PIL import Image
from utils import create_watermark_mask


class RandomBitMask(object):

    def __init__(self, dim=100, nbit=10, const=-10, device=torch.device('cpu')):
        self.dim    = dim
        self.nbit   = nbit
        self.const  = const
        self.device = device
        self.reset_mask()

    def __call__(self, z):
        with torch.no_grad():
            return z.clone().scatter_(1, self.mask.repeat(z.size(0), 1), self.const)

    def save(self, path):
        self.mask = self.mask.cpu()
        torch.save(self.mask, path)
        self.mask = self.mask.to(self.device)

    def load(self, path):
        self.mask = torch.load(path)
        self.mask = self.mask.to(self.device)

    def reset_mask(self):
        self.mask = torch.randperm(self.dim)[:self.nbit].unsqueeze(0).to(self.device)


class ApplyWatermark(object):

    def __init__(self, watermark, wtmk_size=16, norm=False, device=torch.device('cpu')):
        w = create_watermark_mask(watermark, wtmk_size=wtmk_size, norm=norm, device=device)
        self.path = watermark
        self.norm = norm
        self.obj_tmp = w['object']
        self.msk_tmp = w['mask']
        self.obj = torch.empty(1, 1, 1, 1).to(device)
        self.msk = torch.empty(1, 1, 1, 1).to(device)
        self.img_size = (-1, -1, -1)

    def __call__(self, image, position='top-left'):
        bsz, ci, hi, wi = image.size()
        _, hw, ww     = self.obj_tmp.size()

        if position == 'top-left':
            idx_h, idx_w = (0, 0)
        elif position == 'top-right':
            idx_h, idx_w = (0, max(wi - ww, 0))
        elif position == 'bottom-left':
            idx_h, idx_w = (max(hi - hw, 0), 0)
        elif position == 'bottom-right':
            idx_h, idx_w = (max(hi - hw, 0), max(wi - ww, 0))
        elif position == 'center':
            idx_h, idx_w = (max(hi // 2 - hw // 2, 0), max(wi // 2 - ww // 2, 0))
        elif position == 'random':
            h = max(hi - hw, 0)
            w = max(wi - ww, 0)
            idx_h, idx_w = (random.randint(0, h), random.randint(0, w))
        else:
            idx_h, idx_w = (0, 0)

        dh = min(hi, hw)
        dw = min(wi, ww)

        if not self.img_size == (bsz, hi, wi):

            self.obj.resize_(image.size()).fill_(-1 if self.norm else 0)
            self.msk.resize_(image.size()).fill_(1)

            self.obj[..., idx_h:idx_h+dh, idx_w:idx_w+dw].copy_(self.obj_tmp[:, :dh, :dw])
            self.msk[..., idx_h:idx_h+dh, idx_w:idx_w+dw].copy_(self.msk_tmp[:, :dh, :dw])

            self.img_size = (bsz, hi, wi)

        with torch.no_grad():
            return image * self.msk + (1 - self.msk) * self.obj

    def save(self, path):
        os.system(f'cp {self.path} {path}')


class Grayscale(object):

    def __init__(self):
        self.f = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        device = image.device
        image  = image.detach().cpu()
        return torch.cat([self.f(image[i]).unsqueeze(0) for i in range(image.size(0))]).to(device)


class FillWatermark(object):

    def __init__(self, watermark, norm=False):
        self.path = watermark
        wm = Image.open(watermark)
        self.watermark = Image.new("RGB", wm.size, (255, 255, 255))
        self.watermark.paste(wm, mask=wm.split()[3])
        if norm:
            self.transform = lambda size: transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, ] * 3, [0.5, ] * 3),
                transforms.Lambda(lambda x: x.unsqueeze(0)),
            ])
        else:
            self.transform = lambda size: transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.unsqueeze(0)),
            ])
        self.cached_size = (0, 0)
        self.cached_watermark = None

    def __call__(self, image):
        bsz, _, h, w = image.size()
        if not self.cached_size == (h, w):
            self.cached_size = (h, w)
            self.cached_watermark = self.transform((h, w))(self.watermark)
        return self.cached_watermark.repeat(bsz, 1, 1, 1).to(image.device)

    def save(self, path):
        os.system(f'cp {self.path} {path}')


class InvertColor(object):

    def __init__(self, norm=False):
        self.f = (lambda x: x * -1) if norm else (lambda x: 1 - x)

    def __call__(self, image):
        with torch.no_grad(): return self.f(image)


class ApplyBlackPatch(object):

    def __init__(self, patch_size=16, norm=False, device=torch.device('cpu')):
        self.norm = norm
        self.patch = torch.zeros(3, patch_size, patch_size).to(device)
        self.patch_cache = torch.empty(1, 1, 1, 1).to(device)
        if norm: self.patch.fill_(-1)
        self.img_size = (-1, -1, -1)

    def __call__(self, image, position='top-left'):
        bsz, ci, hi, wi = image.size()
        _, hw, ww     = self.patch.size()

        if position == 'top-left':
            idx_h, idx_w = (0, 0)
        elif position == 'top-right':
            idx_h, idx_w = (0, max(wi - ww, 0))
        elif position == 'bottom-left':
            idx_h, idx_w = (max(hi - hw, 0), 0)
        elif position == 'bottom-right':
            idx_h, idx_w = (max(hi - hw, 0), max(wi - ww, 0))
        elif position == 'center':
            idx_h, idx_w = (max(hi // 2 - hw // 2, 0), max(wi // 2 - ww // 2, 0))
        elif position == 'random':
            h = max(hi - hw, 0)
            w = max(wi - ww, 0)
            idx_h, idx_w = (random.randint(0, h), random.randint(0, w))
        else:
            idx_h, idx_w = (0, 0)

        dh = min(hi, hw)
        dw = min(wi, ww)

        if not self.img_size == (bsz, hi, wi):

            self.patch_cache.resize_(image.size()).fill_(-1 if self.norm else 0)
            self.patch_cache[..., idx_h:idx_h+dh, idx_w:idx_w+dw].copy_(self.patch[:, :dh, :dw])

            self.img_size = (bsz, hi, wi)

        image[..., idx_h:idx_h+dh, idx_w:idx_w+dw] = self.patch_cache[..., :dh, :dw]
        return image

    def save(self, path):
        device = self.patch.device
        transforms.ToPILImage()(self.patch.detach()).save(path)
        self.patch = self.patch.to(device)


class ApplyRandomPatch(object):

    def __init__(self, patch_size=16, norm=False, device=torch.device('cpu')):
        self.norm = norm
        self.patch = torch.rand(3, patch_size, patch_size).to(device)
        self.patch_cache = torch.empty(1, 1, 1, 1).to(device)
        if norm: self.patch = self.patch * 2 - 1
        self.img_size = (-1, -1, -1)

    def __call__(self, image, position='top-left'):
        bsz, ci, hi, wi = image.size()
        _, hw, ww     = self.patch.size()

        if position == 'top-left':
            idx_h, idx_w = (0, 0)
        elif position == 'top-right':
            idx_h, idx_w = (0, max(wi - ww, 0))
        elif position == 'bottom-left':
            idx_h, idx_w = (max(hi - hw, 0), 0)
        elif position == 'bottom-right':
            idx_h, idx_w = (max(hi - hw, 0), max(wi - ww, 0))
        elif position == 'center':
            idx_h, idx_w = (max(hi // 2 - hw // 2, 0), max(wi // 2 - ww // 2, 0))
        elif position == 'random':
            h = max(hi - hw, 0)
            w = max(wi - ww, 0)
            idx_h, idx_w = (random.randint(0, h), random.randint(0, w))
        else:
            idx_h, idx_w = (0, 0)

        dh = min(hi, hw)
        dw = min(wi, ww)

        if not self.img_size == (bsz, hi, wi):

            self.patch_cache.resize_(image.size()).fill_(-1 if self.norm else 0)
            self.patch_cache[..., idx_h:idx_h+dh, idx_w:idx_w+dw].copy_(self.patch[:, :dh, :dw])

            self.img_size = (bsz, hi, wi)

        image[..., idx_h:idx_h+dh, idx_w:idx_w+dw] = self.patch_cache[..., :dh, :dw]
        return image

    def save(self, path):
        device = self.patch.device
        transforms.ToPILImage()(self.patch.cpu()).save(path)
        self.patch = self.patch.to(device)

    def load(self, path):
        device = self.patch.device
        patch = transforms.ToTensor()(Image.open(path))
        if self.norm: patch = patch * 2 - 1
        self.patch = patch.to(device)

    def reset_mask(self):
        self.patch = torch.rand_like(self.patch).to(self.patch.device)
