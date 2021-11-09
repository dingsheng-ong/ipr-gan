from PIL import Image
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn

class PasteWatermark(nn.Module):
    def __init__(self, config, **kwargs):
        super(PasteWatermark, self).__init__()
        self.config = config
        self.normalized = kwargs.get('normalized', False)
        self.position = config.get('position', 'tl')
        assert self.position in ('tl', 'tr', 'bl', 'br'), 'invalid position'
        self._create_watemark()

    def _create_watemark(self):
        size = (self.config.size, ) * 2
        
        tmp = Image.open(self.config.watermark).convert('RGBA')
        tmp = TF.resize(tmp, size)

        img = Image.new('RGBA', size, 'white')
        img.paste(tmp, (0, 0), mask=tmp)
        fg = TF.to_tensor(img.convert('RGB'))

        if self.config.opaque:
            bg = torch.zeros_like(fg[0:1, ...]).float()
        else:
            mask = Image.new('RGBA', size, (0, ) * 4)
            mask.paste(tmp, (0, 0), mask=tmp)
            bg = (TF.to_tensor(mask)[3:, ...] == 0).float()

        self.register_buffer('bg', bg.view(1, 1, *size))
        self.register_buffer('fg', fg.view(1, 3, *size))

        if self.normalized:
            self.fg = self.fg.squeeze(0)
            self.fg = TF.normalize(self.fg, [0.5]*3, [0.5]*3)
            self.fg = self.fg.unsqueeze(0)

        y, x = self.position
        s = self.config.size
        self.y = (None, s) if y == 't' else (-s, None)
        self.x = (None, s) if x == 'l' else (-s, None)

    def forward(self, x):
        hi, hj = self.y
        wi, wj = self.x
        with torch.no_grad():
            y = x.clone()
            y[..., hi:hj, wi:wj] *= self.bg
            y[..., hi:hj, wi:wj] += (1 - self.bg) * self.fg
            return y

    def apply_mask(self, x):
        hi, hj = self.y
        wi, wj = self.x
        with torch.no_grad():
            y = torch.ones_like(x[..., hi:hj, wi:wj])
            y *= self.bg
            y += (1 - self.bg) * x[..., hi:hj, wi:wj]
        return y