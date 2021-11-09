from torch.nn.utils import spectral_norm as SN
import torch.nn as nn

class SNDiscriminator(nn.Module):
    def __init__(self, md):
        super(SNDiscriminator, self).__init__()

        block = lambda n_inp, n_out: nn.Sequential(
            SN(nn.Conv2d(n_inp, n_out, 3, 1, 1, bias=True)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            SN(nn.Conv2d(n_out, n_out, 4, 2, 1, bias=True)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.net = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            SN(nn.Conv2d(256, 512, 3, 1, 1, bias=True)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Flatten(),
            SN(nn.Linear(512 * md * md, 1))
        )

    def forward(self, x):
        return self.net(x).view(-1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.flatten(1)

def SNDiscriminator32():
    return SNDiscriminator(md=4)

def SNDiscriminator64():
    return SNDiscriminator(md=8)