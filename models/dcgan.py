import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from models.custom import BatchNorm2d


class Generator(nn.Module):

    def __init__(self, ):
        super(Generator, self).__init__()

        self.linear  = nn.Linear(128, 512 * 4 * 4)
        self.network = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        ])

    def forward(self, z, update_stats=True):
        z = F.relu(self.linear(z)).view(z.size(0), -1, 4, 4)
        return reduce(lambda x, f: f(x, update_stats) if isinstance(f, BatchNorm2d) else f(x), self.network, z)


class Discriminator(nn.Module):

    def __init__(self, ):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.linear = nn.utils.spectral_norm(nn.Linear(4 * 4 * 512, 1))

    def forward(self, x):
        return self.linear(self.network(x).view(x.size(0), -1)).view(-1)

