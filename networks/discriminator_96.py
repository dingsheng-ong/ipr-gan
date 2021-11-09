import torch.nn as nn

class Discriminator96(nn.Sequential):
    def __init__(self):
        super(Discriminator96, self).__init__(

            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            _ConvBlock(64, 64, 3, 2, 1),
            _ConvBlock(64, 128, 3, 1, 1),
            _ConvBlock(128, 128, 3, 2, 1),
            _ConvBlock(128, 256, 3, 1, 1),
            _ConvBlock(256, 256, 3, 2, 1),
            _ConvBlock(256, 512, 3, 1, 1),
            _ConvBlock(512, 512, 3, 2, 1),
            
            nn.Conv2d(512, 1024, 6, 1, 0),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(1024, 1, 1, 1, 0)
        )

    def forward(self, x):
        return super(Discriminator96, self).forward(x).squeeze()

class _ConvBlock(nn.Sequential):
    def __init__(self, n_inp, n_out, k, s=1, p=0):
        super(_ConvBlock, self).__init__(
            nn.Conv2d(n_inp, n_out, k, s, p),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(0.2, True)
        )
        nn.init.kaiming_normal_(self[0].weight.data, a=0.2, mode='fan_in')
        self[0].bias.data.zero_()