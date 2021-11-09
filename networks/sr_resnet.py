import torch.nn as nn

class SRResNet(nn.Sequential):
    def __init__(self, n_block=16):

        res_blocks = [_ResBlock(nn.Sequential(
            _ConvBlock(64, 64, 3, 1, 1, n=True, a=nn.PReLU()),
            _ConvBlock(64, 64, 3, 1, 1, n=True)
        )) for _ in range(n_block)]
        res_blocks.append(_ConvBlock(64, 64, 3, 1, 1, n=True))

        super(SRResNet, self).__init__(
            _ConvBlock(3, 64, 9, 1, 4, a=nn.PReLU()),
            _ResBlock(nn.Sequential(*res_blocks)),
            _UpBlock(64, 64),
            _UpBlock(64, 64),
            _ConvBlock(64, 3, 9, 1, 4)
        )

class _ConvBlock(nn.Sequential):
    def __init__(self, n_inp, n_out, k, s=1, p=0, n=False, a=None):
        block = [nn.Conv2d(n_inp, n_out, k, s, p), ]
        if n: block += [nn.BatchNorm2d(n_out), ]
        if a: block += [a]
        super(_ConvBlock, self).__init__(*block)

        v = 0.25 if a else 1.0
        nn.init.kaiming_normal_(self[0].weight.data, a=v, mode='fan_in')
        self[0].bias.data.zero_()

class _ResBlock(nn.Module):
    def __init__(self, block):
        super(_ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)

class _UpBlock(nn.Sequential):
    def __init__(self, n_inp, n_out):
        super(_UpBlock, self).__init__(
            _ConvBlock(n_inp, n_out * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )