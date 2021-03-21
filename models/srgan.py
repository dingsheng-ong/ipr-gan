from functools import reduce
from models.custom import BatchNorm2d
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self, cnn):
        super(FeatureExtractor, self).__init__()
        features = cnn.features
        index    = [
            i for i, layer in enumerate(features.children()) \
            if isinstance(layer, nn.MaxPool2d)
        ][4]
        self.features = features[:index]
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std  = torch.Tensor([0.229, 0.224, 0.225])
        self.register_buffer('mean', mean.view(1,3,1,1))
        self.register_buffer('std',   std.view(1,3,1,1))

    def forward(self, x):
        z = (x - self.mean) / self.std
        return self.features(x).squeeze()


class SRResNet(nn.Module):

    def __init__(self, ngb=16, upf=4):
        super(SRResNet, self).__init__()
        resnet_blocks = [_resnet_block(64, 64) for _ in range(ngb)]
        resnet_blocks.append(BNConv2dBlock(64, 64, p=1, n=True))
        self.network = nn.ModuleList([
            BNConv2dBlock(3, 64, k=9, p=4, a='PReLU()'),
            ResidualBlock(nn.Sequential(*resnet_blocks)),
            *[UpsampleBlock(64, 64) for _ in range(upf // 2)],
            BNConv2dBlock(64, 3, k=9, p=4),
        ])

    def forward(self, x, update_stats=True):
        return reduce(lambda z, f: f(z, update_stats), self.network, x)


class Discriminator_96(nn.Module):

    def __init__(self):
        super(Discriminator_96, self).__init__()
        self.network = nn.Sequential(
            Conv2dBlock(3,   64,  s=1, p=1, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(64,  64,  s=2, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(64,  128, s=1, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(128, 128, s=2, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(128, 256, s=1, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(256, 256, s=2, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(256, 512, s=1, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(512, 512, s=2, p=1, n=True, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(512, 1024, k=6, a='LeakyReLU(0.2, True)'),
            Conv2dBlock(1024, 1, k=1),
        )

    def forward(self, x):
        return self.network(x).squeeze()


def _resnet_block(nc_inp, nc_out):
    return ResidualBlock(nn.ModuleList([
        BNConv2dBlock(nc_inp, nc_out, p=1, n=True, a='PReLU()'),
        BNConv2dBlock(nc_inp, nc_out, p=1, n=True),
    ]))


class ResidualBlock(nn.Module):
    
    def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.block = block

    def forward(self, x, update_stats=True):
        return x + reduce(lambda z, f: f(z, update_stats), self.block, x)

    def __repr__(self):
        repr_string = str(self.block).split('\n')
        repr_string = list(map(lambda x: '\n│' + x, repr_string))
        repr_string[-1] = repr_string[-1][0] + '▼' + repr_string[-1][2:]
        return self._get_name() + ''.join(repr_string)


class BNConv2dBlock(nn.Module):

    def __init__(self, nc_inp, nc_out, k=3, s=1, p=0, n=False, a=None):
        super(BNConv2dBlock, self).__init__()
        self.P = nn.ZeroPad2d(p) if p else lambda x: x
        self.C = nn.Conv2d(nc_inp, nc_out, k, s)
        self.N = BatchNorm2d(nc_out) if n else lambda x, z: x
        self.A = eval(f'nn.{a}') if a else lambda x: x
        self.weight_init()

    def weight_init(self):
        a = {
            nn.LeakyReLU: 0.2 ,
            nn.PReLU    : 0.25,
            nn.ReLU     : 0.0 ,
        }.get(self.A.__class__, 1.0)

        nn.init.kaiming_normal_(self.C.weight, a=a, mode='fan_in')
        self.C.bias.data.zero_()

        if isinstance(self.N, nn.BatchNorm2d):
            self.N.weight.data.fill_(1)
            self.N.bias.data.zero_()

    def forward(self, x, update_stats=True):
        return self.A(self.N(self.C(self.P(x)), update_stats))


class Conv2dBlock(nn.Module):

    def __init__(self, nc_inp, nc_out, k=3, s=1, p=0, n=False, a=None):
        super(Conv2dBlock, self).__init__()
        self.P = nn.ZeroPad2d(p) if p else lambda x: x
        self.C = nn.Conv2d(nc_inp, nc_out, k, s)
        self.N = nn.BatchNorm2d(nc_out) if n else lambda x: x
        self.A = eval(f'nn.{a}') if a else lambda x: x
        self.weight_init()

    def weight_init(self):
        a = {
            nn.LeakyReLU: 0.2 ,
            nn.PReLU    : 0.25,
            nn.ReLU     : 0.0 ,
        }.get(self.A.__class__, 1.0)

        nn.init.kaiming_normal_(self.C.weight, a=a, mode='fan_in')
        self.C.bias.data.zero_()

        if isinstance(self.N, nn.BatchNorm2d):
            self.N.weight.data.fill_(1)
            self.N.bias.data.zero_()

    def forward(self, x):
        return self.A(self.N(self.C(self.P(x))))


class UpsampleBlock(nn.Module):

    def __init__(self, nc_inp, nc_out, upf=2):
        super(UpsampleBlock, self).__init__()
        self.block =  nn.ModuleList([
            BNConv2dBlock(nc_inp, nc_out * (upf ** 2), p=1),
            nn.PixelShuffle(upf),
            nn.PReLU(),
        ])

    def forward(self, x, update_stats=True):
        return reduce(lambda z, f: f(z, update_stats) if isinstance(f, BNConv2dBlock) else f(z), self.block, x)


if __name__ == '__main__':
    from torchvision.models import vgg19
    F = FeatureExtractor(vgg19(pretrained=True))
    G = SRResNet(16, 4)
    D = Discriminator_96()
    x = torch.randn(64, 3, 24, 24)

    print('======================================= FeatureExtractor =======================================')
    print(F)
    print('Input :', list(x.size()))
    print('Output:', list(F(x).size()))

    print('=========================================== SRResNet ===========================================')
    print(G)
    print('Input :', list(x.size()))
    print('Output:', list(G(x).size()))

    print('======================================= Discriminator_96 =======================================')
    print(D)
    print('Input :', list(G(x).size()))
    print('Output:', list(D(G(x)).size()))
