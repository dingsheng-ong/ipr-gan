import torch.nn as nn

class ResnetGenerator(nn.Sequential):
    def __init__(self, n_block):
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            channel = 2 ** (i + 6)
            model += [
                nn.Conv2d(channel, channel * 2, 3, 2, 1),
                nn.InstanceNorm2d(channel * 2, affine=True),
                nn.ReLU(True)
            ]

        for _ in range(n_block):
            model += [ ResnetBlock(2 ** (6 + n_downsampling)) ]

        for i in range(2):
            channel = 2 ** (6 + n_downsampling - i)
            model += [
                nn.ConvTranspose2d(channel, channel // 2, 3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(channel // 2, affine=True),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0),
            nn.Tanh()
        ]
        super(ResnetGenerator, self).__init__(*model)

class ResnetBlock(nn.Module):
    def __init__(self, channel):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, 1, 0, bias=True),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, 1, 0, bias=True),
            nn.InstanceNorm2d(channel, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)

def Resnet9Blocks():
    return ResnetGenerator(n_block=9)

def Resnet6Blocks():
    return ResnetGenerator(n_block=6)