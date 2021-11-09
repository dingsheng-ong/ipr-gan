import torch.nn as nn

class Decoder32(nn.Sequential):

    class Reshape(nn.Module):
        def __init__(self, *shape):
            super(Decoder32.Reshape, self).__init__()
            self.shape = shape

        def forward(self, x):
            return x.view(-1, *self.shape)

    class Normalize(nn.Module):
        def forward(self, x):
            return x * 2 - 1

    def __init__(self):
        super(Decoder32, self).__init__(

            nn.Linear(128, 2048),
            Decoder32.Reshape(128, 4, 4),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
            Decoder32.Normalize(),
        )