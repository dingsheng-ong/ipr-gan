import torch.nn as nn

class ConvGenerator(nn.Module):
    def __init__(self, mg, z_dim=128):
        super(ConvGenerator, self).__init__()
        
        block = lambda n_inp, n_out: nn.Sequential(
            nn.ConvTranspose2d(n_inp, n_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )
        self.mg = mg
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512 * mg * mg),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.Sequential(
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), -1, self.mg, self.mg)
        return self.convs(z)

def ConvGenerator32():
    return ConvGenerator(mg=4)

def ConvGenerator64():
    return ConvGenerator(mg=8)