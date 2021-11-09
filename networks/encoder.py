import torch
import torch.nn as nn

class Encoder32(nn.Module):
    def __init__(self):
        super(Encoder32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1)
        )

        self.q_mean = nn.Linear(2048, 128)
        self.q_logvar = nn.Linear(2048, 128)

    def forward(self, x):
        q = self.encoder(x)
        q = q.flatten(start_dim=1)

        mean, logvar = self.q_mean(q), self.q_logvar(q)
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        
        return z, (mean, logvar)