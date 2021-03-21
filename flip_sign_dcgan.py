from dataset.loader import loaders
from models.dcgan import Generator
from torch.utils.data import Dataset, DataLoader
from evaluate_dcgan import get_fid_is
from tqdm import tqdm
from utils import *

import json
import numpy as np
import torch
import torch.nn as nn


BSZ = 100
NCPU = 32
directory = 'archive_dcgan/1'

generator = Generator()
set_bitmask(generator)
sd = torch.load(f'{directory}/checkpoint/generator.pt')
generator.eval()

config = json.load(open(f'{directory}/config.json'))
dataset = config['dataset']

index = []
for i, layer in enumerate(generator.network):
    if isinstance(layer, nn.BatchNorm2d):
        for j in range(layer.weight.size(0)):
            index.append([i, j])


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item], 0

    def __len__(self):
        return self.data.size(0)


def evaluate(model, percent, Z):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    real_loader = loaders[dataset](
        img_size=32,
        batch_size=BSZ,
        num_workers=NCPU,
    )
    n = len(real_loader.dataset)

    model.to(device)
    model.eval()

    fake_imgs = []
    real_imgs = []
    Zs = iter(Z)
    for (img, _) in tqdm(real_loader, desc="Generating images", leave=False):
        bsz = img.size(0)
        with torch.no_grad():
            z = next(Zs)[0].to(device)
            fake_imgs.append(generator(z).cpu())
            img = img.to(device)
            real_imgs.append(img.cpu())

    fake_imgs = torch.cat(fake_imgs, dim=0)
    real_imgs = torch.cat(real_imgs, dim=0)
    torch.save(fake_imgs, f'fake_imgs-{percent}.pt')
    fake_loader = DataLoader(
        CustomDataset(fake_imgs),
        batch_size=BSZ,
        num_workers=NCPU,
        pin_memory=True
    )
    del real_loader
    real_loader = DataLoader(
        CustomDataset(real_imgs),
        batch_size=BSZ,
        num_workers=NCPU,
        pin_memory=True
    )

    # fid, is_mean, is_std = get_fid_is(real_loader, fake_loader, device)

    return 0, 0, 0

n = len(index)
history = open(f'{directory}/sign_flip.csv', 'w')
Z = DataLoader(CustomDataset(torch.randn(60000, 128)), batch_size=BSZ, num_workers=NCPU, pin_memory=True)
for percent in range(0, 101, 10):
    generator.load_state_dict(sd)
    w = np.random.permutation(n)[:int(n * percent / 100)]
    for i in w:
        j, k = index[i]
        generator.network[j].weight[k].mul_(-1)

    fid, is_mean, is_std = evaluate(generator, percent, Z)
    history.write(f'{percent},{fid},{is_mean},{is_std}\n')
    history.flush()

history.close()
    
