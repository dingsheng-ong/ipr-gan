from dataset.loader import loaders
from models.srgan import SRResNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from metrics import psnr as calc_psnr
from skimage.measure import compare_ssim
from utils import *
from PIL import Image

import os
import json
import numpy as np
import torch
import torch.nn as nn


BSZ = 100
NCPU = 32
directory = 'logs_srgan/1'

model = SRResNet()
set_bitmask(model)
sd = torch.load(f'{directory}/checkpoint/SRGAN.pt')
model.eval()

config = json.load(open(f'{directory}/config.json'))
dataset = config['dataset']

index = []
for i, layer in enumerate(model.modules()):
    if isinstance(layer, nn.BatchNorm2d):
        index.append(i)
      
def evaluate(model):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device)

    metrics = []
    to_tensor = transforms.ToTensor()
    for dataset in ['Set5', 'Set14', 'BSDS100']:
        eval_images = []
        for r, ds, fs in os.walk(os.path.join('dataset/data/', dataset)):
            eval_images += [os.path.join(r, f) for f in fs]
        
        psnr_scores = StatsAccumulator()
        ssim_scores = StatsAccumulator()

        for img in eval_images:

            hr = Image.open(img).convert('RGB')
            size = list(map(lambda x: x - x % 4, hr.size))
            hr = transforms.CenterCrop(size[::-1])(hr)
            lr = hr.resize(map(lambda x: x // 4, size), Image.BICUBIC)

            with torch.no_grad():
                hr = to_tensor(hr).unsqueeze(0).to(device)
                lr = to_tensor(lr).unsqueeze(0).to(device)

                if hr.size(1) == 1: hr.repeat(1, 3, 1, 1)
                if lr.size(1) == 1: lr.repeat(1, 3, 1, 1)
                sr = model(lr)

                hr = rgb_to_luma(tensor_to_numpy_image(hr.cpu()))
                sr = rgb_to_luma(tensor_to_numpy_image(sr.cpu()))

                hr = hr[4:-4, 4:-4, ...]
                sr = sr[4:-4, 4:-4, ...]

                psnr_score = calc_psnr(sr, hr)
                ssim_score = compare_ssim(sr, hr)

            psnr_scores.update(psnr_score)
            ssim_scores.update(ssim_score)

        metrics.append(float(psnr_scores.avg))
        metrics.append(float(ssim_scores.avg))

    return metrics


n = len(index) * 64
history = open(f'{directory}/sign_flip.csv', 'w')
for percent in range(0, 101, 10):
    model.load_state_dict(sd)
    modules = list(model.modules())
    
    w = np.random.permutation(n)[:int(n * percent / 100)]
    for i in w:
        r = index[i // 64]
        c = i % 64
        modules[r].weight[c].mul_(-1)

    a, b = get_bitmask(model)
    print(sum([c != d for c, d in zip(a, b)]))

    scores = str.join(',', map(str, evaluate(model)))
    print(percent, scores)
    history.write(f'{percent},{scores}\n')
    history.flush()

history.close()
    
