import argparse
import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from dataset.loader import load_cifar10, load_mnist
from models.dcgan import Generator, Discriminator
from metrics import forward_inception, calc_entropy, FID
from utils import create_watermark_mask


SEED = 123

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item], 0

    def __len__(self):
        return self.data.size(0)


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    https://github.com/zepx/pytorch-weight-prune/blob/develop/pruning/methods.py
    '''
    if pruning_perc == 0:
        return

    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() < threshold
            p.data[pruned_inds] = 0


def eval_watermark_quality(percent, mask, generator, discriminator, loader_r):
  
    q_watermark = 0
    n = len(loader_r.dataset)
    images = []

    start = time.time()

    for data, _ in loader_r:

        bs = data.size(0)
        z = torch.randn(bs, 128).cuda()

        with torch.no_grad():
            img = generator(z)
            z[:, mask] = -10
            wtmk_quality = nn.functional.relu(1 + discriminator(generator(z))).mean()

        images.append(img.cpu())
        
        q_watermark = (q_watermark * n + wtmk_quality.item() * bs) / (n + bs)

    images = torch.cat(images, dim=0)
    loader_f = DataLoader(
        CustomDataset(images),
        batch_size=bs,
        num_workers=8,
        pin_memory=True
    )

    return loader_f, [q_watermark, time.time() - start]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--img-size", type=int, default=32, metavar='L',
                        help="size of generated image")
    
    parser.add_argument("-bs", "--batch-size", type=int, default=64, metavar='N',
                        help="input batch size for evaluation")

    parser.add_argument("-p", "--log-dir", default='', metavar="PATH",
                        help="path of log directory")

    parser.add_argument("--split", default=10, type=int, metavar='S',
                        help="split for Inception Score calculation")

    args = parser.parse_args()

    img_size = args.img_size
    bs       = args.batch_size
    split    = args.split
    log_dir  = args.log_dir
    wm_path  = os.path.join(log_dir, "watermark/watermark.png")

    data_loader = load_cifar10(
        img_size=img_size,
        batch_size=bs,
        num_workers=16,
    )

    watermark, wm_mask = create_watermark_mask(wm_path, img_size=img_size)

    watermark = watermark.cuda()
    wm_mask   = wm_mask.cuda()

    mask = list(map(int, open(os.path.join(log_dir, "watermark/mask.txt"), 'r').read().split()))

    os.makedirs(os.path.join(log_dir, "pruning"), exist_ok=True)
    log_file = open(os.path.join(log_dir, "pruning/history.csv"), 'w')

    z = torch.randn(64, 128).cuda()
    z[:, mask] = -10

    generator     = Generator().cuda()
    generator.load_state_dict(
        torch.load(os.path.join(log_dir, "checkpoint/gen_200.pth"))
    )

    discriminator = Discriminator().cuda()
    discriminator.load_state_dict(
        torch.load(os.path.join(log_dir, "checkpoint/dwm_200.pth"))
    )

    img = generator(z)
    img = ((img + 1) / 2).detach().cpu()
    save_image(img, os.path.join(log_dir, "pruning/00.png"), nrow=8)

    fs1, _ = forward_inception(data_loader, torch.device("cuda"))
    mu1, sig1 = np.mean(fs1, axis=0), np.cov(fs1, rowvar=False)

    for percent in range(1, 10):
        generator     = Generator().cuda()
        generator.load_state_dict(torch.load(os.path.join(log_dir, "checkpoint/gen_200.pth")))
        weight_prune(generator, percent * 10)

        img = generator(z)
        img = ((img + 1) / 2).detach().cpu()
        save_image(img, os.path.join(log_dir, f"pruning/{percent * 10}.png"), nrow=8)

        loader, stats = eval_watermark_quality(percent, mask, generator, discriminator, data_loader)

        start = time.time()

        # calculate IS
        fs2, ys2 = forward_inception(loader, torch.device("cuda"))
        is_mean, is_std = calc_entropy(ys2, len(loader.dataset), split)

        # calculate FID
        mu2, sig2 = np.mean(fs2, axis=0), np.cov(fs2, rowvar=False)

        fid = FID(mu1, sig1, mu2, sig2)

        stats[-1] += time.time() - start
        stats = [percent * 10, is_mean, is_std, fid] + stats

        print((
            f"Percentage: {stats[0]:d}% "
            f"IS: {stats[1]:.4f}±{stats[2]:.4f} "
            f"FID: {stats[3]:.4f} "
            f"Q: {stats[4]:.4f} "
            f"({stats[5]:.2f} s)"
        ))

        if percent == 1:
            log_file.write(','.join([
                "percent", "is_mean", "is_std", "fid", "quality", "time"
            ]) + '\n')

        log_file.write(','.join(list(map(str, stats))) + '\n')

    log_file.close()


if __name__ == '__main__':
    main()
