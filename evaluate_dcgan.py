import argparse
import json
import numpy as np
import os

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from dataset.loader import loaders
from models.dcgan import Generator
from metrics import forward_inception, calc_entropy, FID
from utils import (
    get_bitmask,
    set_seed,
    set_bitmask,
    ApplyWatermark,
    RandomBitMask,
    ssim,
)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item], 0

    def __len__(self):
        return self.data.size(0)


def get_fid_is(real_loader, fake_loader, device, splits=10):

    fs1, _   = forward_inception(real_loader, device)
    fs2, ys2 = forward_inception(fake_loader, device)

    is_mean, is_std = calc_entropy(ys2, len(fake_loader.dataset), splits)

    mu1, sig1 = np.mean(fs1, axis=0), np.cov(fs1, rowvar=False)
    mu2, sig2 = np.mean(fs2, axis=0), np.cov(fs2, rowvar=False)

    fid = FID(mu1, sig1, mu2, sig2)

    return fid, is_mean, is_std


def evaluate(path, args):
    # read config
    config    = json.load(open(os.path.join(args.path, 'config.json')))
    dataset   = config['dataset']
    img_size  = config['img_size']
    watermark = config.get('watermark', None)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')

    # data loader
    real_loader = loaders[dataset](
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n = len(real_loader.dataset)

    # load model
    generator = Generator().to(device)
    if not 'baseline' in args.path.split('/')[0]: set_bitmask(generator)
    generator.load_state_dict(torch.load(path))
    generator.eval()

    fake_imgs = []
    real_imgs = []
    for (img, _) in tqdm(real_loader, desc="Generating images", leave=False):
        bsz = img.size(0)
        with torch.no_grad():
            z = torch.randn(bsz, 128).to(device)
            fake_imgs.append(generator(z).cpu())
            img = img.to(device)
            real_imgs.append(img.cpu())

    fake_imgs = torch.cat(fake_imgs, dim=0)
    real_imgs = torch.cat(real_imgs, dim=0)

    fake_loader = DataLoader(
        CustomDataset(fake_imgs),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    del real_loader
    real_loader = DataLoader(
        CustomDataset(real_imgs),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    fid, is_mean, is_std = get_fid_is(real_loader, fake_loader, device)
    
    return [fid, is_mean, is_std, *calculate_ssim(path, args)] if args.eval_watermark else [fid, is_mean, is_std]


def calculate_ssim(path, args):
    # read config
    config    = json.load(open(os.path.join(args.path, 'config.json')))
    dataset   = config['dataset']
    img_size  = config['img_size']
    watermark = config.get('watermark', None)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')

    # data loader
    real_loader = loaders[dataset](
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n = len(real_loader.dataset)

    # load model
    generator = Generator().to(device)
    if not 'baseline' in args.path.split('/')[0]: set_bitmask(generator)
    generator.load_state_dict(torch.load(path))
    generator.eval()

    # load input and output transform function if evaluate watermark
    f_inp = RandomBitMask(dim=128, nbit=10, const=-10, device=device)
    f_inp.load(os.path.join(args.path, 'watermark/mask.pt'))
    f_out = ApplyWatermark(watermark, wtmk_size=img_size // 2, norm=True, device=device)

    scores = []
    for (img, _) in tqdm(real_loader, desc="Generating images", leave=False):
        bsz = img.size(0)
        with torch.no_grad():
            z = torch.randn(bsz, 128).to(device)
            scores.extend(ssim(generator(f_inp(z))[..., :16, :16], f_out(generator(z))[..., :16, :16], size_average=False).tolist())
    return np.mean(scores), np.std(scores)


def compute_sign_deviation(model):
    generator = Generator()
    set_bitmask(generator)
    generator.load_state_dict(torch.load(model))
    generator.eval()

    bm, ws = get_bitmask(generator)
    return np.mean(list(map(lambda x: x[0] != x[1], zip(bm, ws))))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True, metavar='PATH',
                        help='path of model directory')

    parser.add_argument('-bs', '--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for evaluation')

    parser.add_argument('--eval-watermark', default=False, action='store_true',
                        help='evaluate watermarked images')

    parser.add_argument('--eval-sign', default=False, action='store_true',
                        help='evaluate sign loss')

    parser.add_argument('--num-workers', type=int, default=1, metavar='WORKERS',
                        help='num_workers for data loader')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    args = parser.parse_args()

    # read config file
    config = json.load(open(os.path.join(args.path, 'config.json')))
    seed   = config['seed']

    # set seed for reproducibility
    set_seed(seed)

    if args.eval_watermark:
        r   = evaluate(os.path.join(args.path, f'checkpoint/generator.pt'), args)
        f   = evaluate(os.path.join(args.path, f'finetune/generator.pt'), args)
        o   = evaluate(os.path.join(args.path, f'overwrite/generator.pt'), args)

        data = {
            'fid'    : float(r[0]),
            'is_mean': float(r[1]),
            'is_std' : float(r[2]),
            'ssim_mean'    : float(r[3]),
            'ssim_std'     : float(r[4]),
            'ft_fid'    : float(f[0]),
            'ft_is_mean': float(f[1]),
            'ft_is_std' : float(f[2]),
            'ft_ssim_mean'    : float(f[3]),
            'ft_ssim_std'     : float(f[4]),
            'ov_fid'    : float(o[0]),
            'ov_is_mean': float(o[1]),
            'ov_is_std' : float(o[2]),
            'ov_ssim_mean'    : float(o[3]),
            'ov_ssim_std'     : float(o[4]),
        }
        if args.eval_sign:
            percent_deviate_ft  = compute_sign_deviation(os.path.join(args.path, 'finetune/generator.pt'))
            percent_deviate_ovr = compute_sign_deviation(os.path.join(args.path, 'overwrite/generator.pt')) 
            data['percent_deviate_ft'] = percent_deviate_ft
            data['percent_deviate_ovr'] = percent_deviate_ovr

        output_file = os.path.join(args.path, 'metrics.json')
        json.dump(data, open(output_file, 'w'), indent=4, sort_keys=True)

        print(json.load(open(output_file)))
    else:
        fid, is_mean, is_std = evaluate(os.path.join(args.path, 'checkpoint/generator.pt'), args)
        output_file = os.path.join(args.path, 'metrics_baseline.json')
        json.dump({
            'fid': float(fid),
            'is_mean': float(is_mean),
            'is_std': float(is_std),
        }, open(output_file, 'w'), indent=4, sort_keys=True)

        print(f'FID: {fid:.4f}')
        print(f'Inception Score: {is_mean:.4f}±{is_std:.4f}')


if __name__ == '__main__':
    main()

