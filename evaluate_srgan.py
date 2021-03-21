import argparse
import json
import math
import numpy as np
import os

from PIL import Image
from skimage.measure import compare_ssim

import torch
import torch.nn.functional as F

from torchvision import transforms

from dataset.sr_dataset import dataset_list
from models.srgan import SRResNet
from metrics import psnr as calc_psnr
from utils import (
    ApplyRandomPatch,
    ApplyWatermark,
    get_bitmask,
    set_bitmask,
    set_seed,
    ssim as calc_ssim,
    StatsAccumulator,
    tensor_to_numpy_image,
    rgb_to_luma,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True, metavar='PATH',
                        help='path of model directory')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    args = parser.parse_args()

    # read config file
    config    = json.load(open(os.path.join(args.path, 'config.json')))
    watermark = config.get('watermark', None)
    factor    = config['factor']
    seed      = config['seed']

    # set seed for reproducibility
    set_seed(seed)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')

    generator = SRResNet(16, 4).to(device)
    finetune  = SRResNet(16, 4).to(device)
    overwrite = SRResNet(16, 4).to(device)
    if not 'baseline' in args.path.split('/')[0]: set_bitmask(generator)
    if not 'baseline' in args.path.split('/')[0]: set_bitmask(finetune)
    if not 'baseline' in args.path.split('/')[0]: set_bitmask(overwrite)
    generator.load_state_dict(
        torch.load(os.path.join(args.path, f'checkpoint/SRGAN.pt'))
    )
    finetune.load_state_dict(
        torch.load(os.path.join(args.path, f'finetune/SRGAN.pt'))
    )
    overwrite.load_state_dict(
        torch.load(os.path.join(args.path, f'overwrite/SRGAN.pt'))
    )
    generator.eval()
    finetune.eval()
    overwrite.eval()

    bm, ws = get_bitmask(finetune)
    finetune_sign_loss = np.sum(list(map(lambda x: x[0] != x[1], zip(bm, ws))))
    bm, ws = get_bitmask(overwrite)
    overwrite_sign_loss = np.sum(list(map(lambda x: x[0] != x[1], zip(bm, ws))))

    # helper functions
    to_tensor = transforms.ToTensor()

    f_inp = ApplyRandomPatch(patch_size=config['img_size'] // 8, norm=False, device=device)
    f_inp.load(os.path.join(args.path, 'watermark/mask.png'))

    f_out = ApplyWatermark(os.path.join(args.path, 'watermark/watermark.png'), wtmk_size=config['img_size'] // 2, norm=False, device=device)

    for dataset in ['Set5', 'Set14', 'BSDS100']:
        PSNR    = StatsAccumulator()
        PSNR_FT = StatsAccumulator()
        PSNR_OV = StatsAccumulator()
        SSIM    = StatsAccumulator()
        SSIM_FT = StatsAccumulator()
        SSIM_OV = StatsAccumulator()
        WM      = StatsAccumulator()
        WM_FT   = StatsAccumulator()
        WM_OV   = StatsAccumulator()
        
        eval_images = []
        for r, ds, fs in os.walk(os.path.join('dataset/data/', dataset)):
            eval_images += [os.path.join(r, f) for f in fs]

        for img in eval_images:

            hr = Image.open(img).convert('RGB')
            size = list(map(lambda x: x - x % factor, hr.size))
            hr = transforms.CenterCrop(size[::-1])(hr)
            lr = hr.resize(map(lambda x: x // factor, size), Image.BICUBIC)

            with torch.no_grad():
                hr = to_tensor(hr).unsqueeze(0).to(device)
                lr = to_tensor(lr).unsqueeze(0).to(device)

                if hr.size(1) == 1: hr.repeat(1, 3, 1, 1)
                if lr.size(1) == 1: lr.repeat(1, 3, 1, 1)

                sr = generator(lr)
                sr_ft = finetune(lr)
                sr_ov = overwrite(lr)
                wm = generator(f_inp(lr))
                wm_ft = finetune(f_inp(lr))
                wm_ov = overwrite(f_inp(lr))
                hr_wm = f_out(hr)

                hr = rgb_to_luma(tensor_to_numpy_image(hr))
                sr = rgb_to_luma(tensor_to_numpy_image(sr))
                sr_ft = rgb_to_luma(tensor_to_numpy_image(sr_ft))
                sr_ov = rgb_to_luma(tensor_to_numpy_image(sr_ov))

                hr = hr[4:-4, 4:-4]
                sr = sr[4:-4, 4:-4]
                sr_ft = sr_ft[4:-4, 4:-4]
                sr_ov = sr_ov[4:-4, 4:-4]

                x = config['img_size']
                psnr = calc_psnr(sr, hr)
                psnr_ft = calc_psnr(sr_ft, hr)
                psnr_ov = calc_psnr(sr_ov, hr)
                
                ssim = compare_ssim(sr, hr)
                ssim_ft = compare_ssim(sr_ft, hr)
                ssim_ov = compare_ssim(sr_ov, hr)

                wm    = calc_ssim(wm[..., :x, :x], hr_wm[..., :x, :x])
                wm_ft = calc_ssim(wm_ft[..., :x, :x], hr_wm[..., :x, :x])
                wm_ov = calc_ssim(wm_ov[..., :x, :x], hr_wm[..., :x, :x])

            PSNR.update(psnr)
            PSNR_FT.update(psnr_ft)
            PSNR_OV.update(psnr_ov)
            SSIM.update(ssim)
            SSIM_FT.update(ssim_ft)
            SSIM_OV.update(ssim_ov)
            WM.update(wm)
            WM_FT.update(wm_ft)
            WM_OV.update(wm_ov)

        output_file = os.path.join(args.path, f'{dataset}.json')
        data = {
            'dataset': dataset,
            'PSNR'   : float(PSNR.avg),
            'SSIM'   : float(SSIM.avg),
            'WM'     : float(WM.avg),
            'PSNR_FT': float(PSNR_FT.avg),
            'SSIM_FT': float(SSIM_FT.avg),
            'FT'     : float(WM_FT.avg),
            'FT_HD'  : float(finetune_sign_loss),
            'PSNR_OV': float(PSNR_OV.avg),
            'SSIM_OV': float(SSIM_OV.avg),
            'OV'     : float(WM_OV.avg),
            'OV_HD'  : float(overwrite_sign_loss),
        }

        json.dump(data, open(output_file, 'w'), indent=4, sort_keys=True)

        print(data)


if __name__ == '__main__':
    main()
