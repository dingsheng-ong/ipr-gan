import argparse
import json
import os
import time

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.models import vgg19
from torchvision.utils import save_image

from dataset.sr_dataset import dataset_list, loaders
from models.srgan import FeatureExtractor, SRResNet, Discriminator_96
from utils import (
    ApplyRandomPatch,
    ApplyWatermark,
    create_log_directory,
    set_bitmask,
    set_seed,
    ssim,
    sign_loss,
    StatsAccumulator,
)


def train(x, y, Ft, G, D, f_inp, f_out, opt_g, opt_d, device, S, coeff):

    bsz = x.size(0)

    x = x.to(device)
    y = y.to(device)

    gx  = G(x)

    # update Generator
    opt_g.zero_grad()

    lx = F.mse_loss(Ft(gx), Ft(y).detach())
    lg = F.binary_cross_entropy_with_logits(D(gx), torch.ones(bsz, device=device))
    ls = sign_loss(G.module) if S else torch.zeros(1).to(device)

    with torch.no_grad():
        x_wm = f_inp(x)
        y_wm = f_out(y)
    lw = 1 - ssim(G(x_wm, update_stats=False), y_wm)

    (lx + 1e-3 * lg + coeff * lw).backward(retain_graph=True)
    opt_g.step()

    # update discriminator
    opt_d.zero_grad()

    ldgx = F.binary_cross_entropy_with_logits(D(gx.detach()), torch.zeros(bsz, device=device))
    ldy  = F.binary_cross_entropy_with_logits(D(y), torch.ones(bsz, device=device))

    (ldgx + ldy).backward(retain_graph=True)

    opt_d.step()

    return lx.item(), lg.item(), ldgx.item(), ldy.item(), lw.item(), ls.item()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-wm', '--watermark', type=str, required=True, metavar='PATH',
                        help='path of watermark image')

    parser.add_argument('-p', '--path', type=str, required=True, metavar='PATH',
                        help='path of model directory')

    parser.add_argument('-bs', '--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for evaluation')
    
    parser.add_argument('-e', '--iter', type=int, default=int(1e5), metavar='N',
                        help='training iteration')

    parser.add_argument('-sf', '--sample-freq', type=int, default=int(1e4), metavar='N',
                        help='sample frequency')

    parser.add_argument('--num-workers', type=int, default=1, metavar='WORKERS',
                        help='num_workers for data loader')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    parser.add_argument('--load-discriminator', action='store_true', default=False,
                        help='load model weights')

    args = parser.parse_args()
    # set seed for reproducibility
    config    = json.load(open(os.path.join(args.path, 'config.json')))
    dataset   = config['dataset']
    sign_loss = config['sign_loss']
    watermark = config['watermark']
    seed      = config['seed']

    set_seed(seed)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    # get number of gpu(s)
    ngpu = torch.cuda.device_count() or 1

    # data loader
    loader = loaders[dataset](
        scale=config['factor'],
        img_size=config['img_size'],
        batch_size=config['batch_size'] * ngpu,
        num_workers=args.num_workers,
    )

    generator = SRResNet(upf=config['factor'])
    if sign_loss: set_bitmask(generator, string='HIDDEN SIGN WATERMARK (SRResNet)')
    generator.load_state_dict(
        torch.load(os.path.join(args.path, 'checkpoint/SRGAN.pt'))
    )
    generator = nn.DataParallel(generator).to(device)
    generator.train()
    
    discrmntr = Discriminator_96()
    if args.load_discriminator:
        discrmntr.load_state_dict(
            torch.load(os.path.join(args.path, 'checkpoint/Dis96.pt'))
        )
    discrmntr = nn.DataParallel(discrmntr).to(device)
    discrmntr.train()
    
    feature = nn.DataParallel(FeatureExtractor(vgg19(pretrained=True))).to(device)
    feature.eval()

    # optimizers
    opt_g = optim.Adam(generator.parameters(), lr=1e-5)
    opt_d = optim.Adam(discrmntr.parameters(), lr=1e-5)

    # create log directory
    os.makedirs(os.path.join(args.path, 'overwrite'), exist_ok=True)

    f_inp = ApplyRandomPatch(patch_size=config['img_size'] // 8, norm=False, device=device)
    f_inp.reset_mask()
    f_inp.save(os.path.join(args.path, 'overwrite/mask.png'))
    f_out = ApplyWatermark(args.watermark, wtmk_size=config['img_size'] // 2, norm=False, device=device)

    f_inp_old = ApplyRandomPatch(patch_size=config['img_size'] // 8, norm=False, device=device)
    f_inp_old.load(os.path.join(args.path, 'watermark/mask.png'))
    f_out_old = ApplyWatermark(watermark, wtmk_size=config['img_size'] // 2, norm=False, device=device)

    # log training progress
    log_file = open(os.path.join(args.path, 'overwrite/history.csv'), 'w')
    log_file.write(','.join([
        'loss_mse', 'loss_content', 'loss_adversarial', 'loss_dx', 'loss_dy', 'loss_s', 'loss_wm', 'time'
    ]) + '\n')
    log_file.flush()

    # helper functions for saving
    def sample_image(epoch, model, root):
        model.eval()
        root = os.path.join('dataset/data/', root)
        # get images in root
        images = []
        for r, ds, fs in os.walk(root): images += [os.path.join(r, f) for f in fs]
        # create sample directory
        sample_directory = os.path.join(args.path, f'overwrite/{epoch}')
        os.makedirs(sample_directory, exist_ok=False)
        # helper functions
        to_tensor = transforms.ToTensor()
        clamp     = lambda x: x.clamp_(0, 1)

        for image in images:
            
            hr = Image.open(image).convert('RGB')
            size = list(map(lambda x: x - x % config['factor'], hr.size))
            hr = transforms.CenterCrop(size[::-1])(hr)
            lr = hr.resize(map(lambda x: x // config['factor'], size), Image.BICUBIC)
            
            with torch.no_grad():
                hr = to_tensor(hr).unsqueeze(0)
                lr = to_tensor(lr).unsqueeze(0).to(device)
                
                if hr.size(1) == 1: hr = hr.repeat(1, 3, 1, 1)
                if lr.size(1) == 1: lr = lr.repeat(1, 3, 1, 1)

                lr_old = f_inp_old(lr)
                lr_new = f_inp(lr)
                sr_old = clamp(model(f_inp_old(lr)))
                sr_new = clamp(model(f_inp(lr)))
                hr_old = f_out_old(hr.to(device))
                hr_new = f_out(hr.to(device))
                
                lr_old, sr_old, hr_old = lr_old.cpu(), sr_old.cpu(), hr_old.cpu()
                lr_new, sr_new, hr_new = lr_new.cpu(), sr_new.cpu(), hr_new.cpu()

                lr_old = F.interpolate(lr_old, size=size[::-1], mode='bicubic', align_corners=False).cpu()
                lr_new = F.interpolate(lr_new, size=size[::-1], mode='bicubic', align_corners=False).cpu()
                images = torch.cat([lr_old, lr_new, sr_old, sr_new, hr_old, hr_new])
                save_image(images, os.path.join(sample_directory, os.path.basename(image)))

        model.train()

    def save_model(model, path):
        model.eval()
        torch.save(model.module.cpu().state_dict(), path)
        model.module.to(device)
        model.train()

    # training loop
    epoch = 1
    step  = 0

    Loss_content     = StatsAccumulator()
    Loss_adversarial = StatsAccumulator()
    Loss_Dy          = StatsAccumulator()
    Loss_DGx         = StatsAccumulator()
    Loss_WM          = StatsAccumulator()
    Loss_S           = StatsAccumulator()

    start = time.time()
    iter_loader = iter(loader)

    while step < args.iter:

        try:

            lr_img, hr_img = next(iter_loader)
            step += ngpu

            stats = train(
                lr_img,
                hr_img,
                feature,
                generator,
                discrmntr,
                f_inp,
                f_out,
                opt_g,
                opt_d,
                device,
                sign_loss,
                config['reg_coeff']
            )
            
            n = lr_img.size(0)
            Loss_content.update(stats[0], n)
            Loss_adversarial.update(stats[1], n)
            Loss_Dy.update(stats[2], n)
            Loss_DGx.update(stats[3], n)
            Loss_WM.update(stats[4], n)
            Loss_S.update(stats[5], n)

            print((
                f'Epoch {epoch}: [{Loss_Dy.count}/{len(loader.dataset)}] '
                f'C: {Loss_content.avg:.4f} '
                f'G: {Loss_adversarial.avg:.4f} '
                f'D[R/F]: [{Loss_Dy.avg:.4f} / {Loss_DGx.avg:.4f}] '
                f'S: {Loss_S.avg:.4f} '
                f'WM: {Loss_WM.avg:.4f} '
                f'({time.time() - start:.2f} s)'
            ), end='\r')

            if step == ngpu or step % args.sample_freq < ngpu:
                sample_image(
                    step,
                    generator,
                    'Set14'
                )
                save_model(generator, os.path.join(args.path, f'overwrite/SRGAN.pt'))
                save_model(discrmntr, os.path.join(args.path, f'overwrite/Dis96.pt'))

        except StopIteration:
            stats = [
                Loss_content.avg,
                Loss_adversarial.avg,
                Loss_Dy.avg,
                Loss_DGx.avg,
                Loss_S.avg,
                Loss_WM.avg,
                time.time() - start
            ]

            log_file.write(str.join(',', map(str, [float('nan')] + stats)) + '\n')
            log_file.flush()

            Loss_content.reset()
            Loss_adversarial.reset()
            Loss_Dy.reset()
            Loss_DGx.reset()
            Loss_S.reset()
            Loss_WM.reset()
            start = time.time()
            iter_loader = iter(loader)

            epoch += 1

            print()
    
    print()
    log_file.close()
    save_model(generator, os.path.join(args.path, f'overwrite/SRGAN.pt'))
    save_model(discrmntr, os.path.join(args.path, f'overwrite/Dis96.pt'))


if __name__ == '__main__':
    main()

