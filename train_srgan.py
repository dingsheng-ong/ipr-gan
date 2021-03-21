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

    (lx + 1e-3 * lg + ls + coeff * lw).backward(retain_graph=True)
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

    parser.add_argument('-d', '--dataset', type=str, default='ILSVR2016-subset', metavar='D',
                        choices=dataset_list, help='training dataset')

    parser.add_argument('-p', '--pretrained-SRResNet', type=str, default=None, metavar='PATH',
                        help='path to pretrained SRResNet, if provided, pretrain will not execute')

    parser.add_argument('-s', '--img-size', type=int, default=96, metavar='L',
                        help='size of high resolution image')
    
    parser.add_argument('-f', '--factor', type=int, default=4, metavar='F',
                        help='down sample factor')

    parser.add_argument('-r', '--reg-coeff', type=float, default=1.0, metavar='N',
                        help='coefficient for watermark loss')

    parser.add_argument('-bs', '--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for evaluation')
    
    parser.add_argument('-pe', '--pretrain-iteration', type=int, default=int(1e6), metavar='N',
                        help='pre-training iteration')

    parser.add_argument('-e', '--iter', type=int, default=int(2e5), metavar='N',
                        help='training iteration')

    parser.add_argument('-sf', '--sample-freq', type=int, default=int(1e4), metavar='N',
                        help='sample frequency')

    parser.add_argument('--num-workers', type=int, default=1, metavar='WORKERS',
                        help='num_workers for data loader')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    parser.add_argument('--sign-loss', default=False, action='store_true',
                        help='embed sign into BN weight')

    parser.add_argument('--seed', default=None, type=int, metavar='SEED',
                        help='set seed for reproducibility')

    args = parser.parse_args()
    # set seed for reproducibility
    set_seed(args.seed)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    # get number of gpu(s)
    ngpu = torch.cuda.device_count() or 1

    # data loader
    loader = loaders[args.dataset](
        scale=args.factor,
        img_size=args.img_size,
        batch_size=args.batch_size * ngpu,
        num_workers=args.num_workers,
    )

    generator = SRResNet(upf=args.factor)
    if args.sign_loss: set_bitmask(generator, string='HIDDEN SIGN WATERMARK (SRResNet)')
    if args.pretrained_SRResNet:
        generator.load_state_dict(
            torch.load(os.path.join(args.pretrained_SRResNet, 'checkpoint/SRResNet.pt'))
        )
    generator = nn.DataParallel(generator).to(device)
    generator.train()
    
    discrmntr = nn.DataParallel(Discriminator_96()).to(device)
    discrmntr.train()
    
    feature = nn.DataParallel(FeatureExtractor(vgg19(pretrained=True))).to(device)
    feature.eval()

    # optimizers
    opt_g = optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = optim.Adam(discrmntr.parameters(), lr=1e-4)

    # create log directory
    log_directory = create_log_directory(os.path.abspath('logs_srgan/'))

    f_inp = ApplyRandomPatch(patch_size=args.img_size // 8, norm=False, device=device)
    f_out = ApplyWatermark(args.watermark, wtmk_size=args.img_size // 2, norm=False, device=device)
    f_inp.save(os.path.join(log_directory, 'watermark/mask.png'))
    f_out.save(os.path.join(log_directory, 'watermark/watermark.png'))

    # save config 
    json.dump(
        vars(args),
        open(os.path.join(log_directory, 'config.json'), 'w'),
        indent=4,
        sort_keys=True
    )

    # log training progress
    log_file = open(os.path.join(log_directory, 'history.csv'), 'w')
    log_file.write(','.join([
        'loss_mse', 'loss_content', 'loss_adversarial', 'loss_dx', 'loss_dy', 'loss_s', 'loss_wm', 'time'
    ]) + '\n')
    log_file.flush()

    # helper functions for saving
    def sample_image(epoch, model, root, watermark=False):
        model.eval()
        root = os.path.join('dataset/data/', root)
        # get images in root
        images = []
        for r, ds, fs in os.walk(root): images += [os.path.join(r, f) for f in fs]
        # create sample directory
        if watermark:
            sample_directory = os.path.join(log_directory, f'watermark/{epoch}')
        else:
            sample_directory = os.path.join(log_directory, f'sample/{epoch}')
        os.makedirs(sample_directory, exist_ok=False)
        # helper functions
        to_tensor = transforms.ToTensor()
        clamp     = lambda x: x.clamp_(0, 1)

        for image in images:
            
            hr = Image.open(image).convert('RGB')
            size = list(map(lambda x: x - x % args.factor, hr.size))
            hr = transforms.CenterCrop(size[::-1])(hr)
            lr = hr.resize(map(lambda x: x // args.factor, size), Image.BICUBIC)
            
            with torch.no_grad():
                hr = to_tensor(hr).unsqueeze(0)
                lr = to_tensor(lr).unsqueeze(0).to(device)
                
                if hr.size(1) == 1: hr = hr.repeat(1, 3, 1, 1)
                if lr.size(1) == 1: lr = lr.repeat(1, 3, 1, 1)

                if watermark:
                    lr = f_inp(lr)
                    sr = clamp(model(lr))
                    hr = f_out(hr.to(device))
                    lr, sr, hr = lr.cpu(), sr.cpu(), hr.cpu()
                else:
                    sr = clamp(model(lr)).cpu()

                lr = F.interpolate(lr, size=size[::-1], mode='bicubic', align_corners=False).cpu()
                images = torch.cat([lr, sr, hr])
                save_image(images, os.path.join(sample_directory, os.path.basename(image)))

        model.train()

    def save_model(model, path):
        model.eval()
        torch.save(model.module.cpu().state_dict(), path)
        model.module.to(device)
        model.train()

    # pretrained loop
    epoch = 1
    step  = 0
    
    Loss_MSE = StatsAccumulator()
    Loss_S   = StatsAccumulator()
    Loss_WM  = StatsAccumulator()
    start = time.time()
    iter_loader = iter(loader)

    while not args.pretrained_SRResNet and step < args.pretrain_iteration:
        try:
            lr_img, hr_img = next(iter_loader)

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            
            opt_g.zero_grad()
            
            sr_img = generator(lr_img)
            loss_mse = F.mse_loss(sr_img, hr_img)

            loss_sign = sign_loss(generator.module) if args.sign_loss else torch.zeros(1).to(device)

            with torch.no_grad():
                lr_wm = f_inp(lr_img)
                hr_wm = f_out(hr_img)
            sr_wm = generator(lr_wm, update_stats=False)
            loss_wm = 1 - ssim(sr_wm, hr_wm)

            (loss_mse + loss_sign + args.reg_coeff * loss_wm).backward(retain_graph=True)

            Loss_MSE.update(loss_mse.item(), lr_img.size(0))
            Loss_S.update(loss_sign.item(), lr_img.size(0))
            Loss_WM.update(loss_wm.item(), lr_img.size(0))

            opt_g.step()
            step += ngpu

            print((
                f'Epoch {epoch}: [{Loss_MSE.count}/{len(loader.dataset)}] '
                f'MSE: {Loss_MSE.avg:.4f} '
                f'S: {Loss_S.avg:.4f} '
                f'WM: {Loss_WM.avg:.4f} '
                f'({time.time() - start:.2f} s)'
            ), end='\r')

            if step % args.sample_freq < ngpu:
                sample_image(step // args.sample_freq, generator, 'Set14')
                sample_image(step // args.sample_freq, generator, 'Set14', watermark=True)
                save_model(generator, os.path.join(log_directory, f'checkpoint/SRResNet.pt'))

        except StopIteration:
            epoch += 1
            iter_loader = iter(loader)
            log_file.write(','.join([
                str(Loss_MSE.avg),
                *(['nan'] * 4),
                str(Loss_S.avg),
                str(Loss_WM.avg),
                str(time.time() - start)
            ]) + '\n')
            log_file.flush()
            Loss_MSE.reset()
            Loss_S.reset()
            Loss_WM.reset()
            start = time.time()
            print()
    print()

    save_model(generator, os.path.join(log_directory, f'checkpoint/SRResNet.pt'))

    # training loop
    epoch = 1
    step  = 0

    Loss_content     = StatsAccumulator()
    Loss_adversarial = StatsAccumulator()
    Loss_Dy          = StatsAccumulator()
    Loss_DGx         = StatsAccumulator()
    Loss_WM.reset()
    Loss_S.reset()

    start = time.time()
    iter_loader = iter(loader)

    while step < args.iter:

        try:

            lr_img, hr_img = next(iter_loader)
            step += ngpu

            if step == args.iter // 2:
                opt_g.param_groups[0]['lr'] *= 0.1
                opt_d.param_groups[0]['lr'] *= 0.1

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
                args.sign_loss,
                0 if step < args.iter // 2 else args.reg_coeff
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
                    args.pretrain_iteration // args.sample_freq + step // args.sample_freq,
                    generator,
                    'Set14'
                )
                sample_image(
                    args.pretrain_iteration // args.sample_freq + step // args.sample_freq,
                    generator,
                    'Set14',
                    watermark=True
                )
                save_model(generator, os.path.join(log_directory, f'checkpoint/SRGAN.pt'))
                save_model(discrmntr, os.path.join(log_directory, f'checkpoint/Dis96.pt'))

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
    save_model(generator, os.path.join(log_directory, f'checkpoint/SRGAN.pt'))
    save_model(discrmntr, os.path.join(log_directory, f'checkpoint/Dis96.pt'))


if __name__ == '__main__':
    main()

