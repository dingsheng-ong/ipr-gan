import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.utils import save_image

from dataset.loader import loaders
from models.dcgan import Generator, Discriminator
from utils import (
    create_log_directory,
    set_bitmask,
    set_seed,
    sign_loss,
    ssim,
    ApplyWatermark,
    RandomBitMask,
    StatsAccumulator,
)


def train(epoch, loader, G, D, opt_G, opt_D, reg_coeff, f_inp, f_out, device, log_directory):

    Loss_Gz  = StatsAccumulator()
    Loss_Dx  = StatsAccumulator()
    Loss_DGz = StatsAccumulator()
    SignLoss = StatsAccumulator()
    RegLoss  = StatsAccumulator()

    # log training progress
    log_file = open(os.path.join(log_directory, 'history.csv'), 'w')
    log_file.write(','.join([
        'loss_gz', 'loss_dx', 'loss_dgz', 'sign', 'ssim', 'time'
    ]) + '\n')
    log_file.flush()

    # constant z for image sampling purpose
    Z   = torch.randn(64, 128).to(device)
    Zwm = f_inp(Z)

    # functions for saving data
    def sample_img(z, model, path):
        model.eval()
        save_image((model(z).cpu() + 1) / 2, path, nrow=8)
        model.train()

    def save_model(model, path):
        torch.save(model.cpu().state_dict(), path)
        model.to(device)

    tick = time.time()

    iterator = iter(loader)
    for i in range(1, epoch + 1):

        try:
            x, _ = next(iterator)
        except:
            iterator = iter(loader)
            x, _ = next(iterator)

        bsz = x.size(0)
        z = torch.randn(bsz, 128).to(device)

        x  = x.to(device)
        gz = G(z)

        # update Discriminator
        opt_D.zero_grad()

        dx  = D(x)
        dgz = D(gz.detach())

        loss_dx  = F.relu(1.0 - dx).mean()
        loss_dgz = F.relu(1.0 + dgz).mean()
        sloss_d  = sign_loss(D)

        (loss_dx + loss_dgz + sloss_d).backward()

        opt_D.step()

        # update Generator
        opt_G.zero_grad()

        loss_gz = - D(gz).mean()
        sloss_g = sign_loss(G)
        reg_loss = (1 - ssim(G(f_inp(z), update_stats=False), f_out(gz)))

        (loss_gz + reg_coeff * reg_loss).backward()

        opt_G.step()

        # update stats
        Loss_Gz.update(loss_gz.item(), bsz)
        Loss_Dx.update(loss_dx.item(), bsz)
        Loss_DGz.update(loss_dgz.item(), bsz)
        SignLoss.update((sloss_g.item() + sloss_d.item()), bsz)
        RegLoss.update(reg_loss.item(), bsz)

        print((
            f'Iteration [{i:6d}/{epoch}] '
            f'G: {Loss_Gz.avg:.4f} '
            f'D[R/F]: [{Loss_Dx.avg:.4f} / {Loss_DGz.avg:.4f}] '
            f'S: {SignLoss.avg:.4f} '
            f'R: {RegLoss.avg:.4f} '
            f'({time.time() - tick:.2f} s)'
        ), end='\r')

        if i % 1000 == 0:
            print()
            stats = Loss_Gz.avg, Loss_Dx.avg, Loss_DGz.avg, SignLoss.avg, RegLoss.avg, time.time() - tick
            tick = time.time()
            log_file.write(str.join(',', map(str, stats)) + '\n')
            log_file.flush()
            sample_img(Z, G, os.path.join(log_directory, f'sample/{i // 1000:03d}.png'))
            sample_img(Zwm, G, os.path.join(log_directory, f'watermark/{i // 1000:03d}.png'))
            Loss_Gz.reset()
            Loss_Dx.reset()
            Loss_DGz.reset()
            SignLoss.reset()
            RegLoss.reset()
            save_model(G, os.path.join(log_directory, f'checkpoint/generator.pt'))
            save_model(D, os.path.join(log_directory, f'checkpoint/discrmntr.pt'))

    log_file.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-wm', '--watermark', type=str, required=True, metavar='PATH',
                        help='path of watermark image')

    parser.add_argument('-p', '--log-directory', type=str, default=None, metavar='PATH',
                        help='if specified, will save to assigned PATH')

    parser.add_argument('-d', '--dataset', type=str, default='cifar10', metavar='D',
                        choices=['celeb-a', 'cifar10', 'cub200', 'mnist'], help='training dataset')

    parser.add_argument('-s', '--img-size', type=int, default=32, metavar='L',
                        help='size of generated image')
    
    parser.add_argument('-bs', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for evaluation')
    
    parser.add_argument('-e', '--epoch', type=int, default=100000, metavar='N',
                        help='training epoch')

    parser.add_argument('-r', '--reg-coeff', type=float, default=1.0, metavar='N',
                        help='coefficient for watermark loss')

    parser.add_argument('--num-workers', type=int, default=1, metavar='WORKERS',
                        help='num_workers for data loader')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    parser.add_argument('--seed', default=None, type=int, metavar='SEED',
                        help='set seed for reproducibility')

    args = parser.parse_args()
    # set seed for reproducibility
    set_seed(args.seed)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')

    # data loader
    loader = loaders[args.dataset](
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generator = Generator().to(device)
    discrmntr = Discriminator().to(device)

    # set_bitmask(generator, string='EXAMPLE SIGNATURE')

    # optimizers
    opt_g = optim.Adam(
        generator.parameters(),
        lr=0.0001,
        betas=[0.5, 0.999]
    )
    opt_d = optim.Adam(
        discrmntr.parameters(),
        lr=0.0001,
        betas=[0.5, 0.999]
    )

    # input transform function
    f_inp = RandomBitMask(dim=128, nbit=10, const=-10, device=device)
    # output transform function
    f_out = ApplyWatermark(args.watermark, wtmk_size=args.img_size // 2, norm=True, device=device)

    # create log directory
    if not args.log_directory:
        log_directory = create_log_directory(os.path.abspath('logs_dcgan/'))
    else:
        log_directory = args.log_directory
        os.makedirs(log_directory, exist_ok=False)
        for sub_dir in ['checkpoint', 'sample', 'watermark']:
            os.makedirs(os.path.join(log_directory, sub_dir), exist_ok=False)

    # save config 
    json.dump(
        vars(args),
        open(os.path.join(log_directory, 'config.json'), 'w'),
        indent=4,
        sort_keys=True
    )

    # save meta data for transform functions
    f_inp.save(os.path.join(log_directory, 'watermark/mask.pt'))
    f_out.save(os.path.join(log_directory, 'watermark/watermark.png'))

    train(args.epoch, loader, generator, discrmntr, opt_g, opt_d, args.reg_coeff, f_inp, f_out, device, args.log_directory)

    def save_model(model, path):
        torch.save(model.cpu().state_dict(), path)
        model.to(device)

    save_model(generator, os.path.join(log_directory, f'checkpoint/generator.pt'))
    save_model(discrmntr, os.path.join(log_directory, f'checkpoint/discrmntr.pt'))


if __name__ == '__main__':
    main()

