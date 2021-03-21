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
    set_bitmask,
    set_seed,
    sign_loss,
    ssim,
    ApplyWatermark,
    RandomBitMask,
    StatsAccumulator,
)


def train(epoch, loader, G, D, opt_G, opt_D, f_inp, f_out, device):

    Loss_Gz  = StatsAccumulator()
    Loss_Dx  = StatsAccumulator()
    Loss_DGz = StatsAccumulator()
    SignLoss = StatsAccumulator()
    RegLoss  = StatsAccumulator()

    start = time.time()
    N = len(loader.dataset)

    for i, (x, _) in enumerate(loader):

        bsz = x.size(0)
        z = torch.randn(bsz, 100).to(device)

        x  = x.to(device)
        gz = G(z)

        # update Discriminator
        opt_D.zero_grad()

        dx  = D(x)
        dgz = D(gz.detach())

        loss_dx  = F.relu(1.0 - dx).mean()
        loss_dgz = F.relu(1.0 + dgz).mean()

        sloss_d = sign_loss(D)

        (loss_dx + loss_dgz + sloss_d).backward()

        opt_D.step()

        # update Generator
        opt_G.zero_grad()

        loss_gz = - D(gz).mean()
        sloss_g = sign_loss(G)

        # regularization using dssim
        reg_loss = (1 - ssim(G(f_inp(z)), f_out(gz))) / 2

        (loss_gz + sloss_g + reg_loss).backward()

        opt_G.step()

        # update stats
        Loss_Gz.update(loss_gz.item(), bsz)
        Loss_Dx.update(loss_dx.item(), bsz)
        Loss_DGz.update(loss_dgz.item(), bsz)
        SignLoss.update((sloss_g.item() + sloss_d.item()), bsz)
        RegLoss.update(reg_loss.item(), bsz)

        print((
            f'Epoch {epoch}: [{Loss_Gz.count}/{N}] '
            f'G: {Loss_Gz.avg:.4f} '
            f'D[R/F]: [{Loss_Dx.avg:.4f} / {Loss_DGz.avg:.4f}] '
            f'S: {SignLoss.avg:.4f} '
            f'R: {RegLoss.avg:.4f} '
            f'({time.time() - start:.2f} s)'
        ), end='\r')

    print()
    return Loss_Gz.avg, Loss_Dx.avg, Loss_DGz.avg, SignLoss.avg, RegLoss.avg, time.time() - start


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True, metavar='PATH',
                        help='path of model directory')

    parser.add_argument('-bs', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for evaluation')
    
    parser.add_argument('-e', '--epoch', type=int, default=100, metavar='N',
                        help='training epoch')

    parser.add_argument('--fix-weight', action='store_true', default=False,
                        help='fix the weights of convolution')

    parser.add_argument('--load-discriminator', action='store_true', default=False,
                        help='load model weights')

    parser.add_argument('--num-workers', type=int, default=1, metavar='WORKERS',
                        help='num_workers for data loader')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    parser.add_argument('--seed', default=None, type=int, metavar='SEED',
                        help='set seed for reproducibility')

    args = parser.parse_args()

    # read config file
    config    = json.load(open(os.path.join(args.path, 'config.json')))
    dataset   = config['dataset']
    img_size  = config['img_size']
    watermark = config['watermark']

    # set seed for reproducibility
    set_seed(args.seed)

    # define device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')

    # data loader
    loader = loaders[dataset](
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # load generator
    generator_bef = Generator().to(device)
    set_bitmask(generator_bef)  # register buffer
    generator_bef.load_state_dict(torch.load(os.path.join(args.path, 'checkpoint/generator.pt')))

    generator_aft = Generator().to(device)
    set_bitmask(generator_aft)  # register buffer
    generator_aft.load_state_dict(torch.load(os.path.join(args.path, 'checkpoint/generator.pt')))
    set_bitmask(generator_aft)  # reset bitmask
    # fix the weight of convolution
    if args.fix_weight:
        for layer in generator_aft.modules():
            if isinstance(layer, torch.nn.ConvTranspose2d):
                layer.weight.requires_grad_(False)

    # load discriminator
    discrmntr = Discriminator().to(device)
    set_bitmask(discrmntr)
    if args.load_discriminator:
        discrmntr.load_state_dict(torch.load(os.path.join(args.path, 'checkpoint/discrmntr.pt')))

    # input transform functions
    f_inp = RandomBitMask(dim=100, nbit=10, const=-10, device=device)
    f_inp.load(os.path.join(args.path, 'watermark/mask.pt'))

    # output transform function
    f_out = ApplyWatermark(watermark, img_size=img_size, device=device)

    # optimizers
    opt_g = optim.Adam(
        generator_aft.parameters(),
        lr=0.0002,
        betas=[0.5, 0.999]
    )
    opt_d = optim.Adam(
        discrmntr.parameters(),
        lr=0.0002,
        betas=[0.5, 0.999]
    )

    # create attack directory
    directory = 'attack-weight-fixed' if args.fix_weight else 'attack-weight-free'
    os.makedirs(os.path.join(args.path, directory), exist_ok=True)

    # log attacking progress
    log_file = open(os.path.join(args.path, f'{directory}/history.csv'), 'w')

    # constant z for image sampling purpose
    z1 = torch.randn(32, 100).to(device)
    z2 = z1.clone()

    # functions for saving data
    def sample_img(z1, z2, model1, model2, path):
        model1.eval()
        model2.eval()
        img = torch.cat([model1(z1), model2(z2)]).cpu()
        save_image((img + 1) / 2, path, nrow=8)
        model1.train()
        model2.train()

    def save_model(model, path):
        torch.save(model.cpu().state_dict(), path)
        model.to(device)

    # training loop
    for epoch in range(1, args.epoch + 1):

        stats = train(epoch, loader, generator_aft, discrmntr, opt_g, opt_d, f_inp, f_out, device)

        if epoch == 1:
            log_file.write(','.join([
                'loss_gz', 'loss_dx', 'loss_dgz', 'sign', 'ssim', 'time'
            ]) + '\n')
        log_file.write(str.join(',', map(str, stats)) + '\n')
        log_file.flush()

        sample_img(z1, z2, generator_bef, generator_aft,
                   os.path.join(args.path, f'{directory}/{epoch:03d}.png'))

        save_model(generator_aft, os.path.join(args.path, f'{directory}/generator.pt'))
        save_model(discrmntr, os.path.join(args.path, f'{directory}/discrmntr.pt'))

    log_file.close()


if __name__ == '__main__':
    main()