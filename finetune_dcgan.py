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


def train(epoch, loader, G, D, opt_G, opt_D, f_inp, f_out, device, log_directory):

    Loss_Gz  = StatsAccumulator()
    Loss_Dx  = StatsAccumulator()
    Loss_DGz = StatsAccumulator()
    SignLoss = StatsAccumulator()
    RegLoss  = StatsAccumulator()

    # log training progress
    log_file = open(os.path.join(log_directory, 'finetune/history.csv'), 'w')
    log_file.write(','.join([
        'loss_gz', 'loss_dx', 'loss_dgz', 'sign', 'ssim', 'time'
    ]) + '\n')
    log_file.flush()

    # constant z for image sampling purpose
    z1 = torch.randn(32, 128).to(device)
    z2 = f_inp(z1.clone())
    Z  = torch.cat([z1, z2])

    # functions for saving data
    def sample_img(z, model, path):
        model.eval()
        save_image((model(z).cpu() + 1) / 2, path, nrow=8)
        model.train()

    def save_model(model, path):
        torch.save(model.cpu().state_dict(), path)
        model.to(device)

    tick = time.time()

    sample_img(Z, G, os.path.join(log_directory, f'finetune/0.png'))
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

        (loss_dx + loss_dgz).backward()

        opt_D.step()

        # update Generator
        opt_G.zero_grad()

        loss_gz = - D(gz).mean()
        sloss_g = sign_loss(G)
        reg_loss = (1 - ssim(G(f_inp(z), update_stats=False), f_out(gz)))

        loss_gz.backward()

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
            sample_img(Z, G, os.path.join(log_directory, f'finetune/{i // 1000:03d}.png'))
            Loss_Gz.reset()
            Loss_Dx.reset()
            Loss_DGz.reset()
            SignLoss.reset()
            RegLoss.reset()
            save_model(G, os.path.join(log_directory, f'finetune/generator.pt'))
            save_model(D, os.path.join(log_directory, f'finetune/discrmntr.pt'))

    log_file.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True, metavar='PATH',
                        help='path of model directory')

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001, metavar='LR',
                        help='learning rate for fine-tuning')

    parser.add_argument('-bs', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for evaluation')
    
    parser.add_argument('-e', '--epoch', type=int, default=50000, metavar='N',
                        help='training epoch')

    parser.add_argument('--load-discriminator', action='store_true', default=False,
                        help='load model weights')

    parser.add_argument('--num-workers', type=int, default=1, metavar='WORKERS',
                        help='num_workers for data loader')

    parser.add_argument('--cpu', default=False, action='store_true',
                        help='force to use CPU even if CUDA exists')

    args = parser.parse_args()

    # read config file
    config    = json.load(open(os.path.join(args.path, 'config.json')))
    dataset   = config['dataset']
    img_size  = config['img_size']
    watermark = config['watermark']
    seed      = config['seed']

    # set seed for reproducibility
    set_seed(seed)

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
    generator = Generator().to(device)
    set_bitmask(generator)
    generator.load_state_dict(torch.load(os.path.join(args.path, 'checkpoint/generator.pt')))

    # load discriminator
    discrmntr = Discriminator().to(device)
    if args.load_discriminator:
        set_bitmask(discrmntr)
        discrmntr.load_state_dict(torch.load(os.path.join(args.path, 'checkpoint/discrmntr.pt')))

    # optimizers
    opt_g = optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=[0.5, 0.999]
    )
    opt_d = optim.Adam(
        discrmntr.parameters(),
        lr=args.learning_rate,
        betas=[0.5, 0.999]
    )

    f_inp = RandomBitMask(dim=128, nbit=10, const=-10, device=device)
    f_inp.load(os.path.join(args.path, 'watermark/mask.pt'))

    f_out = ApplyWatermark(watermark, wtmk_size=img_size // 2, norm=True, device=device)

    os.makedirs(os.path.join(args.path, 'finetune'), exist_ok=True)

    train(args.epoch, loader, generator, discrmntr, opt_g, opt_d, f_inp, f_out, device, args.path)

    def save_model(model, path):
        torch.save(model.cpu().state_dict(), path)
        model.to(device)

    save_model(generator, os.path.join(args.path, f'finetune/generator.pt'))
    save_model(discrmntr, os.path.join(args.path, f'finetune/discrmntr.pt'))


if __name__ == '__main__':
    main()

