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
    set_seed,
    StatsAccumulator,
)


def train(epoch, loader, G, D, opt_G, opt_D, device, log_directory):

    Loss_Gz  = StatsAccumulator()
    Loss_Dx  = StatsAccumulator()
    Loss_DGz = StatsAccumulator()

    # log training progress
    log_file = open(os.path.join(log_directory, 'history.csv'), 'w')
    log_file.write(','.join([
        'loss_gz', 'loss_dx', 'loss_dgz', 'time'
    ]) + '\n')
    log_file.flush()

    # constant z for image sampling purpose
    Z = torch.randn(64, 128).to(device)

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

        (loss_dx + loss_dgz).backward()

        opt_D.step()

        # update Generator
        opt_G.zero_grad()

        loss_gz = - D(gz).mean()

        loss_gz.backward()

        opt_G.step()

        # update stats
        Loss_Gz.update(loss_gz.item(), bsz)
        Loss_Dx.update(loss_dx.item(), bsz)
        Loss_DGz.update(loss_dgz.item(), bsz)

        print((
            f'Iteration [{i:6d}/{epoch}] '
            f'G: {Loss_Gz.avg:.4f} '
            f'D[R/F]: [{Loss_Dx.avg:.4f} / {Loss_DGz.avg:.4f}] '
            f'({time.time() - tick:.2f} s)'
        ), end='\r')

        if i % 1000 == 0:
            print()
            stats = Loss_Gz.avg, Loss_Dx.avg, Loss_DGz.avg, time.time() - tick
            tick = time.time()
            log_file.write(str.join(',', map(str, stats)) + '\n')
            sample_img(Z, G, os.path.join(log_directory, f'sample/{i // 1000:03d}.png'))
            Loss_Gz.reset()
            Loss_Dx.reset()
            Loss_DGz.reset()
            save_model(G, os.path.join(log_directory, f'checkpoint/generator.pt'))
            save_model(D, os.path.join(log_directory, f'checkpoint/discrmntr.pt'))

    log_file.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='cifar10', metavar='D',
                        choices=['celeb-a', 'cifar10', 'cub200', 'mnist'], help='training dataset')

    parser.add_argument('-s', '--img-size', type=int, default=32, metavar='L',
                        help='size of generated image')
    
    parser.add_argument('-bs', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for evaluation')
    
    parser.add_argument('-e', '--epoch', type=int, default=100000, metavar='N',
                        help='training epoch')

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

    # create log directory
    log_directory = create_log_directory(os.path.abspath('baseline_dcgan/'))
    os.system(f'rm -r {os.path.join(log_directory, "watermark")}')

    # save config 
    json.dump(
        vars(args),
        open(os.path.join(log_directory, 'config.json'), 'w'),
        indent=4,
        sort_keys=True
    )

    train(args.epoch, loader, generator, discrmntr, opt_g, opt_d, device, log_directory)

    def save_model(model, path):
        torch.save(model.cpu().state_dict(), path)
        model.to(device)

    save_model(generator, os.path.join(log_directory, f'checkpoint/generator.pt'))
    save_model(discrmntr, os.path.join(log_directory, f'checkpoint/discrmntr.pt'))


if __name__ == '__main__':
    main()

