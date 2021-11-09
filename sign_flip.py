from configs import Config

import argparse
import experiments
import numpy as np
import os
import random
import re
import tempfile
import torch

parser = argparse.ArgumentParser(description='IPR-GAN ambiguity attack script')

parser.add_argument('-l', '--log', required=True, type=str, metavar='PATH',
                                        help='Path to experiment log directory')

parser.add_argument('-s', '--sample', default=None, type=str, metavar='PATH',
                                        help='Save sample images to PATH/ if provided')

parser.add_argument('--cpu', action='store_true', default=False,
                                        help='Change device to CPU')

args = parser.parse_args()

def main(config):
    
    if not config.resource.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # prevent creating new tfboard log
    with tempfile.TemporaryDirectory() as tmp_dir:

        log = config.log.path
        os.makedirs(os.path.join(log, 'sign'), exist_ok=True)
        
        config.log.path = tmp_dir

        if config.get('sample_dir', None):
            base_sample_dir = config.sample_dir

        for percent in range(10, 101, 10):
            # load experiment state dict
            exp_state_dict = torch.load(
                os.path.join(log, 'checkpoint.pt'),
                map_location='cpu'
            )
            keys_g = list(filter(re.compile(r'G').match, exp_state_dict.keys()))
            # save evaluation metrics into JSON file
            eval_metrics_fpath = os.path.join(log, 'sign', f'{percent:02d}.json')
            if config.get('sample_dir', None):
                config.sample_dir = os.path.join(base_sample_dir, f'{percent:02d}')
                os.makedirs(config.sample_dir, exist_ok=True)

            config.attack_mode = f'SIGN-{percent}'
            Experiment = getattr(experiments, config.experiment)
            experiment = Experiment(config)
            experiment.load_state_dict(exp_state_dict, strict=True)

            for key in keys_g:
                nparams = 0
                for m in getattr(experiment.model, key).modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                        nparams += m.weight.numel()
                nflip = int(nparams * percent / 100)
                flip_index = torch.randperm(nparams)[:nflip]
                flip_mask = torch.ones(nparams)
                flip_mask[flip_index] *= -1
                for m in getattr(experiment.model, key).modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                        n = m.weight.numel()
                        mask = flip_mask[:n]
                        mask = mask.to(m.weight)
                        with torch.no_grad():
                            m.weight.data.mul_(mask)
                        flip_mask = flip_mask[n:]

            experiment.evaluate(eval_metrics_fpath)

if __name__ == '__main__':

    config_fpath = os.path.join(args.log, 'config.yaml')
    assert os.path.exists(config_fpath), f'Invalid experiment log: {args.log}'

    config = Config.parse(config_fpath)
    config.resource.gpu = not args.cpu

    if args.sample:
        config.sample_dir = os.path.join(args.sample, os.path.basename(config.log.path) + '-SIGN')
        os.makedirs(args.sample, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)
    random.seed(config.seed)

    main(config)