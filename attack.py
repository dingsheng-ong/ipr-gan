from collections import OrderedDict
from configs import Config

import argparse
import copy
import experiments
import numpy as np
import os
import random
import re
import tempfile
import torch

parser = argparse.ArgumentParser(description='IPR-GAN evaluation script')

parser.add_argument('-l', '--log', required=True, type=str, metavar='PATH',
                                        help='Path to experiment log directory')

parser.add_argument('-m', '--mode', required=True, type=str, metavar='MODE',
                                        choices=['finetune', 'overwrite'],
                                        help='Attack mode, choices: [finetune, overwrite]')

parser.add_argument('-w', '--watermark', type=str, metavar='PATH',
                                        help='Path to new watermark, used in overwriting')

parser.add_argument('-d', '--load-discriminator',
                                        action='store_true', default=False,
                                        help='Whether to load discriminator\'s weight')

args = parser.parse_args()

def main(config):
    
    if not config.resource.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    Experiment = getattr(experiments, config.experiment)

    # TO-DO: choose update base on mode
    alt_config = {
        'finetune': update_finetune_config,
        'overwrite': update_overwrite_config
    }[args.mode](config)
    
    # load experiment state dict
    exp_state_dict = torch.load(
        os.path.join(config.log.path, 'checkpoint.pt'),
        map_location='cpu'
    )
    keys_g = list(filter(re.compile(r'G').match, exp_state_dict.keys()))
    keys_d = list(filter(re.compile(r'D').match, exp_state_dict.keys()))

    # create attack experiment
    attack_experiment = Experiment(alt_config)
    # load generator's weight
    state_dict = OrderedDict(step=0)
    for key in keys_g: state_dict[key] = exp_state_dict[key]
    # load discriminator weight
    if args.load_discriminator:
        for key in keys_d:
            state_dict[key] = exp_state_dict[key]
            
    attack_experiment.load_state_dict(state_dict, strict=False)

    # reset mask for overwriting experiment
    if args.mode == 'overwrite':
        attack_experiment.model.fn_inp.module.reset()
    # start attack experiment
    attack_experiment.start()

    for k, v in attack_experiment.model.state_dict().items():
        if k.startswith('fn_'):
            k = k + '_ov'
        exp_state_dict[k] = v
    
    # save old config to the log
    config_path = os.path.join(alt_config.log.path, 'config.yaml')
    with open(config_path, 'w') as f:
        _log = config.log.path
        # change log path to new log
        config.log.path = alt_config.log.path
        f.write(config.to_yaml())

    # prevent creating new tfboard log
    with tempfile.TemporaryDirectory() as tmp_dir:

        log = config.log.path
        config.log.path = tmp_dir

        # save new checkpoint
        torch.save(exp_state_dict, os.path.join(log, 'checkpoint.pt'))

        # save evaluation metrics into JSON file
        eval_metrics_fpath = os.path.join(log, 'metrics.json')

        eval_experiment = Experiment(config)
        eval_experiment.load_state_dict(exp_state_dict, strict=True)
        eval_experiment.evaluate(eval_metrics_fpath)

        print(f'Result saved to: {eval_metrics_fpath}')

def update_finetune_config(config):

    alt_config = copy.deepcopy(config)
    alt_config.protection = None
    alt_config.model.opt_param.lr *= 0.1
    if 'pretrain_iter' in alt_config.hparam.to_dict():
        alt_config.hparam.pretrain_iter = 0
    alt_config.hparam.iteration //= 2

    log_path = alt_config.log.path
    postfix = ('-D' if args.load_discriminator else '-ND') + '-FT'
    alt_config.log.path = os.path.abspath(log_path) + postfix
    alt_config.attack_mode = 'FINETUNE'

    return alt_config

def update_overwrite_config(config):
    alt_config = copy.deepcopy(config)

    mssg = 'Experiment not supported, no black-box protection found'
    assert hasattr(alt_config.protection, 'bbox'), mssg
    assert args.watermark, 'please specify --watermark <PATH>'
    alt_config.protection.bbox.fn_out.watermark = args.watermark
    # remove white-box protection settings
    alt_config.protection.wbox = None
    alt_config.model.opt_param.lr *= 0.1
    if 'pretrain_iter' in alt_config.hparam.to_dict():
        alt_config.hparam.pretrain_iter = 0
    alt_config.hparam.iteration //= 2

    log_path = alt_config.log.path
    postfix = ('-D' if args.load_discriminator else '-ND') + '-OV'
    alt_config.log.path = os.path.abspath(log_path) + postfix
    alt_config.attack_mode = 'OVERWRITE'

    return alt_config


if __name__ == '__main__':

    config_fpath = os.path.join(args.log, 'config.yaml')
    assert os.path.exists(config_fpath), f'Invalid experiment log: {args.log}'

    config = Config.parse(config_fpath)

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)
    random.seed(config.seed)

    main(config)