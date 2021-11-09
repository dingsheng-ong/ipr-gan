from configs import Config

import argparse
import experiments
import numpy as np
import os
import random
import torch

parser = argparse.ArgumentParser(description='IPR-GAN training script')

ConfigFile = lambda path: Config.parse(path)
parser.add_argument('-c', '--config', required=True, type=ConfigFile,
                                     metavar='PATH', help='Path to YAML config file')

args = parser.parse_args()

def main(config):
    
    if not config.resource.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    Experiment = getattr(experiments, config.experiment)
    experiment = Experiment(config)

    ckpt_path = os.path.join(config.log.path, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        print('*** LOAD CHECKPOINT ***')
        state_dict = torch.load(ckpt_path)
        experiment.load_state_dict(state_dict)
        print(f'From Step: {experiment.init_step}\n')

    experiment.start()

    # save evaluation metrics into JSON file
    eval_metrics_fpath = os.path.join(config.log.path, 'metrics.json')
    experiment.evaluate(eval_metrics_fpath)
    print(f'Result saved to: {eval_metrics_fpath}')

if __name__ == '__main__':
    config = args.config

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(config.seed)
    random.seed(config.seed)

    main(config)
