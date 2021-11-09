from abc import ABC, abstractmethod
from experiments.util import Logger
from tqdm import tqdm
import importlib
import os
import torch

class Experiment(ABC):
    
    def __init__(self, config):
        super(Experiment, self).__init__()
        assert config is not None, '"config" is undefined'
        self.config = config
        self.logger = Logger(config)
        config_path = os.path.join(config.log.path, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config.to_yaml())
        self.init_step = 1
        self.configure_device()

    @abstractmethod
    def configure_dataset(self): pass

    def configure_device(self):
        print('*** DEVICE ***')
        
        use_gpu = self.config.resource.gpu
        has_gpu = torch.cuda.is_available()
        if has_gpu and use_gpu:
            gpu_count = torch.cuda.device_count()
            gpu_count = min(gpu_count, self.config.resource.get('ngpu', 1))
            self.device = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
        else:
            self.device = [torch.device('cpu'), ]

        if 'pretrain_iter' in self.config.hparam.to_dict():
            self.config.hparam.pretrain_iter //= len(self.device)
        self.config.hparam.iteration //= len(self.device)
        self.config.hparam.bsz *= len(self.device)

        for i, device in enumerate(self.device):
            print(f'{i}: {str(device).upper()}')
        print()

    @abstractmethod
    def configure_protection(self): pass

    @abstractmethod
    def configure_model(self): pass

    @abstractmethod
    def checkpoint(self): pass

    @abstractmethod
    def evaluate(self): pass
    
    def load_state_dict(self, state_dict, strict=False):
        assert hasattr(self, 'model'), '"model" not defined'
        self.model.load_state_dict(state_dict, strict=strict)
        if state_dict['step'] == 'END':
            total_iter = self.config.hparam.get('pretrain_iter', 0)
            total_iter += self.config.hparam.iteration
            self.init_step = total_iter
        else:
            self.init_step = state_dict['step'] + 1

    @abstractmethod
    def train(self, **kwargs): pass

    def start(self):
        pretrain_iteration = self.config.hparam.get('pretrain_iter', 0)
        iteration = self.config.hparam.iteration

        print('*** TRAINING ***')
        rng = range(self.init_step, pretrain_iteration + iteration + 1)
        for step in tqdm(rng):
            self._step = step
            self.train()
            self.checkpoint()

        self._step = 'end'
        self.checkpoint()
        print()