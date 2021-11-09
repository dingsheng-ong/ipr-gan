from abc import ABC, abstractmethod
from collections import OrderedDict

class Model(ABC):

    def __init__(self):
        self._modules = OrderedDict()
    
    @abstractmethod
    def compute_g_loss(self): pass

    @abstractmethod
    def compute_d_loss(self): pass

    @abstractmethod
    def forward_d(self): pass

    @abstractmethod
    def forward_g(self): pass

    @abstractmethod
    def get_metrics(self): pass

    def load_state_dict(self, state_dict, strict=False):
        for name, m in self._modules.items():
            if strict:
                assert name in state_dict, f'Missing key: {name}'
                self._modules[name].load_state_dict(state_dict[name])
            else:
                if name in state_dict:
                    if name in self._modules:
                        self._modules[name].load_state_dict(state_dict[name])

    def state_dict(self):
        state_dict = OrderedDict()
        for name, m in self._modules.items():
            state_dict[name] = m.state_dict()
        return state_dict

    @abstractmethod
    def update_d(self): pass

    @abstractmethod
    def update_g(self): pass

class Wrapper(Model):

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif hasattr(self.model, key):
            return getattr(self.model, key)
        else:
            return None

    @abstractmethod
    def configure(self): pass

    def compute_d_loss(self):
        self.model.compute_d_loss()

    @abstractmethod
    def compute_g_loss(self): pass

    def forward_d(self, data):
        self.model.forward_d(data)

    @abstractmethod
    def forward_g(self): pass

    def update_d(self, data):
        self.model.update_d(data)

    @abstractmethod
    def update_g(self): pass