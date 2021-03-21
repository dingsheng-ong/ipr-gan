import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce


def sign_loss(model, gamma_0=0.1):
    device    = next(model.parameters()).device
    filter_bn = lambda layer: isinstance(layer, nn.BatchNorm2d)
    loss      = lambda layer: F.relu(gamma_0 - layer.weight * layer.bitmask).mean() if hasattr(layer, 'bitmask') else 0
    return reduce(
        lambda prev, curr: loss(curr) + prev,
        filter(filter_bn, model.modules()),
        torch.tensor(0.).to(device)
    )
