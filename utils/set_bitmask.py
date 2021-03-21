import torch
import torch.nn as nn

from itertools import chain


def set_bitmask(model, string=None):
    device = next(model.parameters()).device

    if string is not None:
        # convert string to binary string of signs +1 and -1
        binary_string = list(chain.from_iterable(
            map(
                lambda x: map(int, bin(x)[2:].zfill(8)),
                string.encode('utf-8')
            )
        ))

        # calculate total number of batchnorm weights
        n_params = sum(map(
            lambda norm_layer: norm_layer.num_features,
            filter(
                lambda layer: isinstance(layer, nn.BatchNorm2d),
                model.modules()
            )
        ))

        # duplicate binary string
        binary_string *= n_params // len(binary_string) + 1

        split = lambda x, n: (x[:n], x[n:])
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                with torch.no_grad():
                    # extract part of the string as sign bitmask
                    bitmask, binary_string = split(binary_string, layer.num_features)
                    bitmask = torch.FloatTensor(bitmask) * 2 - 1
                    # register as buffer of BatchNorm2d
                    layer.register_buffer('bitmask', bitmask.float().to(device))
                    # apply the sign bit to weight
                    layer.weight.data.uniform_(0, 1).abs_().mul_(layer.bitmask)

    else:

        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                with torch.no_grad():
                    # random generate bitmask
                    bitmask = torch.randint(2, [layer.num_features]) * 2 - 1
                    # register as buffer of BatchNorm2d
                    layer.register_buffer('bitmask', bitmask.float().to(device))
                    # apply the sign bit to weight
                    layer.weight.data.uniform_(0, 1).abs_().mul_(layer.bitmask)
