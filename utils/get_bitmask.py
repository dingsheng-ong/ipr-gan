import torch.nn as nn
import re


def get_bitmask(model, return_string=False):
    bitmask = []
    weight_sign = []
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            bm = str.join('', map(str, ((layer.bitmask.long() + 1) / 2).tolist()))
            ws = str.join('', map(str, ((layer.weight.data.sign().long() + 1) / 2).tolist()))
            if return_string:
                bm += '0' * (len(bm) % 8)
                ws += '0' * (len(bm) % 8)
                bm = str.join('', map(lambda x: chr(int(x, 2)), re.findall('.' * 8, bm)))
                ws = str.join('', map(lambda x: chr(int(x, 2)), re.findall('.' * 8, ws)))
            bitmask += [bm]
            weight_sign += [ws]

    bitmask = ','.join(bitmask)
    weight_sign = ','.join(weight_sign)
    
    return bitmask, weight_sign
