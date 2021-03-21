import math
import numpy as np


def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse  = np.mean((img1 - img2) ** 2)
    
    if mse == 0: return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
