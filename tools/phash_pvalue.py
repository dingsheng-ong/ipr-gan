from scipy.stats import binom
from torchvision.transforms import functional as TF
import numpy as np
import pdqhash
import torch

def compute_hash(img_tensor):
    # hash batch of images and return phash of each image
    hash_batch = []
    for i in range(img_tensor.size(0)):

        x = np.uint8(TF.to_pil_image(img_tensor[i, ...]))
        h, q = pdqhash.compute(x)
        h = np.bool8(h)
        hash_batch.append(h)

    return np.stack(hash_batch)

def compute_matching_prob(img1, img2, min_size=32):
    # compute the p-value of matching both images' hash
    x = img1.clone()
    y = img2.clone()

    k = min(x.shape[2:])
    if k < min_size:
        h = int(x.shape[2] * min_size / k)
        w = int(x.shape[3] * min_size / k)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode='bicubic', align_corners=False)
        y = torch.nn.functional.interpolate(y, size=(h, w), mode='bicubic', align_corners=False)

    hash_x = compute_hash(x)
    hash_y = compute_hash(y)

    n = hash_x.shape[1]
    r = n - (hash_x ^ hash_y).sum(axis=1)
    prob = np.vectorize(lambda r: 1 - binom(n=n, p=0.5).cdf(r - 1))
    p_value = torch.FloatTensor(prob(r))

    return p_value