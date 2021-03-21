import math
import numpy as np
from scipy import linalg
from scipy.stats import entropy

import torch
import torch.nn.functional as F
import torch.utils.data

from metrics.inception import InceptionV3
from tqdm import tqdm


def calc_entropy(ys, n, splits):
    split_scores = []  # Split inception scores

    for k in range(splits):
        part = ys[k * (n // splits): (k + 1) * (n // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def FID(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def forward_inception(loader, device):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    ys = []
    fs = []

    n = len(loader.dataset); n = n // loader.batch_size + (n % loader.batch_size > 0)
    for i, (imgs, _) in tqdm(enumerate(loader), total=n, desc='Inception', leave=False):
        _, c, h, w = imgs.size()
        imgs = imgs.to(device)

        if c == 1:
            imgs = imgs.expand(-1, 3, -1, -1)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            imgs = F.interpolate(imgs, (299, 299), mode='bilinear', align_corners=False)  # bilinear

        # Feed images to the inception module to get the softmax predictions
        with torch.no_grad():
            f = model(imgs)[0].view(-1, 2048)
            y = F.softmax(model.fc(f), dim=1)

        ys.append(y)
        fs.append(f)

    ys = torch.cat(ys, dim=0).cpu().data.numpy()
    fs = torch.cat(fs, dim=0).cpu().data.numpy()

    return fs, ys

