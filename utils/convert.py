import math
import numpy as np

from torchvision.utils import make_grid


def rgb_to_luma(image):
    assert isinstance(image, np.ndarray), 'image must be numpy array'
    assert image.ndim == 3 or image.ndim == 1, 'image must be 2D or 3D'

    dtype = image.dtype

    image = image.astype(np.float64)
    image = (np.dot(image, [65.481, 128.553, 24.966]) / 255.0 + 16.0).round()

    return image.astype(dtype)


def tensor_to_numpy_image(image, pixel_range=[0, 1]):

    image = image.float().detach().cpu().clamp_(*pixel_range)
    n_dim = image.dim()

    if n_dim == 4:
        n_img = image.size(0)
        image = make_grid(image, nrow=int(math.sqrt(n_img))).numpy()
        image = image.transpose(1, 2, 0)

    if n_dim == 3:
        image = image.numpy()
        image = image.transpose(1, 2, 0)
    
    if n_dim == 2:
        image = image.numpy()

    if n_dim not in [2, 3, 4]:
        raise TypeError('Only support 2D, 3D, 4D tensors, got %dD tensor instead' % n_dim)

    image = (image * 255.0).round().astype(np.uint8)

    return image
