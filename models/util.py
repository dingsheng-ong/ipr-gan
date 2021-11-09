import random
import torch
import torch.nn as nn

class ImagePool(nn.Module):
    def __init__(self, pool_size):
        super(ImagePool, self).__init__()
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.register_buffer('images', torch.tensor([]))
            self.register_buffer('counts', torch.zeros([]))
            # self.images = []
            # self.counts = 0

    def load_state_dict(self, *args, **kwargs):
        self.images = torch.empty_like(args[0]['images'])
        super(ImagePool, self).load_state_dict(*args, **kwargs)

    def __call__(self, images):
        if self.pool_size <= 0:
            return images.detach()
        elif self.counts < self.pool_size:
            self.images = self.images.to(images.device)
            self.images = torch.cat([self.images, images.detach()], dim=0)
            self.images = self.images[:self.pool_size, ...]
            self.counts += images.size(0)
            return images.detach()
        else:
            images = images.detach()
            prob = torch.rand(images.size(0)) > 0.5
            index = torch.randperm(self.pool_size)[:images.size(0)]
            pool_images = self.images[index[prob]].clone()
            self.images[index[prob]] = images[prob].detach()
            images[prob] = pool_images
            return images.detach()
        # if self.pool_size <= 0: return images
        # return_images = []
        # for image in images:
        #     image = image.data[None, ...]
        #     if self.counts < self.pool_size:
        #         self.images.append(image)
        #         self.counts += 1
        #         return_images.append(image)
        #     else:
        #         prob = random.uniform(0, 1)
        #         if prob > 0.5:
        #             index = random.randint(0, self.pool_size - 1)
        #             return_images.append(self.images[index].clone())
        #             self.images[index] = image
        #         else:
        #             return_images.append(image)

        # return torch.cat(return_images, dim=0)

class DisableBatchNormStats(object):
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def __enter__(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                self.cache[name] = m.track_running_stats
                m.track_running_stats = False

    def __exit__(self, *args):
        for name, m in self.model.named_modules():
            if name in self.cache:
                m.track_running_stats = self.cache[name]