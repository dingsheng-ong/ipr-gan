from PIL import Image
from torchvision import transforms
from datasets.util import Dataset, Loader
import glob
import os
import random

class _UnalignedDataset(Dataset):
    def __init__(self, dir_a, dir_b, load_size=143, crop_size=128, test=False):
        super(_UnalignedDataset, self).__init__()
        self.test = test
        self.transform = transforms.Compose([
            transforms.Resize(load_size, transforms.InterpolationMode.BICUBIC),
            (transforms.CenterCrop if test else transforms.RandomCrop)(crop_size),
            transforms.RandomHorizontalFlip(p=0.0 if test else 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ) * 3, (0.5, ) * 3)
        ])
        self.path_a = sorted(glob.glob(os.path.join(dir_a, '*'), recursive=True))
        self.path_b = sorted(glob.glob(os.path.join(dir_b, '*'), recursive=True))
        self.size_a = len(self.path_a)
        self.size_b = len(self.path_b)

    def __len__(self): return max(self.size_a, self.size_b)

    def __getitem__(self, index):
        path_a = self.path_a[index % self.size_a]
        _idxB = index if self.test else random.randint(0, self.size_b - 1)
        path_b = self.path_b[_idxB % self.size_b]
        img_a = Image.open(path_a).convert('RGB')
        img_b = Image.open(path_b).convert('RGB')
        a = self.transform(img_a)
        b = self.transform(img_b)
        return a, b

def _loader(**kwargs):
    test = kwargs.get('test', False)
    if test:
        dir_a = os.path.abspath(os.path.join(kwargs['path'], 'testA'))
        dir_b = os.path.abspath(os.path.join(kwargs['path'], 'testB'))
    else:
        dir_a = os.path.abspath(os.path.join(kwargs['path'], 'trainA'))
        dir_b = os.path.abspath(os.path.join(kwargs['path'], 'trainB'))

    return Loader(
        _UnalignedDataset(dir_a, dir_b, kwargs['load'], kwargs['crop'], test),
        batch_size=1 if test else kwargs['batch_size'],
        shuffle=kwargs.get('shuffle', not test),
        num_workers=kwargs['num_workers'],
        drop_last=kwargs.get('drop_last', not test)
    )

cityscapes = _loader