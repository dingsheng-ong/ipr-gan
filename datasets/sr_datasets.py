from PIL import Image
from torchvision import transforms
from datasets.util import Dataset, Loader
import glob
import os

class _4xDataset(Dataset):
    def __init__(self, root, size=96, test=False):
        super(_4xDataset, self).__init__()
        assert (size % 4 == 0) or (size < 0), f'{size} is not divisble by 4.'
        self.size = size

        self.files = glob.glob(os.path.join(root, '**/*'), recursive=True)
        if test:
            self.transform = _4xCenterCrop()
        else:
            self.transform = transforms.RandomCrop(size)
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert('RGB')

        hr = self.transform(image)
        w, h = hr.size
        lr = hr.resize((w // 4, h // 4), Image.BICUBIC)

        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)

        return lr, hr

class _4xCenterCrop(object):
    def __call__(self, img):
        assert isinstance(img, Image.Image), f'expected Image, got {type(img)}'
        w, h = img.size
        h = (h // 4) * 4
        w = (w // 4) * 4
        return transforms.functional.center_crop(img, (h, w))

def _loader(**kwargs):
    test = kwargs.get('test', False)
    return Loader(
        _4xDataset(kwargs['path'], size=kwargs['size'], test=test),
        batch_size=1 if test else kwargs['batch_size'],
        shuffle=kwargs.get('shuffle', not test),
        num_workers=kwargs['num_workers'],
        drop_last=kwargs.get('drop_last', not test)
    )

bsd100   = _loader
imagenet = _loader
set14    = _loader
set5     = _loader