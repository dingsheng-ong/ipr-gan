import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


__all__ = ['loaders', 'dataset_list']

loaders = {}
dataset_list = [
    'BSDS100',
    'BSDS200',
    'General100',
    'historical',
    'ILSVR2016-subset',
    'Manga109',
    'Set14',
    'Set5',
    'T91',
    'Urban100',
]


def _loader(dataset):
    def _loader_fn(scale=4, img_size=96, batch_size=32, num_workers=1):
        directory = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(directory, f'data/{dataset}')
        assert os.path.exists(root), f'{root} does not exist'
        return DataLoader(
            SRImageDataset(root, scale=scale, img_size=img_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    return _loader_fn


for dataset in dataset_list:
    loaders[dataset] = _loader(dataset)


class SRImageDataset(Dataset):

    def __init__(self, root, scale=4, img_size=96):
        super(SRImageDataset, self).__init__()
        self.root        = root
        self.hr_img_size = img_size
        self.lr_img_size = img_size // scale

        self.data = []
        for r, ds, fs in os.walk(root):
            self.data += [os.path.join(r, f) for f in fs]
        self.data = sorted(self.data)

        self.resize    = transforms.RandomCrop(img_size)
        self.tensor    = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        hr_image = self.resize(image)
        lr_image = hr_image.resize([self.lr_img_size] * 2, Image.BICUBIC)

        hr_image = self.tensor(hr_image)
        lr_image = self.tensor(lr_image)

        return lr_image, hr_image


# if __name__ == '__main__':

#     for dataset in dataset_list:
#         loader = loaders[dataset](batch_size=8, num_workers=16)
#         lr_img, hr_img = next(iter(loader))
#         # display
#         print(dataset)
#         print('N:', len(loader.dataset))
#         print('Sample:', list(lr_img.size()), list(hr_img.size()))
