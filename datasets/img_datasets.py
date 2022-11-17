from datasets.util import Loader
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
import os
import pandas as pd

def cifar10(**kwargs):

    mean = std = [0.5, ] * 3
    transform = transforms.Compose([
        transforms.Resize(kwargs['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    download = not os.path.exists(kwargs['path'])

    train_data = datasets.CIFAR10(
        kwargs['path'],
        train=True,
        download=download,
        transform=transform
    )
    test_data = datasets.CIFAR10(
        kwargs['path'],
        train=False,
        download=download,
        transform=transform
    )

    return Loader(
        ConcatDataset([train_data, test_data]),
        batch_size=kwargs['batch_size'],
        shuffle=kwargs.get('shuffle', True),
        num_workers=kwargs['num_workers'],
        drop_last=kwargs.get('drop_last', False)
    )

def cub200(**kwargs):

    mean = std = [0.5, ] * 3
    transform = transforms.Compose([
        transforms.Resize(kwargs['size']),
        transforms.CenterCrop(kwargs['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    download = not os.path.exists(kwargs['path'])

    train_data = CUB200(
        kwargs['path'],
        train=True,
        download=download,
        transform=transform
    )
    test_data = CUB200(
        kwargs['path'],
        train=False,
        download=download,
        transform=transform
    )

    return Loader(
        ConcatDataset([train_data, test_data]),
        batch_size=kwargs['batch_size'],
        shuffle=kwargs.get('shuffle', True),
        num_workers=kwargs['num_workers'],
        drop_last=kwargs.get('drop_last', False)
    )

class CUB200(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target