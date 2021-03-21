import os
import requests

from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity


loaders = {}

def dataset_loader(tag):
    """
    register function into loaders
    """
    def loader_fn(f):
        loaders[tag] = f
        return f

    return loader_fn


def create_data_dir(dataset_name):
    """
    Create data directory path and create the directory if not exist
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(data_dir, f"data/{dataset_name}")
    os.makedirs(data_dir, exist_ok=True)

    return data_dir


def _download_gdrive(gdrive_id, file, size, md5, chunk_size=32 * 1024):
    '''
    Download file from Google Drive to destination directory
    '''
    if not check_integrity(file, md5):
        url  = 'https://drive.google.com/uc?export=download'
        print(f'Downloading {url}?id={gdrive_id} to {file}')

        session = requests.Session()
        params  = { 'id': gdrive_id }
        resp    = session.get(url, params=params, stream=True)

        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                params['confirm'] = value
                resp = session.get(url, params=params, stream=True)
                break

        with open(file, 'wb') as f:
            total = - (-size // chunk_size)
            for chunk in tqdm(resp.iter_content(chunk_size), total=total):
                if chunk: f.write(chunk)
        
        with zipfile.ZipFile(file) as f:
            f.extractall(os.path.dirname(file))

    print('Files already downloaded and verified')


@dataset_loader('celeb-a')
def load_celeb_a(img_size=32, batch_size=32, num_workers=1):
    '''
    Load Celeb-A dataset, combining the train and valid set
    '''
    data_dir = create_data_dir('celeb-a')

    gdrive_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    file      = os.path.join(data_dir, 'img_align_celeba.zip')
    size      = 1443490838
    md5       = '00d2c5bc6d35e252742224ab0c1e8fcb'

    _download_gdrive(gdrive_id, file, size, md5)

    mean = std = [0.5, ] * 3
    transform = transforms.Compose([
        transforms.Resize(img_size),             # resize image H x W
        transforms.CenterCrop(img_size),         # crop center
        transforms.ToTensor(),                   # convert to PyTorch Tensor
        transforms.Normalize(mean, std),         # normalize data to [-1, +1]
    ])

    loader = datasets.ImageFolder(root=data_dir, transform=transform)

    return DataLoader(
        loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


@dataset_loader('mnist')
def load_mnist(img_size=32, batch_size=32, num_workers=1):
    """
    Load MNIST dataset, combining the train and valid set.
    """
    data_dir = create_data_dir('mnist')

    transform = transforms.Compose([
        transforms.Resize(img_size),                    # resize image H x W
        transforms.ToTensor(),                          # convert to PyTorch Tensor
        transforms.Normalize([0.5, ], [0.5, ]),         # normalize data to [-1, +1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # convert single channel to RGB
    ])

    A = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    B = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    return DataLoader(
        ConcatDataset([A, B]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


@dataset_loader('cifar10')
def load_cifar10(img_size=32, batch_size=32, num_workers=1):
    """
    Load CIFAR10 dataset, combining the train and valid set.
    """
    data_dir = create_data_dir('cifar10')

    mean = std = [0.5, ] * 3
    transform = transforms.Compose([
        transforms.Resize(img_size),     # resize image H x W
        transforms.ToTensor(),           # convert to PyTorch Tensor
        transforms.Normalize(mean, std), # normalize data to [-1, +1]
    ])

    A = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    B = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    return DataLoader(
        ConcatDataset([A, B]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


@dataset_loader('cub200')
def load_cub200(img_size=32, batch_size=32, num_workers=1):
    """
    Load CIFAR10 dataset, combining the train and valid set.
    """
    from dataset.cub200 import CUB200
    data_dir = create_data_dir('cub200')

    mean = std = [0.5, ] * 3
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.ToTensor(),           # convert to PyTorch Tensor
        transforms.Normalize(mean, std), # normalize data to [-1, +1]
    ])

    A = CUB200(data_dir, train=True,  download=True, transform=transform)
    B = CUB200(data_dir, train=False, download=True, transform=transform)

    return DataLoader(
        ConcatDataset([A, B]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == '__main__':

    for key, value in sorted(lodaers.items()):
        print(f'{key:8s}: {value}')
        data, _ = next(iter(value()))
        print('Size:', data.size())
