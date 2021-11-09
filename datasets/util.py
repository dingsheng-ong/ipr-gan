from torch.utils.data import Dataset, DataLoader

class Loader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(Loader, self).__init__(*args, **kwargs)
        self.iterator = iter(self)

    def __len__(self): return len(self.dataset)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self)
            return next(self.iterator)