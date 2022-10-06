from torch.utils.data import Dataset

class AgriFieldDataset(Dataset):
    def __init__(
        self,
        root_dir: str, 
        transform: Callable =None):
        pass

    def __getitem__(self, index):
        pass