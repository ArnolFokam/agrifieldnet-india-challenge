import os
import logging
import getpass
from typing import List

import torch
import numpy as np
from dotenv import load_dotenv
from radiant_mlhub import Dataset
from typing import Callable, Optional

from aic.helpers import get_dir


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s:', datefmt='%H:%M:%S')

MLHUB_API_KEY = os.getenv('MLHUB_API_KEY')



class AgriFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str, 
        train: bool = True,
        download: bool = False,
        bands: Optional[List[str]] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        transform: Optional[Callable] =None):

        self.bands = bands
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        dataset_name = 'ref_agrifieldnet_competition_v1'
        source_collection = f'{dataset_name}_source'
        label_collection = f'{dataset_name}_labels_{"train" if self.train else "test"}'

        if download:
            self.download_data(root_dir, dataset_name, bands)

        self.imgs: List[np.ndarray] = []
        self.field_masks: List[np.ndarray] = []
        self.targets: List[int] = []

    def __getitem__(self, index):
        pass

    def download_data(
        self, 
        download_dir: str,
        dataset_name: str,
        bands: Optional[List[str]] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):

        if not MLHUB_API_KEY:
            logging.warning('MLHub Api Key not found. Do you save it as an environmental variable?')
            os.environ['MLHUB_API_KEY'] =  getpass.getpass(prompt="MLHub API Key: ")

        # fetch informatin about the dataset
        dataset = Dataset.fetch(dataset_name)

        # handle filters
        assets = ['field_ids','raster_labels']
        filters = {
            f'{dataset_name}_labels_train': assets,
            f'{dataset_name}_labels_test': [assets[0]],
            f'{dataset_name}_source': bands
        }

        # download the dataset
        dataset.download(
            output_dir=get_dir(download_dir),
            collection_filter=filters
        )

if __name__ == '__main__':
    ds = AgriFieldDataset('data/source', download=True)