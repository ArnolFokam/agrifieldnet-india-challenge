import os
import logging
import getpass
from typing import List

import torch
import rasterio
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from radiant_mlhub import Dataset
from typing import Callable, Optional

from aic.helpers import get_dir, get_folder_ids


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
        transform: Optional[Callable] = None):

        self.bands = bands
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        dataset_name = 'ref_agrifieldnet_competition_v1'
        source_collection = f'{dataset_name}_source'
        label_collection = f'{dataset_name}_labels_{"train" if self.train else "test"}'

        if download:
            self.download_data(root_dir, dataset_name, bands)

        folder_ids = get_folder_ids(self.root_dir, dataset_name, label_collection)

        self.imgs: List[np.ndarray] = []
        self.field_ids: List[str] = []
        # self.field_id_to_imgs: List[str, int] = {}
        self.field_masks: List[np.ndarray] = []
        self.targets: List[int] = []

        pbar = tqdm(folder_ids)
        pbar.set_description('Extracting Images, Field Masks and Target Variable')

        for fidx in pbar:

            with rasterio.open(f'{self.root_dir}/{dataset_name}/{label_collection}/{label_collection}_{fidx}/field_ids.tif') as src:
                field_data = src.read()[0]

            # get bands for folder id
            bands_src = [rasterio.open(f'{self.root_dir}/{dataset_name}/{source_collection}/{source_collection}_{fidx}/{band}.tif') for band in self.bands]
            bands_array = [np.expand_dims(band.read(1), axis=0) for band in bands_src]
            bands = np.vstack(bands_array)
            

            if self.train:
                with rasterio.open(f'{self.root_dir}/{dataset_name}/{label_collection}/{label_collection}_{fidx}/raster_labels.tif') as src:
                    label_data = src.read()[0]

            # get unique field ids in the field data
            field_ids = list(np.unique(field_data))
            field_ids.remove(0)

            for fid in field_ids:

                # append field ids
                self.field_ids.append(fid)

                # some chips have more that one training field, so, 
                # it is better to have a list of unique chips 
                # and a reference to the correct chip per field 
                # id rather than having the same chip for every field id
                # more space economical
                #
                # NOTE: we can't build a mapping because the field ids 
                # and the bands are in a many-to-many relationship
                #
                # last_img_index = len(self.imgs) - 1 # index of the current spectral bands
                # self.field_id_to_imgs[fid] = last_img_index
                
                # append spectral bands
                self.imgs.append(bands)

                # append field mask
                mask = np.where(field_data == fid, 1.0, 0.0)
                self.field_masks.append(mask)

                if self.train:

                    # get pixels with crop label per band
                    label = int(label_data[field_data == fid].mean())

                    assert (label_data[field_data == fid] == label).all(), "[Corrupted Data] field data has more than one label."

                    # append label
                    self.targets.append(label)


    def __getitem__(self, index: str):
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
    ds = AgriFieldDataset('data/source', train=True)