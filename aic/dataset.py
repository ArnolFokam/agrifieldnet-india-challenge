import os
import logging
import getpass
import pickle
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
        save_cache: bool = False,
        bands: Optional[List[str]] = ['B01', 'B02', 'B03', 'B04','B05','B06','B07','B08','B8A', 'B09', 'B11', 'B12'],
        transform: Optional[Callable] = None):

        self.selected_bands = bands
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        dataset_name = 'ref_agrifieldnet_competition_v1'
        source_collection = f'{dataset_name}_source'
        label_collection = f'{dataset_name}_labels_{"train" if self.train else "test"}'

        if download:
            self.download_data(root_dir, dataset_name, bands)

        self.save_cache_dir = get_dir(f"{self.root_dir}/cache_{'_'.join(self.selected_bands)}")

        try:
            logging.info('Loading data from cache...')
            with open(f'{self.save_cache_dir}/imgs.{"train" if self.train else "test"}.cache.pkl', 'rb') as f:
                self.imgs = pickle.load(f)

            with open(f'{self.save_cache_dir}/field_ids.{"train" if self.train else "test"}.cache.pkl', 'rb') as f:
                self.field_ids = pickle.load(f)

            with open(f'{self.save_cache_dir}/field_masks.{"train" if self.train else "test"}.cache.pkl', 'rb') as f:
                self.field_masks = pickle.load(f)

            with open(f'{self.save_cache_dir}/class_meta.{"train" if self.train else "test"}.cache.pkl', 'rb') as f:
                self.class_meta = pickle.load(f)

            if self.train:
                with open(f'{self.save_cache_dir}/targets.train.cache.pkl', 'rb') as f:
                    self.targets = pickle.load(f)

        except (IOError, EOFError, OSError, pickle.PickleError, pickle.UnpicklingError):
            logging.info('Error occured during cache data loading. Preprocessing data again...')

            folder_ids = get_folder_ids(self.root_dir, dataset_name, label_collection)

            self.imgs: List[np.ndarray] = []
            self.field_ids: List[str] = []
            self.field_masks: List[np.ndarray] = []
            self.targets: List[int] = []
            self.class_meta = self.load_meta_class()

            pbar = tqdm(folder_ids)
            pbar.set_description(f'Extracting data for {"train" if self.train else "test"} set')

            for fidx in pbar:

                with rasterio.open(f'{self.root_dir}/{dataset_name}/{label_collection}/{label_collection}_{fidx}/field_ids.tif') as src:
                    field_data = src.read()[0]

                # get bands for folder id
                bands_src = [rasterio.open(f'{self.root_dir}/{dataset_name}/{source_collection}/{source_collection}_{fidx}/{band}.tif') for band in self.selected_bands]
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

            # get class weights to handle imbalance
            # TODO: find a more effiecient and clean way to do this
            if self.train:
                total_instances = len(self.targets)

                # get the sum of all instances
                for target in self.targets:
                    if "weight" not in self.class_meta[target].keys():
                        self.class_meta[target]["weight"] = 0
                    else:
                        self.class_meta[target]["weight"] +=  1 

                # divided by the total instances
                for key in self.class_meta.keys():
                    self.class_meta[key]["weight"] /= total_instances

                # make sure that every target class has a 
                # weights, else there is an error somewhere
                assert all(["weight" in self.class_meta[target].keys() for target in self.class_meta.keys()]), "[Error in code] Not all target have class weights"

            if save_cache:
                logging.info('Caching data for subsequent use...')

                with open(f'{self.save_cache_dir}/imgs.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(np.vstack(self.imgs), f)

                with open(f'{self.save_cache_dir}/field_ids.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(np.array(self.field_ids), f)

                with open(f'{self.save_cache_dir}/field_masks.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(np.vstack(self.field_masks), f)

                with open(f'{self.save_cache_dir}/class_meta.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(self.class_meta, f)

                if self.train:
                    with open(f'{self.save_cache_dir}/targets.train.cache.pkl', 'wb') as f:
                        pickle.dump(np.array(self.targets), f)

        else:
            logging.info('Data loaded from cached files...')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: str):
        field_id, image, field_mask, target = self.field_ids[index], self.imgs[index], self.field_masks[index], self.targets[index]
        
        if self.transform:
            transformed = self.transform(image=image, mask=field_mask)
            image = transformed["image"]
            field_mask = transformed["mask"]

        return int(field_id), torch.FloatTensor(image), torch.FloatTensor(field_mask), int(self.class_meta[target]["loss_label"])


    def download_data(
        self, 
        download_dir: str,
        dataset_name: str,
        bands: Optional[List[str]] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):

        if not MLHUB_API_KEY:
            logging.warning('MLHub Api Key not found. Consider writing it in an .env file as "MLHUB_API_KEY"')
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

    def load_meta_class(self):
        """
        Returns a mapping of the true index 
        from the dataset to contiguous index 
        from 0 - 13 for classification loss
        """
        crops = {
            1: "Wheat",
            2: "Mustard",
            3: "Lentil",
            4: "No Crop/Fallow",
            5: "Green pea",
            6: "Sugarcane",
            8: "Garlic",
            9: "Maize",
            13: "Gram",
            14: "Coriander",
            15: "Potato",
            16: "Bersem",
            36: "Rice"
        }

        return { k: { 
            "name": crops[k], 
            "loss_label": v,
        } for k, v in zip(crops.keys(), range(len(crops.keys())))}


        


if __name__ == '__main__':
    ds = AgriFieldDataset('data/source',  save_cache=True, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    fids, imgs, masks, target = next(loader)