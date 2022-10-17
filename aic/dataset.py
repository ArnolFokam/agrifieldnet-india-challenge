from collections import defaultdict
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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s:', datefmt='%H:%M:%S')

MLHUB_API_KEY = os.getenv('MLHUB_API_KEY')


class AgriFieldDataset(torch.utils.data.Dataset):

    num_classes = 13

    mean = {
        'B01': 43.2848714,
        'B02': 38.65350979,
        'B03': 37.54214063,
        'B04': 39.3331079,
        'B05': 42.775389,
        'B06': 55.31826163,
        'B07': 63.86237698,
        'B08': 60.60235558,
        'B8A': 70.48009153,
        'B09': 13.49497597,
        'B11': 70.37100837,
        'B12': 48.97029943
    }

    std = {
        'B01': 3.3175816420959223,
        'B02': 4.209897807865492,
        'B03': 5.451427325056402,
        'B04': 8.993650872256744,
        'B05': 7.744416117179397,
        'B06': 6.593167201674543,
        'B07': 7.811851260911143,
        'B08': 7.526517502372515,
        'B8A': 8.873488112652709,
        'B09': 2.5816426796841645,
        'B11': 16.13157818851457,
        'B12': 14.846548889561879
    }

    max_pixel_value = {
        'B01': 130,
        'B02': 138,
        'B03': 137,
        'B04': 148,
        'B05': 148,
        'B06': 154,
        'B07': 159,
        'B08': 168,
        'B8A': 167,
        'B09': 108,
        'B11': 255,
        'B12': 255
    }

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

    def __init__(
            self,
            root_dir: str,
            bands: List[str],
            vegetative_indeces: List[str],
            train: bool = True,
            download: bool = False,
            save_cache: bool = False,
            transform: Optional[Callable] = None,):

        self.selected_bands = bands
        self.vegetative_indces = vegetative_indeces
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        dataset_name = 'ref_agrifieldnet_competition_v1'
        source_collection = f'{dataset_name}_source'
        label_collection = f'{dataset_name}_labels_{"train" if self.train else "test"}'

        if download:
            self.__download_data(root_dir, dataset_name, bands)

        self.save_cache_dir = get_dir(
            f"{self.root_dir}/cache_{'_'.join(self.selected_bands)}_{'_'.join(self.vegetative_indces)}")

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
            logging.info(
                'Error occured during cache data loading. Preprocessing data again...')

            folder_ids = get_folder_ids(
                self.root_dir, dataset_name, label_collection)

            self.imgs: List[np.ndarray] = []
            self.field_ids: List[str] = []
            self.field_masks: List[np.ndarray] = []
            self.targets: List[int] = []
            self.class_meta = self.__load_meta_class()

            pbar = tqdm(folder_ids)
            pbar.set_description(
                f'Extracting data for {"train" if self.train else "test"} set')

            for fidx in pbar:

                with rasterio.open(f'{self.root_dir}/{dataset_name}/{label_collection}/{label_collection}_{fidx}/field_ids.tif') as src:
                    field_data = src.read()[0]

                # get bands for folder id
                bands_src = [rasterio.open(
                    f'{self.root_dir}/{dataset_name}/{source_collection}/{source_collection}_{fidx}/{band}.tif') for band in self.selected_bands]
                bands_array = [np.expand_dims(band.read(1), axis=0) for band in bands_src]
                bands = np.vstack(bands_array).transpose(1, 2, 0)  # convert to H x W x C
                bands = self.__add_spectral_indices(bands, self.vegetative_indces)
                

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
                    mask = field_data.copy()
                    mask[mask != fid] = 0
                    mask[mask == fid] = 1

                    assert np.array_equal(np.unique(mask), np.array(
                        [0, 1])), "[Incorrect code] mask must be binary"

                    self.field_masks.append(mask)

                    if self.train:

                        # get pixels with crop label per band
                        label = int(label_data[field_data == fid].mean())

                        assert (label_data[field_data == fid] == label).all(
                        ), "[Corrupted Data] field data has more than one label."

                        # append label
                        self.targets.append(label)

            self.imgs = np.array(self.imgs)
            self.field_ids = np.array(self.field_ids)
            self.field_masks = np.array(self.field_masks)
            self.targets = np.array(self.targets)

            # normalize data accross dataset
            # for c in range(self.imgs.shape[-1]):
            #    mean = self.imgs[:, :, :, c].mean()
            #    std = self.imgs[:, :, :, c].std()
            #    self.imgs[:, :, :, c] = (self.imgs[:, :, :, c] - mean) / std

            if save_cache:
                logging.info('Caching data for subsequent use...')

                with open(f'{self.save_cache_dir}/imgs.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(self.imgs, f)

                with open(f'{self.save_cache_dir}/field_ids.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(self.field_ids, f)

                with open(f'{self.save_cache_dir}/field_masks.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(self.field_masks, f)

                with open(f'{self.save_cache_dir}/class_meta.{"train" if self.train else "test"}.cache.pkl', 'wb') as f:
                    pickle.dump(self.class_meta, f)

                if self.train:
                    with open(f'{self.save_cache_dir}/targets.train.cache.pkl', 'wb') as f:
                        pickle.dump(self.targets, f)

        else:
            logging.info('Data loaded from cached files...')

    def __len__(self):
        return len(self.field_ids)

    def __getitem__(self, index: str):
        field_id, image, field_mask, target = self.field_ids[index], self.imgs[
            index], self.field_masks[index], self.targets[index] if self.train else torch.empty(1)
        if self.transform:
            transformed = self.transform(image=image.astype(
                np.float64), mask=field_mask.astype(np.float64))
            image = transformed["image"].float()
            field_mask = transformed["mask"].float()
        else:
            image, field_mask = torch.FloatTensor(
                image), torch.FloatTensor(field_mask)

        return int(field_id), image, field_mask, int(self.class_meta[target]["loss_label"]) if self.train else torch.empty(1)

    @staticmethod
    def get_class_weights(targets):
        weights = defaultdict(lambda: 0)
        total_instances = len(targets)

        # get the sum of all instances
        for target in targets:
            weights[target] += 1

        # divided by the total instances
        weights = {k: v / total_instances for k, v in weights.items()}

        return weights

    def __download_data(
            self,
            download_dir: str,
            dataset_name: str,
            bands: Optional[List[str]] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):

        if not MLHUB_API_KEY:
            logging.warning(
                'MLHub Api Key not found. Consider writing it in an .env file as "MLHUB_API_KEY"')
            os.environ['MLHUB_API_KEY'] = getpass.getpass(
                prompt="MLHub API Key: ")

        # fetch informatin about the dataset
        dataset = Dataset.fetch(dataset_name)

        # handle filters
        assets = ['field_ids', 'raster_labels']
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

    def __load_meta_class(self):
        """
        Returns a mapping of the true index 
        from the dataset to contiguous index 
        from 0 - 13 for classification loss
        """

        return {k: {
            "name": self.crops[k],
            "loss_label": v,
        } for k, v in zip(self.crops.keys(), range(len(self.crops.keys())))}

    def __add_spectral_indices(self, bands, indices: str):
        """
        Add vegetative indeces to the data
        """
        idx_formulas = {
            "NDVI": lambda x: (x[:, :, self.selected_bands.index('B08')] - x[:, :, self.selected_bands.index('B04')]) / (x[:, :, self.selected_bands.index('B08')] + x[:, :, self.selected_bands.index('B04')] + 1e-6),
            "NDWI_GREEN": lambda x: (x[:, :, self.selected_bands.index('B03')] - x[:, :, self.selected_bands.index('B08')]) / (x[:, :, self.selected_bands.index('B03')] + x[:, :, self.selected_bands.index('B08')] + 1e-6),
            "NDWI_BLUE": lambda x: (x[:, :, self.selected_bands.index('B02')] - x[:, :, self.selected_bands.index('B08')]) / (x[:, :, self.selected_bands.index('B02')] + x[:, :, self.selected_bands.index('B08')] + 1e-6)
        }
        
        for veg_idx in indices:
            tmp = np.expand_dims(idx_formulas[veg_idx](bands), -1)
            bands = np.dstack((*np.split(bands, bands.shape[-1], -1), tmp))
            
        return bands


if __name__ == '__main__':
    ds = AgriFieldDataset('data/source',  bands=["B04", "B03", "B02", "B08"], vegetative_indeces=["NDVI"], save_cache=True, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    fids, imgs, masks, target = next(loader)
