import numpy as np
from typing import Dict, List, Union
from PIL.Image import Image
import torch
import scipy
from aic.augmentation import RandomFieldAreaCrop, ReduceSkewness, RotateBands

import albumentations as A
import albumentations.pytorch.transforms as TorchT

from aic.dataset import AgriFieldDataset


class BaselineTransfrom:
    def __init__(self, bands: List[str], crop_size: int = 32) -> None:
        self.crop_size = crop_size
        self.bands = bands
        
        # transform that change the value of a voxel in the spectral bands
        self.voxel_value_transform = A.Compose([
            ReduceSkewness(),
            A.Normalize(mean=[AgriFieldDataset.mean[band] for band in self.bands],
                        std=[AgriFieldDataset.std[band] for band in self.bands])
        ])

        # transform that changes the geometric shape of the image (rotation, translation, etc)
        self.geometric_transform = A.Compose([
            RandomFieldAreaCrop(crop_size=self.crop_size),
            A.HorizontalFlip(),
            RotateBands(limit=180)
        ])

        # transform after all the important ones, usually to convert to tensor
        self.final_transform = A.Compose([
            TorchT.ToTensorV2()
        ])

    def __call__(self, image: Union[np.ndarray, Image], mask: Union[np.ndarray, Image]) -> Dict[str,  Union[np.ndarray, Image]]:
        # image = self.voxel_value_transform(image=image)["image"]
        
        # dilate mask to slight increase the region of interest
        mask = scipy.ndimage.binary_dilation(mask.astype(np.uint8), structure=np.ones((5, 5),np.uint8), iterations = 2)
        
        transformed = self.geometric_transform(image=image, mask=mask)
        transformed = self.final_transform(image=transformed["image"], mask=transformed["mask"])

        return {
            "image": transformed["image"],
            "mask": transformed["mask"]
        }

if __name__ == '__main__':
    ds = AgriFieldDataset('data/source', transform=BaselineTransfrom(crop_size=16), train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    fids, imgs, masks, targets = next(loader)