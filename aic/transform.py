import numpy as np
from typing import Dict, Union
from PIL.Image import Image
import torch
from aic.augmentation import RandomFieldAreaCrop

import albumentations as A
import albumentations.pytorch.transforms as TorchT

from aic.dataset import AgriFieldDataset


class BaselineTransfrom:
    def __init__(self, crop_size: int = 32) -> None:
        self.crop_size = crop_size
        
        # transform that change the value of a voxel in the spectral bands
        self.voxel_value_transform = A.Compose([
            
        ])

        # transform that changes the geometric shape of the image (rotation, translation, etc)
        self.geometric_transform = A.Compose([
            RandomFieldAreaCrop(crop_size=self.crop_size)
        ])

        # transform after all the important ones, usually to convert to tensor
        self.final_transform = A.Compose([
            TorchT.ToTensorV2()
        ])

    def __call__(self, image: Union[np.ndarray, Image], mask: Union[np.ndarray, Image]) -> Dict[str,  Union[np.ndarray, Image]]:
        # image = self.voxel_value_transform(image=image)["image"]
        transformed = self.geometric_transform(image=image, mask=mask)
        transformed = self.final_transform(image=transformed["image"], mask=transformed["mask"])

        return {
            "image": transformed["image"],
            "mask": transformed["mask"]
        }

if __name__ == '__main__':
    ds = AgriFieldDataset('data/source', transform=BaselineTransfrom(im_size=16), train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    fids, imgs, masks, targets = next(loader)