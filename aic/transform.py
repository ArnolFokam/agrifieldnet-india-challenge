from typing import Dict, Union

import torch
import numpy as np
import PIL
import cv2
import albumentations as A


class FieldAreaCrop:
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, image: Union[np.ndarray, PIL.Image], mask: Union[np.ndarray, PIL.Image]) -> Dict[str,  Union[np.ndarray, PIL.Image]]:
        contours = cv2.findContours(np.array(mask).astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]
        assert len(contours) == 1, "Only one contour should be detected (a.k.a field area)"

        x, y, w, h = cv2.boundingRect(contours[0])
        

class BaselineTransfrom:
    def __init__(self) -> None:
        # transform that change the value of a voxel in the spectral bands
        self.voxel_value_transform = A.Compose([])

        # transform that changes the geometric shape of the image (rotation, translation, etc)
        self.geometric_transform = A.Compose([
            FieldAreaCrop()
        ])

        # transform after all the important ones, usually to convert to tensor
        self.final_transform = A.Compose([
            A.ToTensorV2()
        ])

    def __call__(self, image: Union[np.ndarray, PIL.Image], mask: Union[np.ndarray, PIL.Image]) -> Dict[str,  Union[np.ndarray, PIL.Image]]:
        image = self.voxel_value_transform(image=image)["image"]
        transformed = self.geometric_transform(image=image, mask=mask)
        transformed = self.final_transform(image=transformed["image"], mask=transformed["mask"])

        return {
            "image": transformed["image"],
            "mask": transformed["mask"]
        }

if __name__ == '__main__':
    im_size = 24
    img = torch.rand((im_size, im_size, 3))
    transform = BaselineTransfrom()