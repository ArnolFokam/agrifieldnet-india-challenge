from PIL.Image import Image
from typing import Dict, Union
import numpy as np
import cv2
import torch

from aic.dataset import AgriFieldDataset

class RandomFieldAreaCrop:
    def __init__(self, crop_size: int) -> None:
        self.crop_size = crop_size

    def __call__(self, image: Union[np.ndarray, Image], mask: Union[np.ndarray, Image]) -> Dict[str,  Union[np.ndarray, Image]]:
        original_height, original_width = image.shape[:2]
        
        assert self.crop_size <= original_height and self.crop_size <= original_height, "crop area should not be bigger than image"
        
        contours = cv2.findContours(mask.astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]
        # assert len(contours) == 1, "Only one contour should be detected (a.k.a field area)"
        
        # get the bounding box coordinates of 
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # TODO: create crops such field area that it not always at the center
        # randomly sample the center our the field of view
        
        # get the center of the image
        cx = x + (w // 2)
        cy = y + (h // 2)
        
        # get the desired cropped portion of the image
        left = cx - self.crop_size // 2
        right = cx + self.crop_size // 2
        top = cy - self.crop_size // 2
        bottom = cy + self.crop_size // 2
        
        # ensure that the crop portion stays in the image
        space_left = self.crop_size // 2 if left >= 0 else cx # calculate remaining x-space from left
        x1 = cx - space_left - max(0, right - original_width) # add the right surplus to the left
        
        space_right = self.crop_size // 2 if right <= original_width else (right - cx) # calculate remaining x-space from left
        x2 = cx + space_right - min(0, left) # add the left offset to the right
        
        space_top = self.crop_size // 2 if top >= 0 else cy # calculate remaining top-space from left
        y1 = cy - space_top - max(0, bottom - original_height) # add the bottom surplus to the top
        
        space_bottom = self.crop_size // 2 if bottom <= original_height else (bottom - cy) # calculate remaining bottom-space from left
        y2 = cy + space_bottom - min(0, top) # add the top offset to the bottom
        
        return {
            "image": image[y1:y2, x1:x2, :],
            "mask": mask[y1:y2, x1:x2],
        }