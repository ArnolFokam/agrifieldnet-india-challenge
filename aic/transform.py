from typing import Dict, Union

import numpy as np
import PIL

class BaselineTransfrom:
    def __init__(self) -> None:
        pass

    def __call__(self, image: Union[np.ndarray, PIL.Image], mask: Union[np.ndarray, PIL.Image]) -> Dict[str,  Union[np.ndarray, PIL.Image]]:
        return image, mask

if __name__ == '__main__':
    img, mask = None
    transform = BaselineTransfrom()

    # TODO: test transfrom for both nd array and torch tensor