import math
from typing import List

import torch
import torch.nn as nn

from aic.transform import BaselineTransfrom
from aic.dataset import AgriFieldDataset

class CropClassifier(nn.Module):
    def __init__(self, n_classes: int, n_bands: int, filters: List[int] = [32]) -> None:
        super().__init__()
        
        assert len(filters) > 0, "[Input error] the model must have at least one filter layer"
        
        self.conv_layers = [self.conv_layer(n_bands, filters[0])]
        
        for i in range(len(filters) - 1):
            self.conv_layers.append(self.conv_layer(filters[i], filters[i + 1]))
        
        self.fc = nn.Linear(filters[-1], n_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.05)
            if module.bias is not None:
                module.bias.data.zero_()
        
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        
        
    def conv_layer(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.GroupNorm(2, out_channels),
            nn.ReLU()
        )
        
    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        out = self.conv_layers[0](image) # input convolution
        
        # pass output through successive cnns
        for i in range(1, len(self.conv_layers)):
            out = self.conv_layers[i](out)
        
        # mask the unwanted pixel features
        mask = mask.view(mask.size(0), 1, mask.size(-2), mask.size(-1)) # expand mask dim to match conv outpu
        out = (out*mask).sum((-2, -1)) / mask.sum((-2, -1))
        
        # logits
        out = self.fc(out)
        
        return out
        

if __name__ == '__main__':
    ds = AgriFieldDataset('data/source', transform=BaselineTransfrom(im_size=16), train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    fids, imgs, masks, targets = next(loader) 
    
    # modelling
    model = CropClassifier(n_classes=len(ds.class_meta.keys()),
                           n_bands=len(ds.selected_bands),
                           filters=[32, 64, 128])
    
    out = model(imgs, masks)   