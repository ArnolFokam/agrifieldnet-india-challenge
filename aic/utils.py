import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def predict(models, dataloader, device, num_classes):
    # [CAUTION]: data loader should be have a batch size of 1
    res = np.zeros((len(dataloader), num_classes), dtype = np.float32)
    
    for model in models:
        
        model.eval()
        results = []
        
        pbar = tqdm(dataloader)
        pbar.set_description(f"Predicting test data with snapshots")
        
        for field_ids, imgs, masks, _ in pbar:
            field_ids = field_ids.to(device)
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # forward
            with torch.set_grad_enabled(False):
                outputs = F.softmax(model(imgs, mask), dim=1)    
                results.append(outputs.detach().cpu().numpy())
                
        results = np.concatenate(results, axis=0)
        
        res += results
    return res / len(models)