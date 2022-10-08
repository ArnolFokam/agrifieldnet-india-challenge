"""
Credits: https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/utils.py
"""

import argparse
from sklearn.metrics import precision_recall_fscore_support

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

from aic.dataset import AgriFieldDataset
from aic.model import CropClassifier
from aic.transform import BaselineTransfrom


parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o','--output_dir', help='save path for output submission file', default='results', type=str)

# experiment
parser.add_argument('-s','--seed', help='seed for experiments', default=42, type=int)
parser.add_argument('-ts','--test_size', help='test size for cross validation', default=0.15, type=float)
parser.add_argument('-ks','--splits', help='number of splits for cross validation', default=10, type=int)

# data
parser.add_argument('-d','--data_dir', help='path to data folder', default='data/source', type=str)
parser.add_argument('-dd','--download_data', help='should we download the data?', default=False, type=bool)
parser.add_argument('-b','--batch_size', help='batch size', default=64, type=int)
parser.add_argument('-w','--num_workers', help='number of workers for dataloader', default=8, type=int)
parser.add_argument('-cs','--crop_size', help='size of the crop image after transform', default=32, type=int)

# model architeture
parser.add_argument('-ft', '--filters', help='list of filters for the CNN used', default=[32], type=list)

# model optimization & training
parser.add_argument('-ep','--epochs', help='number of training epochs', default=10, type=int)
parser.add_argument('-lr','--learning_rate', help='learning rate', default=1e-2, type=float)

args = parser.parse_args()

dataset = AgriFieldDataset(args.data_dir,  
                           download=args.download_data,
                           save_cache=True, 
                           train=True,
                           transform=BaselineTransfrom(crop_size=args.crop_size))

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    kfold = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.seed)
    
    for kfold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset.field_ids, dataset.targets)):
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        
        train_classes_weights = AgriFieldDataset.get_class_weights(train_ds.dataset.targets[train_ds.indices])
        train_classes_weights_inverted = { k: 1 / v for k, v in train_classes_weights.items()}
        dataloaders = {
            "train": DataLoader(train_ds, 
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 sampler=WeightedRandomSampler(
                                      weights=[train_classes_weights_inverted[target]  for target in train_ds.dataset.targets[train_ds.indices]],
                                      num_samples=len(train_ds),
                                      replacement=True)),
            "val": DataLoader(val_ds, 
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)
        }
        
        # model
        model = CropClassifier(n_classes=len(train_classes_weights.keys()), 
                               n_bands=len(train_ds.dataset.selected_bands), 
                               filters=args.filters)
        model = model.to(device)
        
        # loss function
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(list(train_classes_weights_inverted.values())))
        criterion.to(device)
        
        #initialize optimizer and scheduler each cycle
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10*len(dataloaders['train']))
        
        for epoch in range(args.epochs):
            print('\nKFold-{} Epoch {}/{}'.format(kfold_idx, epoch, args.epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_preds = []
                running_targets = []
                
                pbar = tqdm(dataloaders[phase])
                pbar.set_description(f"Phase {phase}")

                # Iterate over data.
                for field_ids, imgs, masks, targets in pbar:
                    field_ids = field_ids.to(device)
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    targets = targets.to(device)
                    

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(imgs, masks)
                        preds = torch.argmax(outputs, 1)
                        loss = criterion(outputs, targets)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    
                    # statistics
                    running_loss += loss.item()
                    running_targets.extend(targets.cpu().detach().numpy())
                    running_preds.extend(preds.cpu().detach().numpy())

                epoch_loss = running_loss / len(dataloaders[phase])
                
                print('{}:::: '
                      'Loss: {:.4f} '
                      'Prec: {:.4f} '
                      'Rec: {:.4f} '
                      'F1: {:.4f}'.format(phase, epoch_loss, *precision_recall_fscore_support(running_targets, running_preds, average='micro')[:3]))

        