"""
Credits: https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/utils.py
"""

import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

from aic.dataset import AgriFieldDataset
from aic.model import CropClassifier


parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o','--output_dir', help='save path for output submission file', default='results', type=str)

# experiment
parser.add_argument('-s','--seed', help='seed for experiments', default=42, type=int)
parser.add_argument('-ts','--test_size', help='test size for cross validation', default=0.15, type=float)
parser.add_argument('-ks','--splits', help='number of splits for cross validation', default=10, type=int)

# data
parser.add_argument('-b','--batch_size', help='batch size', default=64, type=int)
parser.add_argument('-d','--data_dir', help='path to data folder', default='data/source', type=str)
parser.add_argument('-w','--num_workers', help='number of workers for dataloader', default=8, type=int)

# model architeture
parser.add_argument('-ft', '--filters', help='list of filters for the CNN used', default=[32], type=list)

# model optimization & training
parser.add_argument('-ep','--epochs', help='number of training epochs', default=1e-2, type=int)
parser.add_argument('-lr','--learning_rate', help='learning rate', default=1e-2, type=float)

args = parser.parse_args()

dataset = AgriFieldDataset(args.data_dir,  save_cache=True, train=True)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    kfold = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.seed)
    
    for train_indices, val_indices in kfold.split(dataset.field_ids, dataset.targets):
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        
        train_classes, train_classes_weights = AgriFieldDataset.get_class_weights(train_ds.dataset.targets[train_ds.indices])
        train_classes_weights_inverted = 1. / np.array(list(train_classes_weights))
        dataloaders = {
            "train": DataLoader(train_ds, 
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 sampler=WeightedRandomSampler(train_classes_weights_inverted, len(train_classes_weights))),
            "val": DataLoader(val_ds, 
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)
        }
        
        # model
        model = CropClassifier(n_classes=len(train_classes), 
                               n_bands=len(train_ds.dataset.selected_bands), 
                               filters=args.filters)
        model = model.to(device)
        
        # loss function
        criterion = nn.CrossEntropyLoss(weight=train_classes_weights_inverted)
        
        #initialize optimizer and scheduler each cycle
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10*len(dataloaders['train']))
        
        for epoch in range(args.epochs):
            print('Epoch {}/{}'.format(epoch, args.epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, inputs_area, inputs_mask, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    inputs_mask = inputs_mask.to(device)
                    inputs_area = inputs_area.to(device)
                    labels = labels.to(device)
                    

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, inputs_mask)
                        preds = torch.argmax(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders["train"])
                epoch_acc = running_corrects.double() /  len(dataloaders["train"])
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

        