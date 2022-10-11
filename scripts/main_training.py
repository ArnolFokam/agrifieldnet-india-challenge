"""
Credits: https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/utils.py
"""

import copy
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support

from aic.model import CropClassifier
from aic.helpers import seed_everything
from aic.dataset import AgriFieldDataset
from aic.transform import BaselineTransfrom


parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o','--output_dir', help='save path for output submission file', default='results', type=str)

# experiment
parser.add_argument('-s','--seed', help='seed for experiments', default=42, type=int)
parser.add_argument('-ts','--test_size', help='test size for cross validation', default=0.13, type=float)
parser.add_argument('-ks','--splits', help='number of splits for cross validation', default=10, type=int)

# data
parser.add_argument('-d','--data_dir', help='path to data folder', default='data/source', type=str)
parser.add_argument('-dd','--download_data', help='should we download the data?', default=False, type=bool)
parser.add_argument('-b','--batch_size', help='batch size', default=128, type=int)
parser.add_argument('-w','--num_workers', help='number of workers for dataloader', default=8, type=int)
parser.add_argument('-cs','--crop_size', help='size of the crop image after transform', default=32, type=int)
parser.add_argument('-bd','--bands', help='bands to use for our training', 
                    default=['B01', 'B02', 'B03', 'B04','B05','B06','B07','B08','B8A', 'B09', 'B11', 'B12'], nargs='+', type=str)


# model architeture
parser.add_argument('-ft', '--filters', help='list of filters for the CNN used', default=[64, 64, 64], nargs='+', type=int)
parser.add_argument('-k', '--kernel_size', help='kernel size for the convolutions', default=3, type=int)

# model optimization & training
parser.add_argument('-ep','--epochs', help='number of training epochs', default=50, type=int)
parser.add_argument('-lr','--learning_rate', help='learning rate', default=0.1, type=float)

args = parser.parse_args()

def train_val_single_epoch(model, criterion, optimizer, scheduler, dataloader, device, phase):
    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_preds = []
    running_targets = []

    pbar = tqdm(dataloader)
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
            
    return running_loss, running_preds, running_targets

def train_model_snapshot(model, criterion, learning_rate, dataloaders, device, num_cycles, num_epochs_per_cycle):
    
    # time training
    since = time.time()
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('Inf')
    models_weights = []
    
    for cycle in range(num_cycles):
        #initialize optimizer and scheduler each cycle
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10*len(dataloaders['train']))
        
        for epoch in range(num_epochs_per_cycle):
            
            print('\nCycle {}: KFold-{}: Epoch {}/{}'.format(cycle + 1, kfold_idx + 1, epoch + 1, num_epochs_per_cycle))
            print('-' * 15)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                
                running_loss, running_preds, running_targets = train_val_single_epoch(model,
                                                                                      criterion,
                                                                                      optimizer,
                                                                                      scheduler,
                                                                                      dataloaders[phase],
                                                                                      device,
                                                                                      phase)
                
                epoch_loss = running_loss / len(dataloaders[phase])
                
                print('{}:::: '
                      'Loss: {:.4f} '
                      'Prec: {:.4f} '
                      'Rec: {:.4f} '
                      'F1: {:.4f}'.format(phase, epoch_loss, 
                                          *precision_recall_fscore_support(running_targets, 
                                                                           running_preds, 
                                                                           average='micro')[:3]))
                
                # copy the model with the best validation loss as the best model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
        
        # copy the best model to snapshot ensemble
        models_weights.append(copy.deepcopy(best_model_weights.state_dict()))
    
    ensemble_loss = 0.0
    
    #predict on validation using snapshots
    pbar = tqdm(dataloaders['val'])
    pbar.set_description(f"Validating snapshots on validation data")

    # Iterate over data.
    for field_ids, imgs, masks, targets in pbar:
        field_ids = field_ids.to(device)
        imgs = imgs.to(device)
        masks = masks.to(device)
        targets = targets.to(device)

        # forward
        # track history if only in train
        prob = torch.zeros((inputs.shape[0], 7), dtype = torch.float32).to(device)
        for weights in models_weights:
            model.load_state_dict(weights)
            model.eval()
            outputs = model(imgs, masks)
            prob += F.softmax(outputs, dim=1)
        
        prob /= num_cycles
        loss = F.nll_loss(torch.log(prob), labels)    
        ensemble_loss += loss.item() * inputs.size(0)
    
    ensemble_loss /= len(dataloaders['val'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Ensemble Loss : {:4f}, Best cycle val Loss: {:4f}'.format(ensemble_loss, best_loss))
    
    # load snapshot model weights and combine them in array
    best_models = []
    for weights in models_weights:
        model.load_state_dict(weights)   
        best_models.append(model) 
    
    return best_models, ensemble_loss, best_loss

if __name__ == "__main__":
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = AgriFieldDataset(args.data_dir, 
                               bands=bands, 
                               download=args.download_data,
                               save_cache=True, 
                               train=True,
                               transform=BaselineTransfrom(bands=bands, crop_size=args.crop_size))
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
                               filters=args.filters,
                               kernel_size=args.kernel_size)
        model = model.to(device)
        
        # loss function
        # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(list(train_classes_weights_inverted.values()))) hurts performance
        criterion = nn.CrossEntropyLoss()    
        criterion.to(device)
        
        # get a snapshot of model for this k fold
        best_models, _, _ = train_model_snapshot(model_ft,
                                                 criterion,
                                                 args.learning_rate,
                                                 dataloaders,
                                                 device,
                                                 num_cycles=6,
                                                 num_epochs_per_cycle=args.epochs)
        models_arr.extend(best_models)

        
