"""
Credits: https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/utils.py
"""

import os
import copy
import time
import pickle
import logging
import argparse

import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from aic.utils import predict
from aic.model import CropClassifier
from aic.dataset import AgriFieldDataset
from aic.transform import BaselineTransfrom
from aic.helpers import seed_everything, get_dir, generate_random_string


parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o','--output_dir', help='save path for trained models', default='results', type=str)

# experiment
parser.add_argument('-s','--seed', help='seed for experiments', default=42, type=int)
parser.add_argument('-ts','--test_size', help='test size for cross validation', default=0.13, type=float)
parser.add_argument('-ks','--splits', help='number of splits for cross validation', default=10, type=int)
parser.add_argument('-p','--predict', help='predict the classes for the test data in a submission file', default=True, type=bool)
parser.add_argument('-ssp','--sample_submission_path', help='path to the sample submssion path', default='data/source/SampleSubmission.csv', type=str)

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
parser.add_argument('-ep','--epochs', help='number of training epochs', default=10, type=int)
parser.add_argument('-lr','--learning_rate', help='learning rate', default=0.1, type=float)
parser.add_argument('-c','--cycles', help='trainin cycle for the model snapshot', default=5, type=int)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')

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

def train_model_snapshot(model, criterion, learning_rate, dataloaders, device, num_cycles, num_epochs_per_cycle, num_classes, kfold_idx):
    
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
            print()
            logging.info('Fold {}: Cycle {}: Epoch {}/{}'.format(kfold_idx + 1, cycle + 1, epoch + 1, num_epochs_per_cycle))
            print('-' * 20)

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
                
                logging.info(
                    'Fold {}: '
                    'Cycle {}: '
                    'Epoch {}/{}: '
                    'Phase {}: '
                    'Loss: {:.6f} '
                    'Acc {:.6f} '
                    'Prec: {:.6f} '
                    'Rec: {:.6f} '
                    'F1: {:.6f}'.format(kfold_idx + 1, 
                                        cycle + 1, 
                                        epoch + 1, num_epochs_per_cycle,
                                        phase, 
                                        epoch_loss,
                                        accuracy_score(running_targets, running_preds),
                                        *precision_recall_fscore_support(running_targets,running_preds, average='micro')[:3]))
                
                # copy the model with the best validation loss as the best model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
        
        # copy the best model to snapshot ensemble
        models_weights.append(copy.deepcopy(best_model_weights))
        
    print()
    
    ensemble_loss = 0.0
    
    #predict on validation using snapshots
    pbar = tqdm(dataloaders['val'])
    pbar.set_description(f"Fold {kfold_idx + 1}: Validating snapshots on validation data")

    # Iterate over data.
    for field_ids, imgs, masks, targets in pbar:
        field_ids = field_ids.to(device)
        imgs = imgs.to(device)
        masks = masks.to(device)
        targets = targets.to(device)

        # forward
        # track history if only in train
        prob = torch.zeros((imgs.shape[0], num_classes), dtype=torch.float32).to(device)
        for weights in models_weights:
            model.load_state_dict(weights)
            model.eval()
            outputs = model(imgs, masks)
            prob += F.softmax(outputs, dim=1)
        
        prob /= num_cycles
        loss = F.nll_loss(torch.log(prob), targets)    
        ensemble_loss += loss.item() # * imgs.size(0)
    
    ensemble_loss /= len(dataloaders['val'])

    time_elapsed = time.time() - since
    logging.info('Fold {}: Training complete in {:.0f}m {:.0f}s'.format(kfold_idx + 1, time_elapsed // 60, time_elapsed % 60))
    logging.info('Fold {}: Ensemble Loss : {:4f}, Best cycle val Loss: {:4f}'.format(kfold_idx + 1, ensemble_loss, best_loss))
    
    print()
    
    # load snapshot model weights and combine them in array
    best_models = []
    for weights in models_weights:
        model.load_state_dict(weights)   
        best_models.append(model) 
    
    return best_models, ensemble_loss, best_loss

if __name__ == "__main__":
    results_dir = get_dir(f'{args.output_dir}/{generate_random_string()}')
    
    logging.info(f'Preparing dataset...')
    
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = AgriFieldDataset(args.data_dir, 
                               bands=args.bands, 
                               download=args.download_data,
                               save_cache=True, 
                               train=True,
                               transform=BaselineTransfrom(bands=args.bands, crop_size=args.crop_size))
    kfold = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.seed)
    
    # arrays of model from cross validation of each snapshots
    models = []
    
    for kfold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset.field_ids, dataset.targets)):
        
        logging.info(f'Fold {kfold_idx + 1}: {len(train_indices)} trains, {len(val_indices)} vals')
        
        logging.info(f'Fold {kfold_idx + 1}: Loading dataset')
        
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
        logging.info(f'Fold {kfold_idx + 1}: preparing model...')
        model = CropClassifier(n_classes=dataset.num_classes, 
                               n_bands=len(train_ds.dataset.selected_bands), 
                               filters=args.filters,
                               kernel_size=args.kernel_size)
        model = model.to(device)
        
        # loss function
        logging.info(f'Fold {kfold_idx + 1}: preparing loss function...')
        # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(list(train_classes_weights_inverted.values()))) hurts performance
        criterion = nn.CrossEntropyLoss()    
        criterion.to(device)
        
        # get a snapshot of model for this k fold
        logging.info(f'Fold {kfold_idx + 1}: getting model snapshots...')
        best_models, _, _ = train_model_snapshot(model,
                                                 criterion,
                                                 args.learning_rate,
                                                 dataloaders,
                                                 device,
                                                 num_cycles=args.cycles,
                                                 num_epochs_per_cycle=args.epochs,
                                                 num_classes=dataset.num_classes,
                                                 kfold_idx=kfold_idx)
        models.extend(best_models)
        
        # save model
        with open(f'{results_dir}/models.pkl', 'wb') as f:
            pickle.dump(models, f)
            
        # save hyperparameters
        with open(f'{results_dir}/hparams.yaml', 'w') as f:
            yaml.dump(args.__dict__, f)
        
        if args.predict:
            logging.info(f'Predict output of test data...')
            
            # TODO: load model from path if given or load from previous training or throw error
            test_test_dataset = AgriFieldDataset(args.data_dir,
                                       bands=args.bands,
                                       download=args.download_data,
                                       save_cache=True,
                                       train=False,
                                       transform=BaselineTransfrom(bands=args.bands, crop_size=args.crop_size))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
            preds = predict(models, test_loader, device, num_classes=test_dataset.num_classes)
            
            preds = np.concatenate((test_dataset.field_ids[..., np.newaxis], preds), axis=-1, dtype=object)
            preds = pd.DataFrame(preds, columns=['Field_ID', *['Crop_ID_%d'%(i+1) for i in range(test_dataset.num_classes)]])
            preds = preds.groupby('Field_ID').mean()
            
            # make a submission
            sub = pd.read_csv(args.sample_submission_path)
            sub['Field_ID'] = np.unique(test_dataset.field_ids)
            
            for i in range(test_dataset.num_classes):
                sub.iloc[:, i + 1] = preds['Crop_ID_%d'%(i+1)].tolist()

            sub.to_csv(os.path.join(results_dir, 'submission.csv'), index = False)
            logging.info(f'Submission saved at {os.path.join(results_dir, "submission.csv")}')

        
