"""
Credits: https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/utils.py
"""

import os
import copy
import time
import pickle
import logging
import argparse
import datetime
from argparse import Namespace

import yaml
import torch
import wandb
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
from aic.model import CropClassifier, PretrainedClassifer
from aic.dataset import AgriFieldDataset
from aic.transform import BaselineTrainTransform
from aic.helpers import seed_everything, get_dir, reset_wandb_env, generate_random_string


parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o', '--output_dir',
                    help='save path for trained models', default='results', type=str)

# experiment
parser.add_argument(
    '-n', '--name', help='name of experiment', default="agrifield-challenge", type=str)
parser.add_argument(
    '-off', '--offline', help='should we run the experiments offline?', default=True, type=bool)
parser.add_argument(
    '-s', '--seed', help='seed for experiments', default=42, type=int)
parser.add_argument('-ts', '--test_size',
                    help='test size for cross validation', default=0.17, type=float)
parser.add_argument('-dv', '--device',
                    help='cuda device to use', default=0, type=int)
parser.add_argument(
    '-ks', '--splits', help='number of splits for cross validation', default=10, type=int)
parser.add_argument(
    '-p', '--predict', help='predict the classes for the test data in a submission file', default=True, type=bool)
parser.add_argument('-ssp', '--sample_submission_path', help='path to the sample submssion path',
                    default='data/source/SampleSubmission.csv', type=str)

# data
parser.add_argument('-d', '--data_dir',
                    help='path to data folder', default='data/source', type=str)
parser.add_argument('-dd', '--download_data',
                    help='should we download the data?', default=False, type=bool)
parser.add_argument('-b', '--batch_size',
                    help='batch size', default=256, type=int)
parser.add_argument('-w', '--num_workers',
                    help='number of workers for dataloader', default=8, type=int)
parser.add_argument('-cs', '--crop_size',
                    help='size of the crop image after transform', default=32, type=int)
parser.add_argument('-bd', '--bands', help='bands to use for our training',
                    default='B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12', type=str)
parser.add_argument('-vi', '--vegetative_indeces', help='vegetative indeces to use',
                    default='NDVI NDWI_GREEN NDWI_BLUE', type=str)

# model architeture
parser.add_argument('-ft', '--filters', help='list of filters for the CNN used',
                    default="64 64 64", type=str)
parser.add_argument('-k', '--kernel_size',
                    help='kernel size for the convolutions', default=3, type=int)

# model optimization & training
parser.add_argument('-ep', '--epochs',
                    help='number of training epochs', default=10, type=int)
parser.add_argument('-lr', '--learning_rate',
                    help='learning rate', default=0.1, type=float)
parser.add_argument(
    '-c', '--cycles', help='trainin cycle for the model snapshot', default=5, type=int)

parser.add_argument(
    '-sp', '--sweep_path', help='path to sweep configuration if we wish to start a sweep', default=None, type=str)

parser.add_argument(
    '-sc', '--sweep_count', help="number of runs to makefor the sweep", default=None, type=int)

initial_args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')


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
            preds = torch.argmax(F.softmax(outputs, 1), 1)
            loss = criterion(outputs, targets)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
                scheduler.step()

        # metrics
        running_loss += loss.item()
        running_targets.extend(targets.cpu().detach().numpy())
        running_preds.extend(preds.cpu().detach().numpy())

    return running_loss, running_preds, running_targets


def train_model_snapshot(model,
                         criterion,
                         learning_rate,
                         dataloaders,
                         device,
                         num_cycles,
                         num_epochs_per_cycle,
                         num_classes,
                         kfold_idx,
                         logger):

    # time training
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('Inf')
    models_weights = []

    for cycle in range(num_cycles):
        # initialize optimizer and scheduler each cycle
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 10*len(dataloaders['train']))

        for epoch in range(num_epochs_per_cycle):
            print()
            logging.info('Fold {}: Cycle {}: Epoch {}/{}'.format(kfold_idx +
                         1, cycle + 1, epoch + 1, num_epochs_per_cycle))
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
                accuracy = accuracy_score(running_targets, running_preds)
                precision, recall, f1 = precision_recall_fscore_support(
                    running_targets, running_preds, average='micro')[:3]

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
                                        accuracy,
                                        precision,
                                        recall,
                                        f1))

                logger.log({
                    "cycle": cycle + 1,
                    "epoch": (cycle * num_epochs_per_cycle) + epoch + 1,
                    f"{phase}-loss": epoch_loss,
                    f"{phase}-accuracy": accuracy,
                    f"{phase}-precision": precision,
                    f"{phase}-recall": recall,
                    f"{phase}-f1": f1
                })

                # copy the model with the best validation loss as the best model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())

        # copy the best model to snapshot ensemble
        models_weights.append(copy.deepcopy(best_model_weights))

    print()

    ensemble_loss = 0.0
    ensemble_preds = []
    ensemble_targets = []

    # predict on validation using snapshots
    pbar = tqdm(dataloaders['val'])
    pbar.set_description(
        f"Fold {kfold_idx + 1}: Validating snapshots on validation data")

    # Iterate over data.
    for field_ids, imgs, masks, targets in pbar:
        field_ids = field_ids.to(device)
        imgs = imgs.to(device)
        masks = masks.to(device)
        targets = targets.to(device)

        # forward
        # track history if only in train
        prob = torch.zeros(
            (imgs.shape[0], num_classes), dtype=torch.float32).to(device)
        for weights in models_weights:
            model.load_state_dict(weights)
            model.eval()
            outputs = model(imgs, masks)
            prob += F.softmax(outputs, dim=1)

        prob /= len(models_weights)
        preds = torch.argmax(prob, 1)
        loss = F.nll_loss(torch.log(prob), targets)

        # ensemble metrics
        ensemble_loss += loss.item()  # * imgs.size(0)
        ensemble_targets.extend(targets.cpu().detach().numpy())
        ensemble_preds.extend(preds.cpu().detach().numpy())

    ensemble_loss /= len(dataloaders['val'])
    ensemble_accuracy = accuracy_score(ensemble_targets, ensemble_preds)
    ensemble_precision, ensemble_recall, ensemble_f1 = precision_recall_fscore_support(
        ensemble_targets, ensemble_preds, average='micro')[:3]

    time_elapsed = time.time() - since
    logging.info('Fold {}: Training complete in {:.0f}m {:.0f}s'.format(
        kfold_idx + 1, time_elapsed // 60, time_elapsed % 60))
    logging.info('Fold {}: Ensemble Loss : {:4f}, Best cycle val Loss: {:4f}'.format(
        kfold_idx + 1, ensemble_loss, best_loss))

    logger.log({
        "ensemble-loss": ensemble_loss,
        "ensemble-best-cycle-loss": best_loss,
        "ensemble-accuracy": ensemble_accuracy,
        "ensemble-precision": ensemble_precision,
        "ensemble-recall": ensemble_recall,
        "ensemble-f1": ensemble_f1
    })

    print()

    # load snapshot model weights and combine them in array
    best_models = []
    for weights in models_weights:
        model.load_state_dict(weights)
        best_models.append(model)

    return best_models, ensemble_loss, best_loss, ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1


def main():
    sweep_run_name = f"{datetime.datetime.now().strftime(f'%H-%M-%ST%d-%m-%Y')}_{generate_random_string(5)}"

    # directory to save models and parameters
    results_dir = get_dir(f'{initial_args.output_dir}/{sweep_run_name}')

    # combine wwandb config with args to form old args (sweep)
    # dumb init to get configs
    wandb.init(dir=get_dir(initial_args.output_dir))
    args = Namespace(**(vars(initial_args) | dict(wandb.config)))
    wandb.join()

    # save hyperparameters
    with open(f'{results_dir}/hparams.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

    logging.info(f'Preparing dataset...')

    seed_everything(args.seed)
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # get bands and filters from string
    args.bands = args.bands.split(' ')
    args.vegetative_indeces = args.vegetative_indeces.split(' ')
    args.filters = [int(f) for f in args.filters.split(' ')]

    dataset = AgriFieldDataset(args.data_dir,
                               args.bands,
                               vegetative_indeces=args.vegetative_indeces,
                               download=args.download_data,
                               save_cache=True,
                               train=True,
                               transform=BaselineTrainTransform(bands=args.bands,
                                                                vegetative_indeces=args.vegetative_indeces,
                                                                crop_size=args.crop_size))
    kfold = StratifiedShuffleSplit(
        n_splits=args.splits, test_size=args.test_size, random_state=args.seed)

    # arrays of model from cross validation of each snapshots
    models = []
    loss_folds = []
    best_loss_folds = []
    accuracy_folds = []
    precision_folds = []
    recall_folds = []
    f1_folds = []

    for kfold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset.field_ids, dataset.targets)):

        logging.info(
            f'Fold {kfold_idx + 1}: {len(train_indices)} trains, {len(val_indices)} vals')

        logging.info(f'Fold {kfold_idx + 1}: Loading dataset')

        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)

        train_classes_weights = AgriFieldDataset.get_class_weights(
            train_ds.dataset.targets[train_ds.indices])
        train_classes_weights_inverted = {
            k: 1 / v for k, v in train_classes_weights.items()}
        dataloaders = {
            "train": DataLoader(train_ds,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampler=WeightedRandomSampler(
                                    weights=[train_classes_weights_inverted[target]
                                             for target in train_ds.dataset.targets[train_ds.indices]],
                                    num_samples=len(train_ds),
                                    replacement=True)),
            "val": DataLoader(val_ds,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
        }

        # model
        logging.info(f'Fold {kfold_idx + 1}: preparing model...')
        model = PretrainedClassifer(n_classes=dataset.num_classes,
                                    n_channels=len(
                                        train_ds.dataset.selected_bands) + len(train_ds.dataset.vegetative_indeces),
                                    filters=args.filters,
                                    kernel_size=args.kernel_size)
        model = model.to(device)

        # loss function
        logging.info(f'Fold {kfold_idx + 1}: preparing loss function...')
        # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(list(train_classes_weights_inverted.values()))) hurts performance
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

        # reset wandb env
        reset_wandb_env()

        # wandb configs
        run = wandb.init(project=args.name,
                         name=f'kfold_{kfold_idx + 1}',
                         group=sweep_run_name,
                         dir=get_dir(args.output_dir),
                         config=args,
                         reinit=True)

        # get a snapshot of model for this k fold
        logging.info(f'Fold {kfold_idx + 1}: getting model snapshots...')
        best_models, loss, best_loss, accuracy, precision, recall, f1 = train_model_snapshot(model,
                                                                                             criterion,
                                                                                             args.learning_rate,
                                                                                             dataloaders,
                                                                                             device,
                                                                                             num_cycles=args.cycles,
                                                                                             num_epochs_per_cycle=args.epochs,
                                                                                             num_classes=dataset.num_classes,
                                                                                             kfold_idx=kfold_idx,
                                                                                             logger=run)

        loss_folds.append(loss)
        best_loss_folds.append(best_loss)
        accuracy_folds.append(accuracy)
        precision_folds.append(precision)
        recall_folds.append(recall)
        f1_folds.append(f1)
        models.extend(best_models)

        wandb.join()

    sweep_run = wandb.init(project=f"{args.name}-sweeps",
                           name=sweep_run_name,
                           config=args,
                           dir=get_dir(args.output_dir))

    sweep_run.log({
        "loss": sum(loss_folds) / len(loss_folds),
        "best-loss": sum(best_loss_folds) / len(best_loss_folds),
        "accuracy": sum(accuracy_folds) / len(accuracy_folds),
        "precision": sum(precision_folds) / len(precision_folds),
        "recall": sum(recall_folds) / len(recall_folds),
        "f1": sum(f1_folds) / len(f1_folds)
    })
    wandb.join()

    # save models
    for i, m in enumerate(models):
        torch.save(m.state_dict(), f"{results_dir}/model_{i}.pth")

    if args.predict:
        logging.info(f'Predict output of test data...')

        test_dataset = AgriFieldDataset(args.data_dir,
                                        bands=args.bands,
                                        vegetative_indeces=args.vegetative_indeces,
                                        download=args.download_data,
                                        save_cache=True,
                                        train=False,
                                        transform=BaselineTrainTransform(bands=args.bands, vegetative_indeces=args.vegetative_indeces, crop_size=args.crop_size))
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8)
        preds = predict(models, test_loader, device,
                        num_classes=test_dataset.num_classes)

        preds = np.concatenate(
            (test_dataset.field_ids[..., np.newaxis], preds), axis=-1, dtype=object)
        preds = pd.DataFrame(preds, columns=[
                             'Field_ID', *['Crop_ID_%d' % (i+1) for i in range(test_dataset.num_classes)]])
        preds = preds.groupby('Field_ID').mean()

        # make a submission
        sub = pd.read_csv(args.sample_submission_path)
        sub['Field_ID'] = np.unique(test_dataset.field_ids)

        for i in range(test_dataset.num_classes):
            sub.iloc[:, i + 1] = preds['Crop_ID_%d' % (i+1)].tolist()

        sub.to_csv(os.path.join(
            results_dir, 'submission.csv'), index=False)
        logging.info(
            f'Submission saved at {os.path.join(results_dir, "submission.csv")}')


if __name__ == "__main__":
    if initial_args.sweep_path:

        import yaml
        with open(initial_args.sweep_path, "r") as stream:
            try:
                sweep_configuration = yaml.safe_load(stream)
                sweep_id = wandb.sweep(
                    sweep=sweep_configuration, project=initial_args.name)
                wandb.agent(sweep_id, function=main,
                            project=initial_args.name, count=initial_args.sweep_count)

            except yaml.YAMLError as exc:
                logging.error(
                    f"Couldn't load the sweep file. Make sure {initial_args.sweep_path} is a valid path")
                logging.warn("doing a normal run")
                main()
    else:
        main()
