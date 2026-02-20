#Module to store functions for creating dataloaders. 

#Imports 
import os
import sys
import glob
import awkward as ak
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv as gcn
from torch_geometric.nn import EdgeConv 
from torch_geometric.nn import DynamicEdgeConv 

#Global seeds (I actually have no idea whether I need to declare this in all module scripts)
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

#Dataloader creation. Function takes argument of a training and validation directory path, returns a training loader and validation loader. The repositories and splits have to be done on their own in a different script. There is a Jupyter Notebook in the pre-processing directory that can handle this. 
def CreateDataLoaders(training_directory, validation_directory, batch_size_ = 500, drop_last_ = True):
    
    #These lines assume that our signal files start with the char m (which is true by default in signal pre-processing)
    #They also assume that our background files start with the char b (which is true by default in background pre-processing)
    #This will need to be updated if someone changes naming conventions
    signal_training_paths = glob.glob(os.path.join(training_directory, "m*"))
    bkgs_training_paths = glob.glob(os.path.join(training_directory, "b*"))
    signal_validation_paths = glob.glob(os.path.join(validation_directory, "m*"))
    bkgs_validation_paths = glob.glob(os.path.join(validation_directory, "b*"))

    #Load the signal training files
    print("Beginning to load training files")
    signal_training_file_list = []
    for path in signal_training_paths:
        signal_training_file_list.append((torch.load(path, weights_only = False)))
        print(f"Loaded: {path}")

    #Load the background training files
    training_background = torch.load(bkgs_training_paths[0], weights_only = False)
    print("Loaded all training files")
    
    #Concatenating our training files, creating a training loader
    full_dataset_list_training = training_background + signal_training_file_list
    full_dataset_training = ConcatDataset(full_dataset_list_training)
    training_loader = DataLoader(full_dataset_training, batch_size = batch_size_, drop_last = drop_last_, shuffle=True, num_workers=1)
    print(f'Training on {len(training_loader)*training_loader.batch_size} total graphs')

    #Now loading the signal validation files (for ROC and validation)
    print("Beginning to load validation files")
    signal_val_file_list = []
    for path in signal_validation_paths:
        signal_val_file_list.append((torch.load(path, weights_only = False)))
        print(f"Loaded: {path}")

    #Now loading the background validation files
    val_background = torch.load(bkgs_validation_paths[0], weights_only = False)

    #Concatenating our validation files, creating a validation loader
    full_dataset_list_val = val_background + signal_val_file_list
    full_dataset_val = ConcatDataset(full_dataset_list_val)
    validation_loader =  DataLoader(full_dataset_val, batch_size = batch_size_, drop_last = drop_last_, shuffle=True, num_workers=1)

    return(training_loader, validation_loader)








    