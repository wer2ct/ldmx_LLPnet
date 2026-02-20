#Runs and instance of training for a chosen model. python3 training_instance.py <training_path> <validation_path> <out_path>

#Imports 
import sys
import os
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

#Module Imports
sys.path.append('modules') #direct to our module package
import architectures as arch
import training_dataloaders
import training_utils


#Global seeds (I actually have no idea whether I need to declare this in all module scripts)
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

#run our training instance
def main():
    #grab arguments for paths. 
    training_path = sys.argv[1]
    validation_path = sys.argv[2]
    output_path = sys.argv[3]

    #Declare our dataloaders
    print("Loading our loaders")
    training_loader, validation_loader = training_dataloaders.CreateDataLoaders(training_path, 
                                                                                validation_path, 
                                                                                batch_size_ = 500, 
                                                                                drop_last_ = True)
    #Declare our classifier
    print("Declaring our classifier")
    classifier = arch.GNN_v3_dynamic(in_channels = 4, 
                                     hc1 = 10, hc2 = 20, hc3 = 40, hc4 = 50, 
                                     fc1 = 25, fc2 = 12, fc3 = 6, 
                                     k1 = 33, k2 = 25, k3 = 17, k4 = 9,
                                     out_channels = 2)
    #Run training
    print("Beginning Training (!!)")
    training_utils.run_train(model = classifier,
                             training_loader_ = training_loader, 
                             validation_loader_ = validation_loader, 
                             log_dir = output_path, 
                             log_prefix = 'module_test', 
                             optimizer = 'Adam', 
                             lr = 0.001, 
                             max_epochs_ = 10,
                             use_scheduler = True) #this scheduler can be configured by hand. Default is true, minimizes validation loss. 
    

main()
    








    