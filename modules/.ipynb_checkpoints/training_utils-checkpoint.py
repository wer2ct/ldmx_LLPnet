#Module to store a generic training loop and other training utility scripts 

#Imports 
import os
import sys
import csv
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

#A ~generic training loop for our GNN style models
def run_train(model: nn.Module, 
              training_loader_: DataLoader, 
              validation_loader_: DataLoader, 
              log_dir: str = '/scratch/wer2ct/February2026/v3_dynamic_files/GNN_output', 
              log_prefix: str = 'GNN_model_v3_dynamic_kv3', 
              optimizer: str = 'Adam', 
              lr: float = 0.001, 
              max_epochs_: int = 15,
              use_scheduler = True): #this scheduler can be configured by hand. Default is true, minimizes validation loss. 
    
    #Setup the device. Please use a GPU if you can! 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Using global SEED if available, though manual_seed_all is only necessary
        # if using multiple GPUs, which is not fully set up here.
        if 'SEED' in globals():
            torch.cuda.manual_seed_all(globals()['SEED'])
            
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} ({num_gpus} GPUs available)")

    # Move model to device (if GPU avail it'll move, if nothing will stay on the CPU)
    model = model.to(device)

    #Setup the loss and our optimizer
    criterion = nn.CrossEntropyLoss() #use cross entropy, we are doing a two output binary classification
    optimizer_fn = getattr(torch.optim, optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=lr)

    #set up our lr scheduler if flag is chosen
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min', 
                                                               factor=0.5, 
                                                               patience=2)

    # Initialize variables
    iteration = 0
    best_val_acc = 0.0

    #Setup the logging of loss and weights
    os.makedirs(log_dir, exist_ok=True)
    train_log_name = f'{log_dir}/{log_prefix}_train.csv'
    val_log_name = f'{log_dir}/{log_prefix}_val.csv'
    model_save_path = f'{log_dir}/{log_prefix}_best_model.pt'

    with open(train_log_name, 'w', newline='') as trainfile,  open(val_log_name, 'w', newline='') as valfile:
        train_writer = csv.writer(trainfile)
        val_writer = csv.writer(valfile)
        train_writer.writerow(['iter', 'epoch', 'loss'])
        val_writer.writerow(['iter', 'epoch', 'loss', 'accuracy'])

        max_epochs = max_epochs_
        epoch = 0

        #Training loop
        while epoch < max_epochs:
            epoch += 1
            # Training
            model.train()
            total_train_loss = 0.0
            print(f"\n--- Epoch {epoch}/{max_epochs} (Training) ---")
            for training_data in training_loader_: 
            
                training_data = training_data.to(device)
                
                optimizer.zero_grad()
                model_out = model(training_data)
                loss = criterion(model_out, training_data.y)

                #Backpropagation
                loss.backward()
                optimizer.step()

                #logging the training loss
                total_train_loss += loss.item()
                train_writer.writerow([iteration, epoch, loss.item()])
                iteration += 1

            avg_train_loss = total_train_loss / len(training_loader_)
            print(f"Epoch {epoch} finished. Avg Train Loss: {avg_train_loss:.4f}")

            #Post epoch validation
            model.eval()
            print(f"--- Epoch {epoch}/{max_epochs} (Validation) ---")
            
            total_correct = 0
            total_samples = 0
            total_val_loss = 0.0

            #Do not compute gradients when doing the validation
            with torch.no_grad():
                for validation_data in validation_loader_:
                    validation_data = validation_data.to(device)
                
                    out = model(validation_data) 

                    val_loss = criterion(out, validation_data.y)
                    total_val_loss += val_loss.item()
                    
                    n_correct = torch.sum(out.argmax(dim=1) == validation_data.y)
                    total_correct += n_correct.item()
                    total_samples += len(validation_data.y)

                #for logging purposes we want to keep track of the validation loss and accuracy
                acc = total_correct / total_samples
                avg_val_loss = total_val_loss / len(validation_loader_)
                if use_scheduler:
                    scheduler.step(avg_val_loss) #step the scheduler. 
        
                # Get current LR for logging
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {acc:.4f}, Val Loss {avg_val_loss:.4f}, LR:{current_lr:.6f}")
                
                val_writer.writerow([iteration, epoch, avg_val_loss, acc])
                
                #Save the best model (this is something I should've been doing a while ago...)
                if acc > best_val_acc:
                    best_val_acc = acc
                    print(f"--> New best model saved! Accuracy: {acc:.4f}")
                    torch.save(model.state_dict(), model_save_path)
                    
    print(f"\nTraining finished after {max_epochs} epochs. Best Validation Accuracy: {best_val_acc:.4f}")
    
    return(model)


    