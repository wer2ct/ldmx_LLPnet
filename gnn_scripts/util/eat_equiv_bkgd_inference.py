#don't get too excited, this is basically just a scripted out jupyter notebook
#python3 eat_equiv_bkgd_inference.py

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


#Seeding and initialization of global vars:
sys.path.append("modules")
import architectures as arch

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

#grab the GPU if there. 
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Using device: {device} ({num_gpus} GPUs available)")

#Relevant Paths
DYNAMIC_MODEL_PATH   = "/standard/ldmxuva/gnn_files/run_output/dynamic_run_extended_best_model.pt"
STATIC_MODEL_PATH    = "/standard/ldmxuva/gnn_files/run_output/static_run_extended_best_model.pt"
BACKGROUND_DIRECTORY = '/standard/ldmxuva/gnn_files/note_stuff/eat_equiv_eot_graphs/processed/' #pull stuff from here
OUTDIR = '/standard/ldmxuva/gnn_files/note_stuff/eat_equiv_eot_results/' #dump stuff here.

def main():
    print("Preparing to load models")
    trained_model_state_dict_dynamic = torch.load(DYNAMIC_MODEL_PATH, weights_only=True)
    trained_model_state_dict_static  = torch.load(STATIC_MODEL_PATH,  weights_only=True)
    
    model_preloaded_dynamic = arch.GNN_v3_dynamic(
        in_channels=4,
        hc1=10, hc2=20, hc3=40, hc4=50,
        fc1=25, fc2=12, fc3=6,
        k1=33,  k2=25,  k3=17, k4=9,
        out_channels=2,
    )
    model_preloaded_static = arch.GNN_v3_static(
        in_channels=4,
        hc1=10, hc2=20, hc3=40, hc4=50,
        fc1=25, fc2=12, fc3=6,
        out_channels=2,
    )
    
    model_preloaded_static.load_state_dict(trained_model_state_dict_static)
    model_preloaded_static = model_preloaded_static.to(device)
    model_preloaded_static.eval()
    
    model_preloaded_dynamic.load_state_dict(trained_model_state_dict_dynamic)
    model_preloaded_dynamic = model_preloaded_dynamic.to(device)
    model_preloaded_dynamic.eval()
    print("models loaded, set to eval")

    #the infernece output loop. The basic idea of this is that it will run inference on digestable chunks of files, then output continuously until we are out of files
    #This sidesteps the issue of having to load everything into memory at once, which would be like ~2 TBs. 
    print("Beginning the inference process")

    #this is a massive loop in main, could break into functions but will let someone come through with claude do that. 
    background_paths = glob.glob(os.path.join(BACKGROUND_DIRECTORY, "e*"))
    i = 0
    file_list = []
    file_bunch_size = 500 #this should be safe with 100 GB memory 
    while i < len(background_paths):
    
        #this block runs our inference
        if (i % file_bunch_size == 0) and (i > 0):
            #identification
            print("we should create a data loader with these files")
            print(len(file_list))
    
            #create loader
            reform_list = [item for sublist in file_list for item in sublist]
            loader = DataLoader(reform_list, batch_size = 500, drop_last = False, shuffle=True, num_workers=1)
    
            #run inference, these get reset every time we hit this step.
            y_scores_dynamic = []
            y_scores_static  = []
            y_true           = []
            event_numbers    = []
            file_numbers     = []
    
            print(f'beginning inference on bunch {i}')
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
            
                    out_dynamic = model_preloaded_dynamic(data)
                    out_static  = model_preloaded_static(data)
            
                    probs_dynamic = torch.softmax(out_dynamic, dim=1)[:, 1]
                    probs_static  = torch.softmax(out_static,  dim=1)[:, 1]
            
                    y_scores_dynamic.append(probs_dynamic.cpu())
                    y_scores_static.append(probs_static.cpu())
                    y_true.append(data.y.cpu())
                    event_numbers.append(data.event_number.cpu())
                    file_numbers.append(data.file_number.cpu())
            
                    #ensure cleanup
                    del data, out_dynamic, out_static, probs_dynamic, probs_static
    
            #save off our stuff. 
            y_true = torch.cat(y_true).numpy()
            y_scores_static = torch.cat(y_scores_static).numpy()
            y_scores_dynamic = torch.cat(y_scores_dynamic).numpy()
            file_numbers = torch.cat(file_numbers).numpy()
            event_numbers = torch.cat(event_numbers).numpy()
    
            combined_arr = np.column_stack((y_scores_dynamic, y_scores_static, y_true, event_numbers, file_numbers))
            np.savez(OUTDIR + 'bkgd_inference_' + str(i) + '.npz', inference_info = combined_arr)
            print(f"Saved inference results for bunch {i}")
    
            #clean stuff up.
            del y_true, y_scores_static, y_scores_dynamic, file_numbers, event_numbers, loader
            file_list.clear()
            reform_list.clear()
    
            #step
            i += 1
    
    
        #this block steps through and loads for all other states
        else:  
            file_list.append(torch.load(background_paths[i], weights_only = False))
            i += 1
    
    
    #now the cleanup, for non interval of 100 finish.
    reform_list = [item for sublist in file_list for item in sublist]
    loader = DataLoader(reform_list, batch_size = 500, drop_last = False, shuffle=True, num_workers=1)
    #run inference, these get reset every time we hit this step.
    y_scores_dynamic = []
    y_scores_static  = []
    y_true           = []
    event_numbers    = []
    file_numbers     = []
    
    print(f'beginning inference on bunch {i}')
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            out_dynamic = model_preloaded_dynamic(data)
            out_static  = model_preloaded_static(data)
            
            probs_dynamic = torch.softmax(out_dynamic, dim=1)[:, 1]
            probs_static  = torch.softmax(out_static,  dim=1)[:, 1]
            
            y_scores_dynamic.append(probs_dynamic.cpu())
            y_scores_static.append(probs_static.cpu())
            y_true.append(data.y.cpu())
            event_numbers.append(data.event_number.cpu())
            file_numbers.append(data.file_number.cpu())
            
            #ensure cleanup
            del data, out_dynamic, out_static, probs_dynamic, probs_static
    
    #save off our stuff. 
    y_true = torch.cat(y_true).numpy()
    y_scores_static = torch.cat(y_scores_static).numpy()
    y_scores_dynamic = torch.cat(y_scores_dynamic).numpy()
    file_numbers = torch.cat(file_numbers).numpy()
    event_numbers = torch.cat(event_numbers).numpy()
    
    combined_arr = np.column_stack((y_scores_dynamic, y_scores_static, y_true, event_numbers, file_numbers))
    np.savez(OUTDIR + 'bkgd_inference_' + str(i) + '.npz', inference_info = combined_arr)
    print(f"Saved inference results for bunch {i}")
    
    #clean stuff up.
    del y_true, y_scores_static, y_scores_dynamic, file_numbers, event_numbers, loader
    file_list.clear()
    reform_list.clear()

    print("completed inference! All files now saved off")

main()











