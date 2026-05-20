#This script takes a file that has been converted to .npz (so triggering and cuts applied), and turns it into a point cloud for use in the dynamic GNN or eventually the PointTransformer approach.  
#Use like --> python3 PointCloudMaker.py <Input .npz file> <outpath> <tag>

#Imports
import awkward as ak
import numpy as np
import uproot
import torch
from torch_geometric.data import Data, Dataset
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import glob
from torch_geometric.nn import GCNConv as gcn 
import csv
import sys

#Dataset class def
class FileGraphDataset(Dataset):
    def __init__(self, root, file_number=None, signal_status=None, data_list=None, tag=None, transform=None, pre_transform=None):
        self.file_number = file_number
        self.data_list = data_list
        self.signal_status = int(signal_status)
        self.tag = tag #should be a string with type of background / signal + validation or training.
        super().__init__(root, transform, pre_transform)

        if self.data_list is None: #ie, trying to access 
            self.data_list = torch.load(self.processed_paths[0])
        else: #ie, trying to save
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(self.data_list, self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.signal_status == 0:
            self.converted_status = 'background'
        if self.signal_status == 1:
            self.converted_status = 'signal'
        return( [f'{self.tag}_{self.converted_status}_{self.file_number}_graphs.pt'])

    def len(self):
        return( len(self.data_list) ) #we require datasets to have both a length attribute

    def get(self, idx):
        return(self.data_list[idx]) #as well as a get function to grab by index. 

#Main function
def main():
    
    #Parse command line args
    ecal_and_hcal_file = np.load(sys.argv[1])
    output_file_path = sys.argv[2]
    file_tag = sys.argv[3]

    #Pull the hit arrays and status
    hcal_hits_array = ecal_and_hcal_file["hcal_hits_array"]
    ecal_hits_array = ecal_and_hcal_file["ecal_hits_array"]
    signal_status_ = torch.tensor([int(hcal_hits_array[0,2])], dtype = torch.long)
    file_number_ = torch.tensor(int(hcal_hits_array[0,0]), dtype = torch.long)

    #Now do our graph creation (this needs to be modularized!!)
    print(f"Beginning point cloud creation for file {file_number_}, signal status = {signal_status_[0]}")
    graph_list = []

    for i in np.unique(hcal_hits_array[:,1]):
        #grab our event hits and info
        event_hcal_hits = hcal_hits_array[hcal_hits_array[:,1] == i]
        event_ecal_hits = ecal_hits_array[ecal_hits_array[:,0] == i]
        #this shouldn't happen, but to prevent crashes skip event if no Ecal hits
        if (len(event_ecal_hits) == 0):
            print(f"Encountered an event with no ECal hits, event {i}")
            continue
        
        hcal_hit_x = event_hcal_hits[:,3]
        hcal_hit_y = event_hcal_hits[:,4] 
        ecal_hit_x = event_ecal_hits[:,1]
        ecal_hit_z = event_ecal_hits[:,3]
        ecal_hit_y = event_ecal_hits[:,2]
        ecal_hit_energies = event_ecal_hits[:,4]
        hcal_hit_section = event_hcal_hits[:,-1] 
        
        #Now we want to do some data quality stuff, strike OOV hits and side Hcal hits (can explore later)
        oov_x_idx = list(np.where(abs(hcal_hit_x) > 1005)[0]) #out of volume for 2m long bar
        oov_y_idx = list(np.where(abs(hcal_hit_y) > 1005)[0]) #out of volume for 2m long bar
        hcal_section_idx = list(np.where(hcal_hit_section != 0)[0])
        bad_indices = list(set(oov_x_idx + oov_y_idx + hcal_section_idx)) #make sure no duplicate indices.
        cleaned_hcal_hits = np.delete(event_hcal_hits, bad_indices, axis = 0)

        #transform the z position of our hits in the hcal such that they always start in the first layer
        #this ensures that we can use Ecal information, and we are not biasing on our signal. Replaces the normalization 
        hcal_hit_layers = cleaned_hcal_hits[:,7]
        hcal_hit_z_cleaned = cleaned_hcal_hits[:,5]
        
        if min(hcal_hit_layers) > 1:
            transformed_hcal_zs = hcal_hit_z_cleaned - (min((hcal_hit_z_cleaned) - 879))
        else:
            transformed_hcal_zs = hcal_hit_z_cleaned

        #post cleaning and z transformation
        #the transformation bit is legacy code, leaving it in in case I want to return to it. 
        #we perform a z-score normalization for each feature so if we act with a k-nn during a dynamic version it doesn't break. 
        hcal_x_mean = np.mean(cleaned_hcal_hits[:,3])
        hcal_x_std = np.std(cleaned_hcal_hits[:,3])
        hcal_y_mean = np.mean(cleaned_hcal_hits[:,4])
        hcal_y_std = np.std(cleaned_hcal_hits[:,4])
        hcal_z_mean = np.mean(transformed_hcal_zs)
        hcal_z_std = np.std(transformed_hcal_zs)
        hcal_e_mean = np.mean(cleaned_hcal_hits[:,6])
        hcal_e_std = np.std(cleaned_hcal_hits[:,6])

        #Final hcal values are normalized to z-score
        final_hcal_x = (cleaned_hcal_hits[:,3] - hcal_x_mean) / (hcal_x_std + 1e-8)
        final_hcal_y = (cleaned_hcal_hits[:,4] - hcal_y_mean) / (hcal_y_std + 1e-8)
        final_hcal_z = (transformed_hcal_zs - hcal_z_mean) / (hcal_z_std + 1e-8)
        final_hcal_energies = (cleaned_hcal_hits[:,6] - hcal_e_mean) / (hcal_e_std + 1e-8)

        #Same for ecal
        ecal_x_mean = np.mean(ecal_hit_x)
        ecal_x_std = np.std(ecal_hit_x)
        ecal_y_mean = np.mean(ecal_hit_y)
        ecal_y_std = np.std(ecal_hit_y)
        ecal_z_mean = np.mean(ecal_hit_z)
        ecal_z_std = np.std(ecal_hit_z)
        ecal_e_mean = np.mean(ecal_hit_energies)
        ecal_e_std = np.std(ecal_hit_energies)

        #Final ecal values are normalized to z-score
        final_ecal_x = (ecal_hit_x - ecal_x_mean) / (ecal_x_std + 1e-8)
        final_ecal_y = (ecal_hit_y - ecal_y_mean) / (ecal_y_std + 1e-8)
        final_ecal_z = (ecal_hit_z - ecal_z_mean) / (ecal_z_std + 1e-8)
        final_ecal_energies = (ecal_hit_energies - ecal_e_mean) / (ecal_e_std + 1e-8)

        hcal_node_features = np.column_stack((final_hcal_x, final_hcal_y, final_hcal_z, final_hcal_energies))
        ecal_node_features = np.column_stack((final_ecal_x, final_ecal_y, final_ecal_z, final_ecal_energies))

        #Create tensors
        first_hit_layer = min(hcal_hit_layers) #grab the layer of the earliest hit (pre transforamtion, so not all are = 1)
        hcal_feature_vector = torch.tensor((hcal_node_features), dtype = torch.float) #these are our hcal nodes
        ecal_feature_vector = torch.tensor((ecal_node_features), dtype = torch.float)
        event_number = torch.tensor(int(i), dtype = torch.long)
        event_first_layer = torch.tensor(int(first_hit_layer), dtype = torch.long)

        #Best approach is probably storing in a single graph, but with the ECal and HCal hits disconnected. 
        N_ECal = ecal_feature_vector.size(0) 
        N_HCal = hcal_feature_vector.size(0)
    
        #create a combined feature vector 
        x_combined = torch.cat([ecal_feature_vector, hcal_feature_vector], dim=0)

        det_idx = torch.cat([
            torch.zeros(N_ECal, dtype=torch.long), # Indices 0 to N_ECal - 1 are ECal
            torch.ones(N_HCal, dtype=torch.long)   # Indices N_ECal to N_Total - 1 are HCal
        ]).unsqueeze(1) # Shape: [N_Total, 1]
    
        event_label = signal_status_

        #Now create a combined torch data object. 
        event_data = Data(
            x=x_combined,                     # Node Features [N_Total, F]
            det_idx=det_idx,                  # Detector Identifier [N_Total, 1] (New Attribute)
            y=event_label,                    # Event Label [1]
        
            # Store metadata for tracking/debugging:
            file_number=file_number_,
            event_number=event_number,        # Original event number
            first_layer=event_first_layer,    # Earliest HCal layer (useful auxiliary feature)
            num_nodes_ecal=torch.tensor(N_ECal), # N_ECal (useful for custom operations)
            num_nodes_hcal=torch.tensor(N_HCal)  # N_HCal (useful for custom operations)
        )
        graph_list.append(event_data)

    #So now we have finished our graph creation for the file and appended everything to the running graph list, now we save it!
    FileGraphDataset(root=output_file_path, data_list = graph_list, file_number=file_number_, signal_status=signal_status_[0], tag=file_tag)

    print(f"Point Clouds Saved to {output_file_path}")

main()










    
    


    
