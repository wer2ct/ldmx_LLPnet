#Module to store our myriad of different model architectures. 

#Imports
import os
import sys
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

#Utility Functions

#Standard, 4 x StaticEdgeConv Block (works off a fixed graph)
#Called in: GNN_v3_static
class StaticEdgeConvBlock(nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, hc4):
        super(StaticEdgeConvBlock, self).__init__()
        
        #We need 4 MLPs for our edge convolutions to work properly
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels * 2, hc1), nn.ReLU(), nn.Linear(hc1, hc1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hc1 * 2, hc2), nn.ReLU(), nn.Linear(hc2, hc2)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(hc2 * 2, hc3), nn.ReLU(), nn.Linear(hc3, hc3)
        )

        self.mlp4 = nn.Sequential(
            nn.Linear(hc3 * 2, hc4), nn.ReLU(), nn.Linear(hc4, hc4)
        )

        #Now we can define out edge convolutions
        self.edgeconv1 = EdgeConv(self.mlp1, aggr='max') 
        self.edgeconv2 = EdgeConv(self.mlp2, aggr='max')
        self.edgeconv3 = EdgeConv(self.mlp3, aggr='max')
        self.edgeconv4 = EdgeConv(self.mlp4, aggr='max')
        self.batchnorm = nn.BatchNorm1d(hc4)

    #This handles the forward pass of all our edgeconv layers
    def forward(self, x, edge_index):
        #EdgeConv1
        x = F.relu(self.edgeconv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)

        #EdgeConv2
        x = F.relu(self.edgeconv2(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)

        # EdgeConv3
        x = F.relu(self.edgeconv3(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        
        # EdgeConv4
        x = self.batchnorm(self.edgeconv4(x, edge_index))
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        return(x)

#Standard, 4 x DynamicEdgeConv Block (dynamically updates graphs from input point cloud)
#Called in: GNN_v3_dynamic
class DynamicEdgeConvBlock(nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, hc4, k1, k2, k3, k4):
        super(DynamicEdgeConvBlock, self).__init__()
        
        #We need 4 MLPs for our edge convolutions to work properly
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels * 2, hc1), nn.ReLU(), nn.Linear(hc1, hc1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hc1 * 2, hc2), nn.ReLU(), nn.Linear(hc2, hc2)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(hc2 * 2, hc3), nn.ReLU(), nn.Linear(hc3, hc3)
        )

        self.mlp4 = nn.Sequential(
            nn.Linear(hc3 * 2, hc4), nn.ReLU(), nn.Linear(hc4, hc4)
        )

        #Now we can define out edge convolutions
        self.edgeconv1 = DynamicEdgeConv(nn = self.mlp1, k = k1, aggr='max') 
        self.edgeconv2 = DynamicEdgeConv(nn = self.mlp2, k = k2, aggr='max')
        self.edgeconv3 = DynamicEdgeConv(nn = self.mlp3, k = k3, aggr='max')
        self.edgeconv4 = DynamicEdgeConv(nn = self.mlp4, k = k4, aggr='max')
        self.batchnorm1 = nn.BatchNorm1d(hc1)
        self.batchnorm2 = nn.BatchNorm1d(hc2)
        self.batchnorm3 = nn.BatchNorm1d(hc3)
        self.batchnorm4 = nn.BatchNorm1d(hc4)

    #This handles the forward pass of all our dynamic edgeconv layers. We pass the vector and batch (no edge index here!)
    def forward(self, x, batch):
        #EdgeConv1
        x = self.batchnorm1(self.edgeconv1(x, batch))
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        #EdgeConv2
        x = self.batchnorm2(self.edgeconv2(x, batch))
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # EdgeConv3
        x = self.batchnorm3(self.edgeconv3(x, batch))
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # EdgeConv4
        x = self.batchnorm4(self.edgeconv4(x, batch))
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        return(x)

#Model Architectures

#GNN_v3_static architecture
class GNN_v3_static(nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, hc4, fc1, fc2, fc3, out_channels):
        super(GNN_v3_static, self).__init__()
        
        #Initialize our two different pathways (for ECal and HCal)
        self.ECal_branch = StaticEdgeConvBlock(in_channels, hc1, hc2, hc3, hc4) #will receive input masked to only include ECal hits
        self.HCal_branch = StaticEdgeConvBlock(in_channels, hc1, hc2, hc3, hc4) #will receive input masked to only include HCal hits

        #This defines the discriminator MLP head
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(hc4 * 2, fc1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(fc1, fc2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(fc2, fc3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(fc3, out_channels) # Final prediction
        )

    #Parameter initialization
    def parameter_init(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu') #set weights to random from uniform weighted on Relu
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias) #set biases to zero (if there are any)

    def forward(self, data : Data):
        #Grab everything we need out of our data loader for the forward pass
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        det_idx = data.det_idx.squeeze()

        #Create the masks based on detectors
        mask_ecal = (det_idx == 0)
        mask_hcal = (det_idx == 1)

        #Run the edgeconvs 
        h_ecal_all = self.ECal_branch(x, edge_index)
        h_hcal_all = self.HCal_branch(x, edge_index)

        #Global mean pools, with detector masks applied:

        #ECal
        h_ecal_nodes = h_ecal_all[mask_ecal]
        batch_ecal = batch[mask_ecal]
        h_ecal = global_mean_pool(h_ecal_nodes, batch_ecal)

        #HCal
        h_hcal_nodes = h_hcal_all[mask_hcal]
        batch_hcal = batch[mask_hcal]
        h_hcal = global_mean_pool(h_hcal_nodes, batch_hcal)

        #Now we concatenate our ECal and HCal vectors for our linear discriminator 
        h_combined = torch.cat([h_ecal, h_hcal], dim=-1)
        output = self.discriminator(h_combined)

        #Return the output
        return(output)

#GNN_v3_dynamic architecture
class GNN_v3_dynamic(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hc1, hc2, hc3, hc4, 
                 fc1, fc2, fc3, 
                 k1, k2, k3, k4, 
                 out_channels):
        super(GNN_v3_dynamic, self).__init__()
        
        #Initialize our two different pathways (for ECal and HCal)
        self.ECal_branch = DynamicEdgeConvBlock(in_channels, 
                                         hc1, hc2, hc3, hc4, 
                                         k1, k2, k3, k4) 
        self.HCal_branch = DynamicEdgeConvBlock(in_channels, 
                                         hc1, hc2, hc3, hc4, 
                                         k1, k2, k3, k4) #will receive input masked to only include HCal hits

        #This defines the discriminator MLP head
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(hc4 * 2, fc1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(fc1, fc2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(fc2, fc3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(fc3, out_channels) # Final prediction
        )

    #Parameter initialization
    def parameter_init(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu') #set weights to random from uniform weighted on Relu
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias) #set biases to zero (if there are any)

    def forward(self, data : Data):
        #Grab everything we need out of our data loader for the forward pass
        x = data.x
        batch = data.batch
        det_idx = data.det_idx.squeeze()

        #Create the masks based on detectors
        mask_ecal = (det_idx == 0)
        mask_hcal = (det_idx == 1)

        #Run the edgeconvs 
        h_ecal_nodes = self.ECal_branch(x[mask_ecal], batch[mask_ecal])
        h_hcal_nodes = self.HCal_branch(x[mask_hcal], batch[mask_hcal])

        #Global mean pools, with detector masks applied:
        h_ecal = global_mean_pool(h_ecal_nodes, batch[mask_ecal])
        h_hcal = global_mean_pool(h_hcal_nodes, batch[mask_hcal])

        #Now we concatenate our ECal and HCal vectors for our linear discriminator 
        h_combined = torch.cat([h_ecal, h_hcal], dim=-1)
        output = self.discriminator(h_combined)

        #Return the output
        return(output)















