# Combined EaT Pre-Discrimination Cuts + Graph Maker Pipeline
# Runs PreDiscCuts filtering and GraphMaker graph creation in a single pass,
# skipping the intermediate .npz file (optionally saved with --save-npz).
#
# Use like:
#   python3 EaTGraphPipeline.py <Input Root File> <Is Signal? (1/0)> <file number>
#                               <ECal Threshold> <HCal Threshold> <type flag>
#                               <outfile location> [--save-npz]
#
# Example:
#   python3 EaTGraphPipeline.py input.root 0 200 3160 4440 enriched_nuclear /home/wer2ct
#   python3 EaTGraphPipeline.py input.root 0 200 3160 4440 enriched_nuclear /home/wer2ct --save-npz

# Imports
import awkward as ak
import numpy as np
import uproot
import sys
import os
import torch
from torch_geometric.data import Data, Dataset


# ── Dataset class (unchanged from GraphMaker.py) ─────────────────────────────

class FileGraphDataset(Dataset):
    def __init__(self, root, file_number=None, signal_status=None,
                 data_list=None, tag=None, transform=None, pre_transform=None):
        self.file_number = file_number
        self.data_list = data_list
        self.signal_status = int(signal_status)
        self.tag = tag
        super().__init__(root, transform, pre_transform)

        if self.data_list is None:
            self.data_list = torch.load(self.processed_paths[0])
        else:
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(self.data_list, self.processed_paths[0])

    @property
    def processed_file_names(self):
        self.converted_status = 'signal' if self.signal_status == 1 else 'background'
        return [f'{self.tag}_{self.converted_status}_{self.file_number}_graphs.pt']

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# ── Pre-discrimination cuts (from PreDiscCuts.py) ────────────────────────────

def run_predisccuts(input_file, is_signal, file_number,
                    ecal_energy_threshold, hcal_energy_threshold):
    """
    Opens a ROOT file, applies ECal/HCal energy cuts, and returns flat numpy
    arrays ready for graph construction — without touching disk.

    Returns
    -------
    hcal_output_array : np.ndarray, columns:
        [file_number, event_id, signal_status, x, y, z, energy, layer, section]
    ecal_output_array : np.ndarray, columns:
        [event_id, x, y, z, energy]
    stats : np.ndarray  [total_events, passing_events]
    """
    if is_signal:
        hcal_rec_pass = 'eat_vis'
        ecal_rec_pass = 'eat_vis'
    else:
        hcal_rec_pass = 'eat'
        ecal_rec_pass = 'eat'
    is_noise_name = 'is_noise_'

    print("Starting Event Processing")
    with uproot.open(input_file) as f:
        big_tree = f["LDMX_Events"]

        branches = {
            "ecal_energy":  f"EcalRecHits_{ecal_rec_pass}.energy_",
            "ecal_noise":   f"EcalRecHits_{ecal_rec_pass}.{is_noise_name}",
            "ecal_x":       f"EcalRecHits_{ecal_rec_pass}.xpos_",
            "ecal_y":       f"EcalRecHits_{ecal_rec_pass}.ypos_",
            "ecal_z":       f"EcalRecHits_{ecal_rec_pass}.zpos_",
            "hcal_energy":  f"HcalRecHits_{hcal_rec_pass}.energy_",
            "hcal_section": f"HcalRecHits_{hcal_rec_pass}.section_",
            "hcal_x":       f"HcalRecHits_{hcal_rec_pass}.xpos_",
            "hcal_y":       f"HcalRecHits_{hcal_rec_pass}.ypos_",
            "hcal_z":       f"HcalRecHits_{hcal_rec_pass}.zpos_",
            "hcal_layer":   f"HcalRecHits_{hcal_rec_pass}.layer_",
        }
        arrays = big_tree.arrays(branches.values(), library="ak")

        ecal_energy  = arrays[branches["ecal_energy"]]
        ecal_noise   = arrays[branches["ecal_noise"]]
        hcal_energy  = arrays[branches["hcal_energy"]]
        hcal_section = arrays[branches["hcal_section"]]

        # ECal cut
        ecal_effective_energy = ak.sum(ecal_energy * (~ecal_noise), axis=1)
        ecal_pass = ecal_effective_energy < ecal_energy_threshold

        # HCal cut (back section only)
        hcal_mask     = hcal_section == 0
        hcal_effective = 12 * ak.sum(hcal_energy * hcal_mask, axis=1)
        hcal_pass     = hcal_effective > hcal_energy_threshold

        event_mask = ecal_pass & hcal_pass
        print(f"Total events:    {len(event_mask)}")
        print(f"Passing events:  {ak.sum(event_mask)}")
        print(f"Efficiency:      {float(ak.sum(event_mask)) / len(event_mask):.4f}")

        # Apply mask and flatten HCal branches
        hcal_x        = arrays[branches["hcal_x"]][event_mask]
        hcal_y        = arrays[branches["hcal_y"]][event_mask]
        hcal_z        = arrays[branches["hcal_z"]][event_mask]
        hcal_layer    = arrays[branches["hcal_layer"]][event_mask]
        hcal_e_pass   = 12 * arrays[branches["hcal_energy"]][event_mask]
        hcal_sec_pass = arrays[branches["hcal_section"]][event_mask]

        # Broadcast event-level metadata down to hit level
        placeholder   = arrays[branches["hcal_x"]]
        event_ids     = ak.broadcast_arrays(ak.local_index(placeholder, axis=0), placeholder)[0]
        passed_ids    = event_ids[event_mask]
        file_numbers  = ak.broadcast_arrays(file_number,    hcal_x)[0]
        sig_status    = ak.broadcast_arrays(int(is_signal), hcal_x)[0]

        hcal_output_array = np.column_stack((
            ak.to_numpy(ak.flatten(file_numbers)),
            ak.to_numpy(ak.flatten(passed_ids)),
            ak.to_numpy(ak.flatten(sig_status)),
            ak.to_numpy(ak.flatten(hcal_x)),
            ak.to_numpy(ak.flatten(hcal_y)),
            ak.to_numpy(ak.flatten(hcal_z)),
            ak.to_numpy(ak.flatten(hcal_e_pass)),
            ak.to_numpy(ak.flatten(hcal_layer)),
            ak.to_numpy(ak.flatten(hcal_sec_pass)),
        ))

        # Apply mask and flatten ECal branches
        ecal_x   = arrays[branches["ecal_x"]][event_mask]
        ecal_y   = arrays[branches["ecal_y"]][event_mask]
        ecal_z   = arrays[branches["ecal_z"]][event_mask]
        ecal_e   = arrays[branches["ecal_energy"]][event_mask]

        ecal_placeholder = arrays[branches["ecal_x"]]
        ecal_event_ids   = ak.broadcast_arrays(ak.local_index(ecal_placeholder, axis=0), ecal_placeholder)[0]
        ecal_passed_ids  = ecal_event_ids[event_mask]

        ecal_output_array = np.column_stack((
            ak.to_numpy(ak.flatten(ecal_passed_ids)),
            ak.to_numpy(ak.flatten(ecal_x)),
            ak.to_numpy(ak.flatten(ecal_y)),
            ak.to_numpy(ak.flatten(ecal_z)),
            ak.to_numpy(ak.flatten(ecal_e)),
        ))

        stats = np.array([len(event_mask), int(ak.sum(event_mask))])

    return hcal_output_array, ecal_output_array, stats


# ── Graph construction (from GraphMaker.py) ──────────────────────────────────

def build_graphs(hcal_hits_array, ecal_hits_array):
    """
    Converts pre-cut hit arrays into a list of torch_geometric Data objects.
    One Data object per passing event.
    """
    signal_status_ = torch.tensor([int(hcal_hits_array[0, 2])], dtype=torch.long)
    file_number_   = torch.tensor(int(hcal_hits_array[0, 0]),   dtype=torch.long)

    print(f"Beginning graph creation for file {file_number_}, "
          f"signal status = {signal_status_[0]}")

    graph_list = []

    for i in np.unique(hcal_hits_array[:, 1]):
        event_hcal_hits = hcal_hits_array[hcal_hits_array[:, 1] == i]
        event_ecal_hits = ecal_hits_array[ecal_hits_array[:, 0] == i]

        if len(event_ecal_hits) == 0:
            print(f"Encountered an event with no ECal hits, event {i} — skipping")
            continue

        hcal_hit_x       = event_hcal_hits[:, 3]
        hcal_hit_y       = event_hcal_hits[:, 4]
        hcal_hit_section = event_hcal_hits[:, -1]
        ecal_hit_x       = event_ecal_hits[:, 1]
        ecal_hit_y       = event_ecal_hits[:, 2]
        ecal_hit_z       = event_ecal_hits[:, 3]
        ecal_hit_e       = event_ecal_hits[:, 4]

        # Remove out-of-volume and side HCal hits
        oov_x      = list(np.where(abs(hcal_hit_x) > 1005)[0])
        oov_y      = list(np.where(abs(hcal_hit_y) > 1005)[0])
        side_hcal  = list(np.where(hcal_hit_section != 0)[0])
        bad_idx    = list(set(oov_x + oov_y + side_hcal))
        cleaned    = np.delete(event_hcal_hits, bad_idx, axis=0)

        hcal_layers   = cleaned[:, 7]
        hcal_z_raw    = cleaned[:, 5]

        # Layer-origin transform (legacy, kept for potential reuse)
        if min(hcal_layers) > 1:
            transformed_z = hcal_z_raw - (min(hcal_z_raw) - 879)
        else:
            transformed_z = hcal_z_raw

        # Z-score normalisation — HCal
        def zscore(arr):
            return (arr - arr.mean()) / (arr.std() + 1e-8)

        final_hcal_x = zscore(cleaned[:, 3])
        final_hcal_y = zscore(cleaned[:, 4])
        final_hcal_z = zscore(transformed_z)
        final_hcal_e = zscore(cleaned[:, 6])

        # Z-score normalisation — ECal
        final_ecal_x = zscore(ecal_hit_x)
        final_ecal_y = zscore(ecal_hit_y)
        final_ecal_z = zscore(ecal_hit_z)
        final_ecal_e = zscore(ecal_hit_e)

        hcal_nodes = np.column_stack((final_hcal_x, final_hcal_y, final_hcal_z, final_hcal_e))
        ecal_nodes = np.column_stack((final_ecal_x, final_ecal_y, final_ecal_z, final_ecal_e))

        # Fully connected edges (no self-loops) for each sub-graph
        def full_edges(n):
            pairs = [(k, j) for k in range(n) for j in range(n) if k != j]
            return torch.tensor(pairs, dtype=torch.long).t().contiguous()

        ecal_edge_index = full_edges(len(ecal_nodes))
        hcal_edge_index = full_edges(len(hcal_nodes))

        # Combine ECal + HCal into one disconnected graph
        N_ECal = len(ecal_nodes)
        N_HCal = len(hcal_nodes)

        x_combined         = torch.cat([
            torch.tensor(ecal_nodes, dtype=torch.float),
            torch.tensor(hcal_nodes, dtype=torch.float),
        ], dim=0)
        edge_index_combined = torch.cat([
            ecal_edge_index,
            hcal_edge_index + N_ECal,   # shift HCal indices
        ], dim=1)
        det_idx = torch.cat([
            torch.zeros(N_ECal, dtype=torch.long),
            torch.ones(N_HCal,  dtype=torch.long),
        ]).unsqueeze(1)

        event_data = Data(
            x=x_combined,
            edge_index=edge_index_combined,
            det_idx=det_idx,
            y=signal_status_,
            file_number=file_number_,
            event_number=torch.tensor(int(i),              dtype=torch.long),
            first_layer=torch.tensor(int(min(hcal_layers)), dtype=torch.long),
            num_nodes_ecal=torch.tensor(N_ECal),
            num_nodes_hcal=torch.tensor(N_HCal),
        )
        graph_list.append(event_data)

    return graph_list, signal_status_, file_number_


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 8:
        print("Usage: python3 EaTGraphPipeline.py <root file> <is_signal 1/0> "
              "<file_number> <ecal_threshold> <hcal_threshold> <type_flag> "
              "<outfile_dir> [--save-npz]")
        sys.exit(1)

    input_file            = sys.argv[1]
    is_signal             = bool(int(sys.argv[2]))
    file_number           = int(sys.argv[3]) 
    ecal_energy_threshold = int(sys.argv[4])
    hcal_energy_threshold = int(sys.argv[5])
    type_flag             = sys.argv[6]
    outfile               = sys.argv[7]
    save_npz              = "--save-npz" in sys.argv

    print(f"Registered is_signal: {is_signal}")

    # ── Step 1: Pre-discrimination cuts ──────────────────────────────────────
    hcal_hits_array, ecal_hits_array, stats = run_predisccuts(
        input_file, is_signal, file_number,
        ecal_energy_threshold, hcal_energy_threshold,
    )

    # Optionally persist the intermediate arrays
    if save_npz:
        signal_string = 'signal' if is_signal else 'background'
        npz_path = os.path.join(
            outfile,
            f'{type_flag}_filtered_{signal_string}_{file_number}.npz'
        )
        np.savez(npz_path,
                 hcal_hits_array=hcal_hits_array,
                 ecal_hits_array=ecal_hits_array,
                 stats_array=stats)
        print(f"Intermediate .npz saved to {npz_path}")

    # ── Step 2: Graph construction ────────────────────────────────────────────
    graph_list, signal_status_, file_number_ = build_graphs(
        hcal_hits_array, ecal_hits_array
    )

    # ── Step 3: Save graphs ───────────────────────────────────────────────────
    FileGraphDataset(
        root=outfile,
        data_list=graph_list,
        file_number=file_number_,
        signal_status=signal_status_[0],
        tag=type_flag,
    )
    print(f"Graphs saved to {outfile}")


main()
