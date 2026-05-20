"""
large_set_comparison.py

Loads validation data, runs inference with both GNN dynamic and static models,
computes ROC curves, saves scores + ROC arrays to disk, and plots the results.

Outputs:
  - gnn_results.npz  : y_scores_m100_dynamic, y_scores_m100_static,
                       fpr100_d, tpr100_d, fpr100_s, tpr100_s
  - comparison_roc.png / .pdf
"""

import os
import sys
import glob

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

VALIDATION_DIRECTORY = "/standard/ldmxuva/gnn_files/note_stuff/validation_graphs_all"
BIG_BKGD_PATH        = "/standard/ldmxuva/gnn_files/note_stuff/big_enriched_merged_1.pt"
DYNAMIC_MODEL_PATH   = "/standard/ldmxuva/gnn_files/run_output/dynamic_run_extended_best_model.pt"
STATIC_MODEL_PATH    = "/standard/ldmxuva/gnn_files/run_output/static_run_extended_best_model.pt"

OUTPUT_NPZ  = "/standard/ldmxuva/gnn_files/note_stuff/run_output/gnn_results.npz"
OUTPUT_PNG  = "/standard/ldmxuva/gnn_files/note_stuff/run_output/comparison_roc.png"
OUTPUT_PDF  = "/standard/ldmxuva/gnn_files/note_stuff/run_output/comparison_roc.pdf"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Using device: {device} ({num_gpus} GPUs available)")

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------

sys.path.append("modules")
import architectures as arch

# ---------------------------------------------------------------------------
# Load validation files
# ---------------------------------------------------------------------------

validation_paths = glob.glob(os.path.join(VALIDATION_DIRECTORY, "v*"))
print("Validation paths found:", validation_paths)

print("Beginning to load validation files")
val_file_list = []
for i, path in enumerate(validation_paths):
    if i > 3:
        break
    val_file_list.append(torch.load(path, weights_only=False))
    print(f"Loaded: {path}")

# Load big background
val_big_bkgd = torch.load(BIG_BKGD_PATH, weights_only=False)

# ---------------------------------------------------------------------------
# Flatten nested lists and build dataset / loader
# ---------------------------------------------------------------------------

val_big_bkgd = [item for sublist in val_big_bkgd for item in sublist]
print(f"val_big_bkgd type check: {type(val_big_bkgd[0])}")

# m100 signal sample (index 2 = signal_100)
m100_bkgd_roc_sample = val_big_bkgd + val_file_list[2]
m100_roc_sample_loader = DataLoader(
    m100_bkgd_roc_sample,
    batch_size=500,
    drop_last=True,
    shuffle=True,
    num_workers=1,
)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Inference  ← this is the slow cell; suitable for batch submission
# ---------------------------------------------------------------------------

y_scores_m100_dynamic = []
y_scores_m100_static  = []
y_true_m100           = []
event_numbers         = []
file_numbers          = []

with torch.no_grad():
    for data in m100_roc_sample_loader:
        data = data.to(device)

        out_dynamic = model_preloaded_dynamic(data)
        out_static  = model_preloaded_static(data)

        probs_dynamic = torch.softmax(out_dynamic, dim=1)[:, 1]
        probs_static  = torch.softmax(out_static,  dim=1)[:, 1]

        y_scores_m100_dynamic.append(probs_dynamic.cpu())
        y_scores_m100_static.append(probs_static.cpu())
        y_true_m100.append(data.y.cpu())
        event_numbers.append(data.event_number.cpu())
        file_numbers.append(data.file_number.cpu())

print("Inference done!")

y_true_m100           = torch.cat(y_true_m100).numpy()
y_scores_m100_static  = torch.cat(y_scores_m100_static).numpy()
y_scores_m100_dynamic = torch.cat(y_scores_m100_dynamic).numpy()
event_numbers         = torch.cat(event_numbers).numpy()
file_numbers          = torch.cat(file_numbers).numpy()

# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

fpr100_s, tpr100_s, _ = roc_curve(y_true_m100, y_scores_m100_static)
roc_auc_100_s          = auc(fpr100_s, tpr100_s)

fpr100_d, tpr100_d, _ = roc_curve(y_true_m100, y_scores_m100_dynamic)
roc_auc_100_d          = auc(fpr100_d, tpr100_d)

print(f"AUC (static):  {roc_auc_100_s:.7f}")
print(f"AUC (dynamic): {roc_auc_100_d:.7f}")

# ---------------------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------------------

#quite the npz file. 
np.savez(
    OUTPUT_NPZ,
    y_scores_m100_dynamic=y_scores_m100_dynamic,
    y_scores_m100_static=y_scores_m100_static,
    y_true_m100 = y_true_m100,
    event_numbers = event_numbers,
    file_numbers = file_numbers,
    fpr100_d=fpr100_d,
    tpr100_d=tpr100_d,
    fpr100_s=fpr100_s,
    tpr100_s=tpr100_s
)
print(f"Arrays saved to {OUTPUT_NPZ}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.figure()
plt.plot(fpr100_d, tpr100_d, color="green", lw=2,
         label=f"Dynamic\nAUC = {roc_auc_100_d:.7f}")
plt.plot(fpr100_s, tpr100_s, color="red",   lw=2,
         label=f"Static\nAUC = {roc_auc_100_s:.7f}")
plt.xscale("log")
plt.ylim([0.3, 1])
plt.xlabel("False Signal Rate")
plt.ylabel("True Signal Rate")
plt.title("ROC Curve Comparison, Larger Background (Static, Dynamic)")
plt.legend(loc="lower right")
plt.savefig(OUTPUT_PNG)
plt.savefig(OUTPUT_PDF)
print(f"Plots saved to {OUTPUT_PNG} and {OUTPUT_PDF}")
