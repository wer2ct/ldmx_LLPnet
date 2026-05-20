#!/bin/bash
# ============================================================================
# train_combined_bdt.slurm
# ============================================================================
# SLURM batch script to train a combined ECal+HCal BDT on UVA's Rivanna HPC.
#
# Submit with:
#     sbatch train_combined_bdt.slurm
#
# Monitor with:
#     squeue -u $USER
#     sacct -j <JOBID>
# ============================================================================

# ── SLURM Resource Directives ──────────────────────────────────────────────

# Request 1 task (single-node, single-process job; XGBoost is single-threaded
# as configured in the training script with nthread=1)
#SBATCH --ntasks=1

# Wall-clock time limit. ROOT I/O + per-event feature computation can be
# slow in Python; 48 hours provides ample time for large datasets.
#SBATCH --time=48:00:00

# Memory allocation in MB. We load many ROOT files into memory for feature
# extraction; 16 GB provides headroom for large datasets.
#SBATCH --mem=16000

# Run on the standard (free-tier) partition available to all Rivanna users.
#SBATCH --partition=standard

# Fairshare allocation account for the LDMX group at UVA.
#SBATCH -A ldmxuva

# Redirect stdout and stderr to files in your scratch space.
# IMPORTANT: Update the path prefix to your own scratch directory.
#SBATCH --output=/scratch/%u/combined_bdt_training.out
#SBATCH --error=/scratch/%u/combined_bdt_training.err

# Optional: give the job a descriptive name (visible in squeue output).
#SBATCH --job-name=combined_bdt

# ── Environment Setup ──────────────────────────────────────────────────────

# Required packages (must be available in your default Python environment):
#   - uproot, awkward   (ROOT file I/O)
#   - xgboost           (BDT training)
#   - numpy, matplotlib (numerics + plotting)
#   - onnxmltools       (optional, for ONNX conversion)

# Print environment info for debugging/reproducibility in the log file.
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Start time   : $(date)"
echo "Python       : $(which python3)"
echo "Python ver   : $(python3 --version)"
echo "Working dir  : $(pwd)"
echo "============================================"

# ── Data Paths ─────────────────────────────────────────────────────────────


# Background ROOT files directory (enriched nuclear, same files as GNN pipeline)
BKG_DIR="/standard/ldmxuva/gnn_files/note_stuff/training_root/enriched"

# Signal ROOT files directory
SIG_DIR="/standard/ldmxuva/gnn_files/note_stuff/training_root/signal"

# Output directory on scratch (fast parallel filesystem, appropriate for
# job I/O on Rivanna). Results should be copied to /home or /standard
# after the job completes, as scratch is periodically purged.
OUT_DIR="/scratch/$USER/combined_bdt/"

# Path to the training script.
SCRIPT_DIR="/sfs/gpfs/tardis/home/yun5pc/ldmx/eat-vis/scripts"

# ── Pre-Flight Checks ─────────────────────────────────────────────────────

# Create the output directory if it doesn't already exist.
mkdir -p "$OUT_DIR"

# Verify the input directories exist and contain ROOT files.
echo ""
echo "Background dir : $BKG_DIR"
echo "  ROOT files   : $(ls "$BKG_DIR"/*.root 2>/dev/null | wc -l)"
echo "Signal dir     : $SIG_DIR"
echo "  ROOT files   : $(ls "$SIG_DIR"/*.root 2>/dev/null | wc -l)"
echo "Output dir     : $OUT_DIR"
echo ""

# ── Run Training ───────────────────────────────────────────────────────────

# Train the combined BDT. Arguments match the defaults from the existing
# HCal BDT (trainBDT.py) and PreDiscCuts.py thresholds:
#   --roc_file             : radii-of-containment CSV for ECal containment features
#   --ecal_threshold 3160  : ECal energy cut in MeV (same as PreDiscCuts.py)
#   --hcal_threshold 4440  : HCal energy cut in MeV (same as PreDiscCuts.py)
#   --eta 0.023            : XGBoost learning rate
#   --depth 6              : max tree depth
#   --tree_number 1000     : number of boosting rounds
#   --seed 4               : random seed for reproducibility
#   --convert_onnx         : also produce an ONNX model file for inference in ldmx-sw

python3 "$SCRIPT_DIR/trainCombinedBDT.py" \
    --bkg_dir "$BKG_DIR" \
    --sig_dir "$SIG_DIR" \
    --out_dir "$OUT_DIR" \
    --out_name combined_bdt \
    --roc_file /home/yun5pc/ldmx/ldmx-sw/Ecal/data/RoC_v14_8gev.csv \
    --ecal_threshold 3160 \
    --hcal_threshold 4440 \
    --eta 0.023 \
    --depth 6 \
    --tree_number 1000 \
    --seed 4 \
    --convert_onnx

# ── Post-Training ──────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "Job completed at: $(date)"
echo "Output files in : $OUT_DIR"
echo "============================================"

# List what was produced so you can verify in the log file.
ls -lh "$OUT_DIR"/combined_bdt_*/
