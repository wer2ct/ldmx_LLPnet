#!/bin/bash
# ============================================================================
# merge_3e13.slurm
# ============================================================================
# Merges the 6 parallel job outputs into final ROC curves and plots.
# Run this AFTER all validate_3e13.slurm array jobs have completed.
#
# Submit with:  sbatch merge_3e13.slurm
# ============================================================================

#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16000
#SBATCH --partition=standard
#SBATCH -A ldmxuva
#SBATCH --output=/scratch/%u/bdt_3e13_merge.out
#SBATCH --error=/scratch/%u/bdt_3e13_merge.err
#SBATCH --job-name=bdt_merge

module load miniforge

SCRIPT_DIR="/sfs/gpfs/tardis/home/yun5pc/ldmx/eat-vis/scripts"
DATA_DIR="/scratch/$USER/bdt_3e13/"

echo "============================================"
echo "Merging BDT validation results"
echo "Start time: $(date)"
echo "============================================"

# Check that all 6 background chunks exist
for i in 0 1 2 3 4 5; do
    if [ ! -f "$DATA_DIR/bkg_scores_job${i}.npz" ]; then
        echo "ERROR: Missing bkg_scores_job${i}.npz — job $i may not have finished."
        exit 1
    fi
done
echo "All 6 background score files found."

python3 "$SCRIPT_DIR/mergeBDTResults.py" \
    --input_dir "$DATA_DIR" \
    --out_dir "$DATA_DIR"

echo ""
echo "============================================"
echo "Merge completed at: $(date)"
echo "============================================"
ls -lh "$DATA_DIR"
