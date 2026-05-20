#!/bin/bash
# ============================================================================
# validate_3e13.slurm
# ============================================================================
# Submit 6 parallel jobs to score ~30k background files.
# Each job processes ~5k files. Signal is only processed by job 0.
# After all jobs complete, run merge_3e13.slurm to combine results.
#
# Submit with:  sbatch --array=0-5 validate_3e13.slurm
# Monitor with: squeue -u $USER
# ============================================================================

#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=16000
#SBATCH --partition=standard
#SBATCH -A ldmxuva
#SBATCH --output=/scratch/%u/bdt_3e13_job%a.out
#SBATCH --error=/scratch/%u/bdt_3e13_job%a.err
#SBATCH --job-name=bdt_3e13

# ── Environment ──
module load miniforge

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Array Task   : $SLURM_ARRAY_TASK_ID of 6"
echo "Start time   : $(date)"
echo "============================================"

# ── Paths ──
SCRIPT_DIR="/sfs/gpfs/tardis/home/yun5pc/ldmx/eat-vis/scripts"
MODEL="/scratch/yun5pc/combined_bdt/combined_bdt_0/combined_bdt_0_weights.pkl"
SIG_DIR="/standard/ldmxuva/gnn_files/note_stuff/validation_root/signal"

# Three background directories
BKG_DIR_1="/standard/ldmxuva/EaT_background_8gev/eat_note_run_2026/enriched_nuclear"
BKG_DIR_2="/standard/ldmxuva/EaT_background_8gev/eat_note_run_2026/enriched_nuclear/supplemental"
BKG_DIR_3="/standard/ldmxuva/EaT_background_8gev/eat_note_run_2026/EaT_eot_equivalent_enriched"

# All jobs write to the same output directory
OUT_DIR="/scratch/$USER/bdt_3e13/"
mkdir -p "$OUT_DIR"

# ── Run ──
# --bkg_run_ranges: *:* means no filter for dirs 1 & 2; 51000:66000 filters dir 3
# --n_jobs 6 splits the background files across 6 array tasks
# --job_index uses the SLURM array task ID

python3 "$SCRIPT_DIR/validateBDT.py" \
    --model "$MODEL" \
    --sig_dir "$SIG_DIR" \
    --bkg_dir "$BKG_DIR_1" "$BKG_DIR_2" "$BKG_DIR_3" \
    --bkg_run_ranges "*:*" "*:*" "51000:66000" \
    --out_dir "$OUT_DIR" \
    --roc_file /home/yun5pc/ldmx/ldmx-sw/Ecal/data/RoC_v14_8gev.csv \
    --ecal_threshold 3160 \
    --hcal_threshold 4440 \
    --job_index $SLURM_ARRAY_TASK_ID \
    --n_jobs 6

echo ""
echo "============================================"
echo "Job $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "============================================"
