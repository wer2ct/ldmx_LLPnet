#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=50000
#SBATCH --output=/scratch/wer2ct/October2025/log/EaT_skimmer_bkgd_%a.out
#SBATCH --partition=standard
#SBATCH -A ldmxuva
 
python3 /sfs/gpfs/tardis/home/wer2ct/ldmx_LLPnet/pre_processing/PreDiscCuts.py /standard/ldmxuva/EaT_background_8gev/eat_note_run_2026/enriched_nuclear/enriched_nuclear_MaxEvents_5000000_skim_True_run_${SLURM_ARRAY_TASK_ID}.root 0 ${SLURM_ARRAY_TASK_ID} 3160 4440 enriched_nuclear_validation /standard/ldmxuva/gnn_files/note_stuff/larger_test_npz/
