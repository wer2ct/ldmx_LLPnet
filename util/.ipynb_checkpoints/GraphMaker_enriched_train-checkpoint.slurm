#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --output=/scratch/wer2ct/October2025/log/EaT_skimmer_bkgd_%a.out
#SBATCH --partition=standard
#SBATCH -A ldmxuva
 
python3 /sfs/gpfs/tardis/home/wer2ct/ldmx_LLPnet/pre_processing/GraphMaker.py /standard/ldmxuva/gnn_files/note_stuff/training_npz/enriched/enriched_nuclear_training_filtered_background_${SLURM_ARRAY_TASK_ID}* /standard/ldmxuva/gnn_files/note_stuff/training_graphs/enriched training_enriched
