#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=50000
#SBATCH --output=/scratch/wer2ct/April2026/npz_log/PreDiscCuts_%a.out
#SBATCH --partition=standard
#SBATCH -A ldmxuva
#SBATCH --job-name=eat_equiv_eot_npz
#SBATCH --array=0-29771%1000


FILES=(/standard/ldmxuva/EaT_background_8gev/eat_note_run_2026/EaT_eot_equivalent_enriched/*)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$FILE")

FILE_ID=${BASENAME##*_run_}
FILE_ID=${FILE_ID%.root}

echo "Processing file: $FILE with run ID: $FILE_ID"
 
python3 /sfs/gpfs/tardis/home/wer2ct/ldmx_LLPnet/pre_processing/PreDiscCuts.py $FILE 0 $FILE_ID 3160 4440 eat_equiv_enriched /standard/ldmxuva/gnn_files/note_stuff/eat_equiv_eot_npz/
