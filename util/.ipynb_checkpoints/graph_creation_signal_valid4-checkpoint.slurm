#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --output=/scratch/wer2ct/December2025/log/graphs_sgnl_%a.out
#SBATCH --partition=standard
#SBATCH -A ldmxuva
 
python3 /sfs/gpfs/tardis/home/wer2ct/ldmx_LLPnet/pre_processing/GraphMaker.py /standard/ldmxuva/gnn_files/note_stuff/validation_npz/signal/signal_validation_filtered_signal_100.npz /standard/ldmxuva/gnn_files/note_stuff/validation_graphs/signal validation_signal

