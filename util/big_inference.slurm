#!/bin/bash

#SBATCH --time=18:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --gres=gpu:a40:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=250G   # memory
#SBATCH --output=/standard/ldmxuva/gnn_files/note_stuff/big_inference_notes.out
#SBATCH -J "big_inference"   # job name
#SBATCH --account=ldmxuva   # allocation name


python3 /home/wer2ct/ldmx_LLPnet/eat_equiv_bkgd_inference.py 
