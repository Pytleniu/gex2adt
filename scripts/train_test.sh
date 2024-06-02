#!/bin/bash

#SBATCH -A plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH -t 100
#SBATCH -c 16
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1

srun python vanilla_training_better.py