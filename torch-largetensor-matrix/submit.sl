#!/bin/bash -e 

#SBATCH --account       nesi99999
#SBATCH --job-name      largetensor
#SBATCH --ntasks        6
#SBATCH --mem           80G
#SBATCH --time          00:15:00
#SBATCH --gpus-per-node A100:2
#SBATCH --partition     hgx
#SBATCH --output        slog/%j.out

module purge
module load PyTorch

python single-gpu.py
#python multiple-gpu.py
