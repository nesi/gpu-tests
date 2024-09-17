#!/bin/bash -e

#SBATCH --account       nesi99999
#SBATCH --job-name      largetensor
#SBATCH --mem           80G
#SBATCH --time          00:10:00
#SBATCH --gpus-per-node A100:1
#SBATCH --partition     hgx
#SBATCH --output        slog/%j.out

module purge
module load PyTorch/1.12.1-gimkl-2022a-Python-3.10.5-CUDA-11.6.2

python single-gpu.py

# change GPU numbers 1st
#python multiple-gpu.py
