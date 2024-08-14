#!/bin/bash -e 

#SBATCH --account       nesi99999
#SBATCH --job-name      lattice-julia
#SBATCH --ntasks        6
#SBATCH --mem           6G
#SBATCH --time          00:05:00
#SBATCH --gpus-per-node A100:2
#SBATCH --partition     hgx
#SBATCH --output        slog/%j.out

module purge >/dev/null 2>&1
module load Julia
module load CUDA/11.8.0

julia ./singlegpu-lattice.jl
#julia ./multigpu-lattice.jl
