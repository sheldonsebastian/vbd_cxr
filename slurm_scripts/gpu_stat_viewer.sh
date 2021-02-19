#!/bin/bash

#SBATCH -o ../slurm_outputs/test_%j.out
#SBATCH -e ../slurm_outputs/test_%j.err
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ssebastian94@gwu.edu
#SBATCH -p gpu

module load anaconda/2020.07
source /modules/apps/anaconda3/etc/profile.d/conda.sh
conda activate vbd_cxr
gpustat
