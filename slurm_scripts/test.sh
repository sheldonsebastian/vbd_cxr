#!/bin/bash

#SBATCH -o ../slurm_outputs/test_%j.out
#SBATCH -e ../slurm_outputs/test_%j.err
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -w node009
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ssebastian94@gwu.edu

module load anaconda/2020.07
source /modules/apps/anaconda3/etc/profile.d/conda.sh
conda activate vbd_cxr
/home/ssebastian94/.conda/envs/vbd_cxr/bin/python "/home/ssebastian94/vbd_cxr/4_testing_data/test predictor.py"
