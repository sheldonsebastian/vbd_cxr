#!/bin/bash

#SBATCH -o ../slurm_outputs/2_class_test_%j.out
#SBATCH -e ../slurm_outputs/2_class_test_%j.err
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -w node010
#SBATCH --mail-type=all
#SBATCH --mail-user=ssebastian94@gwu.edu

module load anaconda/2020.07
source /modules/apps/anaconda3/etc/profile.d/conda.sh
conda activate vbd_cxr
/home/ssebastian94/.conda/envs/vbd_cxr/bin/python /home/ssebastian94/vbd_cxr/4_testing_data/2_class_classifier/test_binary_predictor.py