#!/bin/bash

#SBATCH -o ../slurm_outputs/detectron2_retinanet_train_%A.%a.out
#SBATCH -e ../slurm_outputs/detectron2_retinanet_train_%A.%a.err
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -w node010
#SBATCH --mail-type=all
#SBATCH --mail-user=ssebastian94@gwu.edu
#SBATCH --array=0-3

module load anaconda/2020.07
source /modules/apps/anaconda3/etc/profile.d/conda.sh
conda activate vbd_cxr
/home/ssebastian94/.conda/envs/vbd_cxr/bin/python /home/ssebastian94/vbd_cxr/detectron2_codes/retinanet/retinanet_nms.py
