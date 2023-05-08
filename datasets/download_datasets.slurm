#!/bin/bash

#SBATCH --time=3-0
#SBATCH --job-name=ARRahman_data_download



#SBATCH --output=dataset_download.out
#SBATCH --error=dataset_download.err

cd ~/projects/dsacstar/homography_loss_function/datasets 

python datasetup.py Cambridge
python datasetup.py 7-Scenes

echo $HOSTNAME