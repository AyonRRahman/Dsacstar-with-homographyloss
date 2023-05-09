#!/bin/bash

#SBATCH --time=7-0
#SBATCH --job-name=ARRahman_dsac_run_with_homography_without
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=final_without.out
#SBATCH --error=final_without.err


# cd ~/git/Homography_based_loss_project

source /mundus/mrahman527/.bashrc
which python

source activate dsacstar


env>env_slurm.out
which python
pwd
python dsacstar_training_with_homography.py 7-Scenes fire --iterations 10000