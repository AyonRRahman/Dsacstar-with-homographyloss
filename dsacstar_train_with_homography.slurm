#!/bin/bash

#SBATCH --time=7-0
#SBATCH --job-name=ARRahman_dsac_run_with_homography_without
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --output=final_without.out
#SBATCH --error=final_without.err


cd ~/git/Homography_based_loss_project 
python dsacstar_training_with_homography.py 7-Scenes fire --iterations 10000