#!/bin/bash

#SBATCH --time=3-0
#SBATCH --job-name=ARRahman_dsac_run_with_homography_with
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --output=test.out
#SBATCH --error=test.err


cd ~/git/Homography_based_loss_project 
python dsacstar_training_with_homography.py 7-Scenes fire --iterations 10