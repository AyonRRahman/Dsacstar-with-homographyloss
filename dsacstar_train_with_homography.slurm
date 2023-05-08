#!/bin/bash

#SBATCH --time=3-0
#SBATCH --job-name=ARRahman_dsac_run_with_homography_with
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --output=train_with_homography_chess_with.out
#SBATCH --error=train_with_homography_chess_with.err


cd ~/projects/dsacstar/ 
python dsacstar_training_with_homography.py 7-Scenes chess --network_in our_init.net