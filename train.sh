#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --mem=6G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --error=logs/train/error_TF_WF.log
#SBATCH --output=logs/train/TF_WF.log


source env.env
python ./birddet_baseline.py
