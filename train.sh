#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --mem=6G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --error=logs/train/error.log
#SBATCH --output=logs/train/30_10_norm.log


source env.env
python ./birddet_baseline.py
