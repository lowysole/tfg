#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --mem=2G      # Max CPU Memory
#SBATCH --error=logs/preprocess/error_norm.log
#SBATCH --output=logs/preprocess/60_20_80_full.log

input_path="/home/usuaris/veu/llorenc.sole/data/BirdVox-DCASE20k_wav/"
output_path="/home/usuaris/veu/llorenc.sole/bird_detect/workingfiles/features_high_temporal/60_20_80_full/BirdVox-DCASE-20k"
type_spectro="mel"
process="temporal"
norm="full"

source env.env
python ./preprocess_signal.py $input_path $output_path --type $type_spectro --process $process --norm $norm
