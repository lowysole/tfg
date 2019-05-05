#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --mem=2G      # Max CPU Memory
#SBATCH --error=logs/preprocess/error_20_10_180_f_norm.log
#SBATCH --output=logs/preprocess/20_10_180_f_norm.log

input_path="/home/usuaris/veu/llorenc.sole/data/ff1010bird_wav/"
output_path="/home/usuaris/veu/llorenc.sole/bird_detect/workingfiles/features_high_temporal/20_10_180_norm/ff1010bird_wav/"
type_spectro="mel"
process="frequential"
norm="individual"

source env.env
python ./preprocess_signal.py $input_path $output_path --type $type_spectro --process $process --norm $norm
