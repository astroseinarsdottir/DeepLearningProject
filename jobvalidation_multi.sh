#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J valmul
#BSUB -n 1
#BSUB -W 00:30
#BSUB -R "rusage[mem=32GB]"
#BSUB -o %J.out
#BSUB -e %J.err

module load python3/3.8.0
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

unset PYTHONHOME
unset PYTHONPATH

# export PATH="$HOME/.local/bin:$PATH"

cd ~/DeepLearning/DeepLearningProject

echo "Running script"
python3 validation_multi.py --run_name 500_levels_hard_dvRELU
