#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J 50_levels
#BSUB -n 1
#BSUB -W 6:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o %J.out
#BSUB -e %J.err

N_LEVEL=50
N_STEPS=5e6

module load python3/3.8.0
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

unset PYTHONHOME
unset PYTHONPATH

# export PATH="$HOME/.local/bin:$PATH"

cd ~/DeepLearning/DeepLearningProject

RUN_NAME=$N_LEVEL
RUN_NAME+=_levels_hard

echo $RUN_NAME

echo "Running script"
python3 training.py --run_name $RUN_NAME --total_steps $N_STEPS --num_levels $N_LEVEL --num_envs 32 --value_coef 0.5 --distribution_mode hard
