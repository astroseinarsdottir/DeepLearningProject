#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J train
#BSUB -n 1
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o %J.out
#BSUB -e %J.err

N_LEVEL=5000
N_STEPS=8e6
SEED=21

module load python3/3.8.0
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

unset PYTHONHOME
unset PYTHONPATH

# export PATH="$HOME/.local/bin:$PATH"

cd ~/DeepLearning/DeepLearningProject

RUN_NAME=$N_LEVEL
RUN_NAME+=_model_deep_value_overfit
RUN_NAME+=_$SEED

echo $RUN_NAME

echo "Running script"
python3 training.py --run_name $RUN_NAME --total_steps $N_STEPS --num_levels $N_LEVEL --distribution_mode hard --num_steps 256 --seed_levels $SEED --num_envs 32
