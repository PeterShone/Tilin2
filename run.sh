#!/bin/bash
#SBATCH --job-name=No
#SBATCH --partition=gpu_24h
#SBATCH --gres=gpu:1
#SBATCH --output=out.log
eval "$(/research/dept8/fyp21/cwf2101/peter/miniconda3/bin/conda shell.bash hook)"
conda activate graph
CUDA_LAUNCH_BLOCKING=1 xvfb-run python3 network_train.py