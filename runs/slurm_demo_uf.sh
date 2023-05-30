#!/bin/bash

# SLURM script for UF NVIDIA Hackathon 2023 

#SBATCH --job-name=wlaitest
#SBATCH --output=out/wlaitest.out
#SBATCH --error=out/wlaitest.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --reservation=hackathon

module purge   

module load conda
conda activate tf

python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
python qc_model_nn.py
