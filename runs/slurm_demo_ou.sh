#!/bin/bash

# SLURM script for OU Supercomputing Center for Education and Research (OSCER)
# Login node: schooner.oscer.ou.edu

#SBATCH --job-name=wlaitest
#SBATCH --output=out/wlaitest.out
#SBATCH --error=out/wlaitest.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80gb
#SBATCH --partition=ai2es
#SBATCH --time=00:30:00
#SBATCH --exclusive

# Path to water level data
DATADIR="/ourdisk/hpc/ai2es/datasets/wlai/"

module purge

# Setup Anaconda environment
. /home/fagg/tf_setup.sh
conda activate wlai

python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
python qc_model_nn.py -d $DATADIR
