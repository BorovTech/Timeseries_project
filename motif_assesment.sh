#!/bin/env bash
#SBATCH --job-name="motif_assesment_init_0_2"
#SBATCH --time=0-05:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --constraint="type_d"

chmod +x motif_assesment.py

module purge

module load Python/Anaconda_v11.2020

python3 motif_assesment.py