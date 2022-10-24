#!/bin/env bash
#SBATCH --job-name="Entropy_Complexity_Length"
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --constraint="type_b"

chmod +x Entropy_Complexity_length.py

module purge

module load Python/Anaconda_v11.2020

python3 Entropy_Complexity_length.py