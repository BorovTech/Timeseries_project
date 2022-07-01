#!/bin/env bash
#SBATCH --job-name="motifs_assesment_1"
#SBATCH --time=0-10:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --constraint="type_d"

chmod +x motif_assesment_1.py

module purge

module load Python/Anaconda_v11.2020

python3 motif_assesment_1.py