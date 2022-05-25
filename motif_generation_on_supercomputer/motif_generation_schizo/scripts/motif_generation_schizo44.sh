#!/bin/env bash
#SBATCH --job-name="motifs_ckdtree_schizo_44"
#SBATCH --time=0-04:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --constraint="type_b"

chmod +x motif_generation_schizo44.py

module purge

module load Python/Anaconda_v11.2020

python3 motif_generation_schizo44.py