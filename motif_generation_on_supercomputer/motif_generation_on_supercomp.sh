#!/bin/env bash
#SBATCH --job-name="motifs_test_run_ckdtree_10_10"
#SBATCH --time=0-30:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=24
#SBATCH --constraint="type_b"

module purge

module load Python/Anaconda_v11.2020

python3 motif_generation_on_supercomp.py
