#!/bin/env bash
#SBATCH --job-name="clustering_test_run1"
#SBATCH --time=0-01:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --constraint="type_d"

module purge

module load Python/Anaconda_v11.2020

python3 onefile_clustering.py
