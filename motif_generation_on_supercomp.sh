#!/bin/env bash

#SBATCH --job-name="motif_gen1"
#SBATCH --cpus-per-task=24
#SBATCH --time=0-2:0
#SBATCH --constraint="type_d"


module purge

module load Python/Anaconda_v11.2020

python3 motif_generation_on_supercomp.py
