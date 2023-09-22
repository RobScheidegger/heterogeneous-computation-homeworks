#!/bin/bash

#SBATCH -n 32
#SBATCH --mem=16G
#SBATCH -t 1:00:00

export OMP_NUM_THREADS=32
export KMP_AFFINITY=compact

make
./benchmark | tee stats/$(date +%y-%m-%d-%s).csv
