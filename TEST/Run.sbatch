#!/bin/bash
#SBATCH --job-name=HF_MoSe2
#SBATCH --output=HF_MoSe2
#SBATCH --time=24:00:00
#SBATCH -n 1 --cpus-per-task 8 --mem=48000
module load intel

hostname
date
time ./MoireBands input_run.inp > out_run.txt
date
