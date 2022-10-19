#!/bin/bash
#SBATCH -p stampede
#SBATCH -N 1
#SBATCH -t 72:00:00
 

#SBATCH -J train
#SBATCH -o /mnt/data/home/manuel/aic/logs/train/outputs_slurm/2OIQ8UGSNG.%N.%j.out
#SBATCH -e /mnt/data/home/manuel/aic/logs/train/errors_slurm/2OIQ8UGSNG.%N.%j.err

cd /mnt/data/home/manuel/aic
python scripts/main_training.py --sweep_path=exps/sweep.yaml

