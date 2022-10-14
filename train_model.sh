#!/usr/bin/env bash

python -m src.scripts.run_exp.py \
    --partition_name='batch' \
    -yaml_sweep_file=exps/sweep.yaml \
    --max_runs=50
    --use_slurm=True
