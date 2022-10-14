#!/usr/bin/env bash

python -m scripts.run_exp train \
    --partition_name='batch' \
    -yaml_sweep_file=exps/sweep.yaml \
    --max_runs=50 \
    --use_slurm=True
