#!/usr/bin/env bash

python -m scripts.run_exp train \
    --partition_name='batch' \
    --use_slurm=True \
    --sweep_path=exps/sweep.yaml \
    --sweep_count=50
