#!/usr/bin/env bash
# Sync latest trial JSONL files from taco and print a one-line summary per trial.
rsync -avzrP 'taco:code/corethink/retrosynformer/results/hypertune/trial_???/*.jsonl' ./results/hypertune/
tail  -q -n1 results/hypertune/trial_???/train_progress.jsonl > train_progress_tail1.jsonl
ls -1 results/hypertune/trial_00?/train_progress.jsonl > train_progress_tail1_paths.txt
python -c 'import pandas as pd; print(pd.read_json(open("train_progress_tail1.jsonl"), lines=True));'
