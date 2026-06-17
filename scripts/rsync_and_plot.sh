#!/usr/bin/env bash
export STUDY_NAME=standard-v2-dropout-details
if [[ -n $1 ]] ; then
   STUDY_NAME=$1
fi
rsync -av \
    --include='*/' \
      --include='study.db' \
      --include='*.yaml' \
      --include='pred_routes_train_progress.json' \
      --include='train_progress.jsonl' \
      --include '*config*' \
      --include '*.log' \
    --exclude='*' \
    taco:code/corethink/retrosynformer/results/ \
    results/  \
&& rs-plot-learning-curves \
    --study standard-v2-dropout-details \
    --yscale linear
