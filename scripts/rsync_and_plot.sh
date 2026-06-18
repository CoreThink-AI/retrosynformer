#!/usr/bin/env bash
export STUDY_NAME=standard-v2-dropout-details

if [[ -n $1 ]] ; then
   STUDY_NAME=$1
fi
rsync -av \
  --include='*/' \
    --include='study.db' \
    --include='pred_routes_train_progress.json' \
    --include='train_progress.jsonl' \
    --include '*.log' \
    --include='model.config.yaml' \
    --include='run.jsonl' \
  --exclude='*' \
  taco:code/corethink/retrosynformer/results/ \
  results/

rs-plot-learning-curves \
    --study $STUDY_NAME \
    --yscale linear
