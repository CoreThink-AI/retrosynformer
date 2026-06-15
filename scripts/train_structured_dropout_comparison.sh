#!/usr/bin/env bash
# Run paired baseline vs structured-dropout hypertune studies on the small dataset.

# 1 — Baseline (standard dropout)
# .venv-rocm/bin/python scripts/hypertune.py \
retrosynformer-hypertune \
  -c results/config/small.yaml \
  --study-name compare_small_standard_dropout_baseline \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small

# 2 — Structured dropout
# .venv-rocm/bin/python scripts/hypertune.py \
retrosynformer-hypertune \
  -c results/config/compare_small_structured_dropout.yaml \
  --study-name small_sd \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small
