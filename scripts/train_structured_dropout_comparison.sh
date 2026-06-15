#!/usr/bin/env bash
# Run paired baseline vs structured-dropout hypertune studies on the small dataset.

# 1 — Baseline (standard dropout)
# .venv-rocm/bin/python scripts/hypertune.py \
retrosynformer-hypertune \
  -c results/config/small_standard.yaml \
  --study-name compare2_small_standard_dropout \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small

# 2 — Structured dropout
# .venv-rocm/bin/python scripts/hypertune.py \
retrosynformer-hypertune \
  -c results/config/small_structured.yaml \
  --study-name compare2_small_structured_dropout \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small
