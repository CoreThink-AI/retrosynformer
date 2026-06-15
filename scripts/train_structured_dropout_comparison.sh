# 1 — Baseline (standard dropout)
.venv-rocm/bin/python scripts/hypertune.py \
  -c results/config/small.yaml \
  --study-name small_baseline \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small

# 2 — Structured dropout
.venv-rocm/bin/python scripts/hypertune.py \
  -c results/config/small_sd.yaml \
  --study-name small_sd \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small
