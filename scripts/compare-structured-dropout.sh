source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
rs-hypertune \
  -c results/config/small_nodropout_baseline.yaml \
  --study-name nodropout-baseline \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small

# Ctrl-b c   (new window)
cd ~/code/corethink/retrosynformer
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
rs-hypertune \
  -c results/config/small_nodropout_sd.yaml \
  --study-name nodropout-sd \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small
