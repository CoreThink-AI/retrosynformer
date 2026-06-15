# 1. Start a named tmux session
tmux new-session -s hypertune

# Inside tmux — run the baseline first
cd ~/code/corethink/retrosynformer
git pull
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
rs-hypertune \
  -c results/config/small_nodropout_baseline.yaml \
  --study-name nodropout-baseline \
  --n-trials 10 \
  --n-epochs 200 \
  --dataset small

When baseline finishes, open a second window for the SD arm:

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

Detach / reattach:
Ctrl-b d                  # detach (leaves it running)
tmux attach -t hypertune  # reattach later

Switch between windows:
Ctrl-b 0   # window 0 (baseline)
Ctrl-b 1   # window 1 (SD)

If you want to run both arms in parallel (only do this if VRAM is large enough for two training jobs), open the second window before the first finishes — the lock files are per-study-name so they won't conflict.

