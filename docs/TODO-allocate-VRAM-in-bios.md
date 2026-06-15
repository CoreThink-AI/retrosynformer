# TODO: Set up VRAM in BIOS (UEFI) for AMD GPU Framework Desktop server running llama.cpp and pytorch-rocm

## Context

The taco server has an AMD Strix Halo iGPU (Radeon 8060S, gfx1151) with 128 GB of system RAM.
ROCm 7.2.0 is installed but the iGPU only has 512 MB of VRAM allocated in firmware — too small
for PyTorch training. This allocation is set in UEFI/BIOS and must be increased before training works.

## Step 1: Increase iGPU VRAM in UEFI

1. Reboot taco and enter UEFI firmware (usually `Del` or `F2` at POST)
2. Navigate to: **Advanced → AMD CBS → NBIO Common Options → GFX Configuration**
   (exact path varies by motherboard; search for "UMA Frame Buffer Size" or "iGPU Memory")
3. Change **UMA Frame Buffer Size** from `Auto` (512 MB) to `8G` or `16G`
4. Save and exit

Recommended: **8 GB minimum**, 16 GB for comfortable training with larger batch sizes.

> On some AMI BIOS variants the setting is under **Chipset → IGFX** or
> **Advanced → System Agent Configuration → Graphics Configuration → DVMT Pre-Allocated**.

## Step 2: Add hobs to the render group

```bash
sudo usermod -aG render hobs
# Then log out and back in (or: newgrp render)
groups   # should now include "render"
```

## Step 3: Install PyTorch with ROCm support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
# Verify:
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Step 4: Set GFX version override

gfx1151 (RDNA 3.5 / Strix Halo) is not yet in PyTorch's shipped ROCm kernel list.
Use the gfx1100 kernels instead:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Add this to `~/.bashrc` or `~/.profile` on taco to make it permanent:

```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
```

## Step 5: Install project dependencies

```bash
cd ~/code/corethink/retrosynformer
uv sync
```

## Step 6: Run training

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python scripts/train.py -c results/config.yaml -d large -n 100
```

## Verification

```bash
# Check GPU is visible to PyTorch
python3 -c "
import torch
print('ROCm available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"

# Check ROCm sees the GPU
rocm-smi --showmeminfo vram
```

## Hardware summary (as of 2026-06-12)

| Property | Value |
|---|---|
| GPU | AMD Strix Halo (Radeon 8060S) |
| GFX version | gfx1151 |
| ROCm version | 7.2.0 |
| System RAM | 128 GB |
| Current VRAM allocation | 512 MB (too small) |
| Target VRAM allocation | 8–16 GB |
| Python | 3.13.7 |
