#!/usr/bin/env bash
# GeoCore-9B pre-training with GSA on 8 GPUs (ZeRO-2).
set -euo pipefail

DATA_DIR=${DATA_DIR:?"set DATA_DIR to the Git-10M root"}
VAE_DIR=${VAE_DIR:?"set VAE_DIR to the Flux.2 VAE (Apache-2.0), e.g. the vae/ folder shipped with the GeoCore-9B weights"}
DINOV3_REPO=${DINOV3_REPO:?"set DINOV3_REPO to a local dinov3 clone with the sat493m weights"}

accelerate launch --config_file configs/zero2_8gpu.yaml train.py \
    --exp-name GeoCore-9B \
    --data-dir "$DATA_DIR" \
    --vae-dir "$VAE_DIR" \
    --dinov3-repo "$DINOV3_REPO" \
    --resolution 256 \
    --batch-size 1024 \
    --max-train-steps 300000 \
    --checkpointing-steps 15000 \
    --learning-rate 1e-4 \
    --adam-weight-decay 0.001 \
    --adam-epsilon 1e-15 \
    --cfg-prob 0.1 \
    --enc-type dinov3-vit-7b \
    --proj-coeff 0.5 \
    --mixed-precision bf16 \
    "$@"
