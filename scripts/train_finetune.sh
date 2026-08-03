#!/usr/bin/env bash
# Full fine-tuning of GeoCore-9B on the high-quality Git-10M subset.
set -euo pipefail

DATA_DIR=${DATA_DIR:?"set DATA_DIR to the Git-10M root"}
VAE_DIR=${VAE_DIR:?"set VAE_DIR to the Flux2 ae.safetensors"}
BASE_CKPT=${BASE_CKPT:?"set BASE_CKPT to the pre-trained GeoCore-9B checkpoint"}
DINOV3_REPO=${DINOV3_REPO:?"set DINOV3_REPO to a local dinov3 clone with the sat493m weights"}

accelerate launch --config_file configs/zero2_8gpu.yaml finetune.py \
    --exp-name GeoCore-9B-finetune \
    --base-ckpt "$BASE_CKPT" \
    --data-dir "$DATA_DIR" \
    --vae-dir "$VAE_DIR" \
    --dinov3-repo "$DINOV3_REPO" \
    --score-threshold 4.8 \
    --resolution 256 \
    --max-train-steps 50000 \
    --checkpointing-steps 5000 \
    --mixed-precision bf16 \
    "$@"
