#!/usr/bin/env bash
# LoRA fine-tuning of GeoCore-9B (text-to-image by default).
# For image-conditioned adaptation add: --cond-image --dataset-class my_pkg.my_module:MyPairedDataset
set -euo pipefail

DATA_DIR=${DATA_DIR:?"set DATA_DIR to the fine-tuning dataset root"}
VAE_DIR=${VAE_DIR:?"set VAE_DIR to the Flux.2 VAE weights (Apache-2.0 copy: FLUX.2-klein-base-4B vae/diffusion_pytorch_model.safetensors)"}
BASE_CKPT=${BASE_CKPT:?"set BASE_CKPT to the pre-trained GeoCore-9B checkpoint"}

accelerate launch --config_file configs/zero2_8gpu.yaml finetune_lora.py \
    --exp-name GeoCore-9B-lora \
    --base-ckpt "$BASE_CKPT" \
    --data-dir "$DATA_DIR" \
    --vae-dir "$VAE_DIR" \
    --lora-rank 64 \
    --lora-alpha 128 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --max-train-steps 50000 \
    --checkpointing-steps 5000 \
    --mixed-precision bf16 \
    "$@"
