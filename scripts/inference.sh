#!/usr/bin/env bash
# Sample from GeoCore-9B with text + geospatial conditioning.
set -euo pipefail

CKPT=${CKPT:?"set CKPT to a checkpoint (.pt) or a converted safetensors directory"}
VAE_DIR=${VAE_DIR:?"set VAE_DIR to the Flux2 ae.safetensors"}

python inference.py \
    --ckpt "$CKPT" \
    --vae "$VAE_DIR" \
    --prompt "A satellite view of a highly dense urban city with towering skyscrapers" \
    --lon 126.97 --lat 37.56 --res 0.0 \
    --num-samples 4 \
    --num-steps 50 \
    --cfg-scale 4.0 \
    --out samples \
    "$@"
