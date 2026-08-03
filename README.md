# GeoCore-9B: Towards Geo-Aware Generative Foundation Models in Earth Observation

Official implementation of **GeoCore-9B**, a 9-billion-parameter generative foundation model for
Earth Observation (EO), trained from scratch exclusively on EO data.

[Project page](https://kaist-viclab.github.io/GeoCore-9B_site/) ·
[Model weights](https://huggingface.co/JeonghyeokDo/GeoCore-9B)

[Jeonghyeok Do](https://jeonghyeokdo.github.io/) ·
[Munchurl Kim](https://scholar.google.com/citations?user=bGXte_4AAAAJ&hl=en)

---

## Overview

GeoCore-9B is a Flow Matching Diffusion Transformer (DiT) that natively conditions generation on
text descriptions **and** continuous geospatial metadata — ground sample distance (GSD), latitude,
and longitude. Unlike prior EO generative models that fine-tune natural-image priors, it is trained
from scratch on the global-scale [Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) corpus,
avoiding the perspective biases that conflict with the orthographic, physically-anchored nature of
satellite imagery.

| | |
|---|---|
| Parameters | 9.24 B |
| Backbone | Flow Matching DiT (8 double-stream + 24 single-stream blocks, hidden 4096, 32 heads) |
| Conditioning | CLIP + T5 text embeddings, GSD, latitude, longitude |
| Pre-training | 300K steps, Git-10M, global batch 1024, AdamW lr 1e-4, bf16, DeepSpeed ZeRO-2 |
| Hardware | 8x NVIDIA B200, ~15 days |

### Geospatial Semantic Alignment (GSA)

Training a 9B DiT from scratch on EO data converges slowly and produces spatially unstable samples.
GSA is a **training-only** feature alignment objective: intermediate DiT tokens at block `k = 8` are
projected into the feature space of a frozen [DINOv3-Sat](https://github.com/facebookresearch/dinov3)
teacher and aligned with its dense structural representations.

```
L_total = L_FlowMatching + mu * L_GSA        (mu = 0.5)
```

Because the teacher and the projection head exist only during training, GSA adds **zero inference
overhead**. In this codebase GSA appears as:

| Component | Location |
|---|---|
| Projection head `W_proj` | `REPAEmbedder` in [`models/flux2.py`](models/flux2.py) |
| Alignment block `k = 8` | `depth_repa` field on the `*Params` dataclasses |
| Alignment loss | `SILoss_Flux2.proj_loss` in [`loss.py`](loss.py) |
| Frozen teacher | `load_encoders` in [`utils.py`](utils.py) (`--enc-type dinov3-vit-7b`) |
| Weight `mu` | `--proj-coeff 0.5` |

`SILoss_Flux2_no_repa` in [`loss.py`](loss.py) is the GSA-free variant used for LoRA and downstream
adaptation.

### Metadata conditioning

GSD `r`, latitude `phi`, and longitude `lambda` are each encoded as 256-dimensional sinusoidal
features and projected by separate MLP embedders. Their sum is added to the timestep embedding and
the pooled text embedding to form the global conditioning vector that modulates the DiT through
AdaLN. See `Flux2.make_vec` in [`models/flux2.py`](models/flux2.py).

Missing metadata is not simply zeroed: the model holds **learnable null embeddings**
(`null_res_emb`, `null_lon_emb`, `null_lat_emb`) that substitute for absent fields. Passing
`-999.0` for a field selects its null embedding, which is also how classifier-free guidance dropout
(`p_cfg = 0.1`) is implemented during training (see `CFGDataset` in
[`data/dataset.py`](data/dataset.py)).

**Resolution convention.** The `res` conditioning value is `17 - z`, where `z` is the Google XYZ
tile zoom level, so `res = 0` corresponds to roughly 1.2 m/px at the equator and each `+1` doubles
the GSD (`res = 5` is roughly 38 m/px). Images upscaled during preprocessing have
`log2(scale_factor)` subtracted. This is computed in `Git10MDataset.__getitem__`.

### Preprocessing and data

Git-10M is consumed in HuggingFace `datasets` format. Preprocessing lives in
[`data/dataset.py`](data/dataset.py):

* `XYZToLonLat` converts a Google `z_x_y` tile id to longitude/latitude, and `17 - z` to the GSD index.
* `process_image` resizes so the short side reaches 256 px (only when needed), random-crops to
  256x256, and normalizes to `[-1, 1]`.
* `CFGDataset` applies per-field conditioning dropout for classifier-free guidance.
* [`data/dataset_finetune.py`](data/dataset_finetune.py) additionally filters on
  `img_quality_score >= 4.8` for the progressive refinement stage.

---

## Installation

```bash
git clone <this-repo> && cd GeoCore-9B
pip install -r requirements.txt
```

### External assets

None of the following are redistributed here; download them from their original sources and respect
their licenses.

| Asset | Needed for | Source |
|---|---|---|
| Git-10M dataset | pre-training, fine-tuning | [`lcybuaa/Git-10M`](https://huggingface.co/datasets/lcybuaa/Git-10M) (CC BY-NC-ND 4.0) |
| Flux.2 VAE `ae.safetensors` | everything | Black Forest Labs |
| DINOv3-Sat ViT-7B/16 (`sat493m`) | GSA pre-training only | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) |

CLIP and T5 text encoders are pulled from the HuggingFace Hub at runtime.

```bash
export DATA_DIR=/path/to/Git-10M/train
export VAE_DIR=/path/to/ae.safetensors
export DINOV3_REPO=/path/to/dinov3      # local clone containing the sat493m checkpoint
```

---

## Usage

### Pre-training (with GSA)

```bash
bash scripts/train_pretrain.sh
```

Reproduces the paper run: 300K steps, global batch 1024, `--proj-coeff 0.5`, ZeRO-2 across 8 GPUs
via [`configs/zero2_8gpu.yaml`](configs/zero2_8gpu.yaml). The exact hyperparameters are recorded in
[`configs/pretrain_geocore9b.json`](configs/pretrain_geocore9b.json).

### Full fine-tuning (progressive refinement)

```bash
BASE_CKPT=/path/to/0300000.pt bash scripts/train_finetune.sh
```

Continues training on the `img_quality_score >= 4.8` subset of Git-10M.

### LoRA fine-tuning

```bash
BASE_CKPT=/path/to/0300000.pt bash scripts/train_lora.sh
```

Freezes the backbone and trains LoRA adapters (`r = 64`, `alpha = 128`) on the `qkv`, `proj`,
`linear1` and `linear2` projections.

For **image-conditioned** downstream tasks such as cloud removal or SAR-to-optical translation, pass
`--cond-image`. This widens `img_in` with zero-initialised channels so a conditioning latent can be
channel-concatenated onto the noised input — the expanded model starts out identical to the
pre-trained one. You supply the paired dataset:

```bash
python finetune_lora.py --exp-name cloud-removal \
    --base-ckpt /path/to/0300000.pt --vae-dir "$VAE_DIR" \
    --data-dir /path/to/pairs \
    --cond-image --dataset-class my_pkg.my_module:MyPairedDataset
```

The class is constructed as `MyPairedDataset(data_dir)` and each item must be
`(cond_img, target_img, caption, meta)`, where `cond_img` and `target_img` are `[3, H, W]` tensors in
`[-1, 1]` and `meta` is a dict with float tensors `res`, `lon`, `lat` (use `-999.0` when unknown).
Checkpoints save the LoRA adapter plus `img_in_weight.pt`, since `img_in` is trained outside LoRA.

### Inference

```bash
python inference.py \
    --ckpt /path/to/0300000.pt --vae "$VAE_DIR" \
    --prompt "A satellite view of a highly dense urban city with towering skyscrapers" \
    --lon 126.97 --lat 37.56 --res 0.0 \
    --num-samples 4 --out samples/
```

Omit any of `--res`, `--lon`, `--lat` to generate without that condition (the learned null embedding
is used instead). Add `--lora /path/to/lora_adapter` to merge an adapter before sampling.
`--ckpt` accepts a training `.pt`, a converted `.safetensors` file, or a sharded directory.

### Frozen probes

Linear probes on frozen DiT features, following the pre-registered protocol described in
[`eval/frozen_probe.py`](eval/frozen_probe.py): `z_tau = (1 - tau) z_0 + tau * eps` with `tau = 0.25`,
one forward pass with empty text and null metadata, features hooked at global blocks
`{4, 8, 16, 24}` (block 8 is the GSA-aligned one).

```bash
python eval/frozen_probe.py --task eurosat \
    --ckpt /path/to/0300000.pt --vae-dir "$VAE_DIR" \
    --data-root /path/to/downstream_datasets --gpu 0
```

Tasks: `loveda` (7-class segmentation, mIoU), `eurosat` (10-class classification, top-1), `bright`
(siamese building-damage change detection, mIoU). `--data-root` must contain `LoveDA/`,
`EuroSAT_RGB/` and `BRIGHT/`.

### Metrics

[`eval/metrics.py`](eval/metrics.py) provides batched PSNR / SSIM (3-D Gaussian) / LPIPS for
validation and downstream evaluation.

---

## Pre-trained weights

Convert the training checkpoint into sharded bf16 safetensors for the Hub. The training `.pt`
bundles the DeepSpeed ZeRO-2 optimizer state (~65 GB); the export keeps only the EMA weights
(~18.5 GB), which are the ones reported in the paper.

```bash
python scripts/convert_checkpoint.py \
    --ckpt exps/GeoCore-9B/checkpoints/0300000.pt \
    --out huggingface/ --weights ema --dtype bf16
```

Loading the exported weights:

```python
import torch
from models.flux2 import Flux2, GeoCore9BParams
from inference import load_state_dict

model = Flux2(GeoCore9BParams()).to("cuda", torch.bfloat16)
model.load_state_dict(load_state_dict("path/to/huggingface"), strict=True)
model.eval()
```

---

## Repository layout

```
train.py               pre-training with GSA
finetune.py            full fine-tuning on the high-quality subset
finetune_lora.py       LoRA adaptation (text-to-image or image-conditioned)
inference.py           text + geospatial conditioned sampling
loss.py                flow matching objective, with and without GSA
samplers.py            Euler / Euler-Maruyama flow matching samplers
utils.py               DINOv3-Sat teacher loading
models/flux2.py        DiT backbone, metadata conditioning, GSA head
models/vae_flux2.py    Flux.2 autoencoder
models/text_encoder.py CLIP + T5 text encoders
data/                  Git-10M dataset, metadata extraction, CFG dropout
eval/                  frozen linear probes, image quality metrics
configs/               ZeRO-2 accelerate config, pre-training hyperparameters
scripts/               launch scripts and checkpoint conversion
```

---

## Citation

```bibtex
@article{do2026geocore,
  title   = {GeoCore-9B: Towards Geo-Aware Generative Foundation Models in Earth Observation},
  author  = {Do, Jeonghyeok and Kim, Munchurl},
  year    = {2026}
}
```

## Acknowledgements

The transformer and autoencoder in [`models/`](models/) are adapted from the
[FLUX reference implementation](https://github.com/black-forest-labs/flux) by Black Forest Labs
(Apache-2.0). GSA builds on representation alignment for diffusion training, and the frozen
teacher is [DINOv3-Sat](https://github.com/facebookresearch/dinov3). Pre-training uses the
[Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) dataset from Text2Earth.

## License

Code is released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution of
adapted third-party code. The Git-10M dataset, the Flux.2 VAE weights and the DINOv3-Sat teacher
weights are not redistributed here and are governed by their own licenses.
