---
license: apache-2.0
pipeline_tag: text-to-image
library_name: safetensors
tags:
  - text-to-image
  - remote-sensing
  - earth-observation
  - satellite-imagery
  - diffusion
  - flow-matching
  - diffusion-transformer
---

# GeoCore-9B

A 9-billion-parameter generative foundation model for Earth Observation, trained from scratch
exclusively on EO data. GeoCore-9B is a Flow Matching Diffusion Transformer that conditions
generation on text **and** continuous geospatial metadata — ground sample distance (GSD), latitude
and longitude.

Paper: *GeoCore-9B: Towards Geo-Aware Generative Foundation Models in Earth Observation* —
[Jeonghyeok Do](https://jeonghyeokdo.github.io/),
[Munchurl Kim](https://scholar.google.com/citations?user=bGXte_4AAAAJ&hl=en)

Code: [KAIST-VICLab/GeoCore-9B](https://github.com/KAIST-VICLab/GeoCore-9B) ·
Project page: [kaist-viclab.github.io/GeoCore-9B_site](https://kaist-viclab.github.io/GeoCore-9B_site/)

## Model details

| | |
|---|---|
| Parameters | 9.24 B |
| Weights | EMA, bfloat16, sharded safetensors |
| Architecture | Flow Matching DiT — 8 double-stream + 24 single-stream blocks, hidden 4096, 32 heads |
| Conditioning | CLIP + T5 text embeddings, GSD, latitude, longitude |
| Training data | [Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) |
| Training | 300K steps, global batch 1024, AdamW lr 1e-4, bf16, DeepSpeed ZeRO-2, 8x B200 |
| Resolution | 256x256 |

Training used a **Geospatial Semantic Alignment (GSA)** loss that aligns intermediate DiT features
(block 8) with a frozen DINOv3-Sat teacher, weighted by `mu = 0.5`. GSA is training-only — the
teacher and projection head are not needed for inference, and the exported weights add no overhead.

## Usage

These weights use a custom `Flux2` architecture, so load them with the model definition from the
GitHub repository rather than a stock `diffusers` pipeline.

```python
import torch
from huggingface_hub import snapshot_download

from models.flux2 import Flux2, GeoCore9BParams
from inference import load_state_dict

path = snapshot_download("JeonghyeokDo/GeoCore-9B")

model = Flux2(GeoCore9BParams()).to("cuda", torch.bfloat16)
model.load_state_dict(load_state_dict(path), strict=True)
model.eval()
```

Sampling, including the Euler flow-matching sampler and classifier-free guidance over both text and
metadata, is provided by `inference.py`:

```bash
python inference.py \
    --ckpt /path/to/GeoCore-9B \
    --vae /path/to/ae.safetensors \
    --prompt "A satellite view of a highly dense urban city with towering skyscrapers" \
    --lon 126.97 --lat 37.56 --res 0.0 \
    --num-samples 4 --out samples/
```

### Conditioning inputs

* `res` — resolution index, defined as `17 - z` for Google XYZ tile zoom `z`. `res = 0` is roughly
  1.2 m/px at the equator; each `+1` doubles the GSD.
* `lon`, `lat` — degrees.
* Any field set to `-999.0` falls back to the model's learned null embedding for that field, so
  metadata is fully optional.

### Additional requirements

The Flux.2 VAE (`ae.safetensors`, Black Forest Labs) is required to decode latents and is not
included here. CLIP and T5 text encoders are downloaded from the Hub at runtime.

## Limitations

* Trained on 256x256 RGB optical imagery; other resolutions and sensor modalities require adaptation.
* Git-10M coverage is uneven across the globe, so generation quality varies by region.
* Geospatial conditioning reflects correlations in the training corpus and is not a substitute for
  real observations of a location.

## Citation

```bibtex
@article{do2026geocore,
  title   = {GeoCore-9B: Towards Geo-Aware Generative Foundation Models in Earth Observation},
  author  = {Do, Jeonghyeok and Kim, Munchurl},
  year    = {2026}
}
```
