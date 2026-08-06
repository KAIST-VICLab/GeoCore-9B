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

<div align="center">

# GeoCore-9B

### Towards Geo-Aware Generative Foundation Models in Earth Observation

[Jeonghyeok Do](https://jeonghyeokdo.github.io/) &nbsp;·&nbsp;
[Munchurl Kim](https://scholar.google.com/citations?user=bGXte_4AAAAJ&hl=en)

Korea Advanced Institute of Science and Technology (KAIST)

[![Project Page](https://img.shields.io/badge/Project%20Page-GeoCore--9B-1a6d5a?style=for-the-badge)](https://kaist-viclab.github.io/GeoCore-9B_site/)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github)](https://github.com/KAIST-VICLab/GeoCore-9B)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

<img src="teaser.jpg" alt="Text-conditioned generation compared with prior methods" width="100%">

</div>

A 9-billion-parameter generative foundation model for Earth Observation, trained from scratch
exclusively on EO data. GeoCore-9B is a Flow Matching Diffusion Transformer that conditions
generation on text **and** continuous geospatial metadata — ground sample distance (GSD), latitude
and longitude.

> [!NOTE]
> **2026-08-04 — corrected weights.** The initial upload accidentally contained an EMA state that
> was never updated during training (pure initialization weights — see
> [GitHub issue #2](https://github.com/KAIST-VICLab/GeoCore-9B/issues/2)). The current files are the
> final 300K-step training weights, verified to load strictly and denoise correctly. If you
> downloaded the weights before this date, please re-download and check them against `SHA256SUMS`.

## Model details

| | |
|---|---|
| Parameters | 9.24 B |
| Weights | final training weights (non-EMA), bfloat16, sharded safetensors |
| Included | frozen Flux.2 VAE in `vae/` (Apache-2.0, Black Forest Labs) |
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
    --vae /path/to/GeoCore-9B/vae \
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

### The frozen VAE is included

`vae/` in this repository holds the frozen Flux.2 autoencoder that encodes and decodes the latents,
so `snapshot_download` gives you everything the model needs:

```bash
python inference.py --ckpt /path/to/GeoCore-9B --vae /path/to/GeoCore-9B/vae ...
```

`models/vae_flux2.py` reads it directly (`load_autoencoder`), and `vae/config.json` is included so
`diffusers >= 0.37` can load it too:

```python
from diffusers import AutoencoderKLFlux2
vae = AutoencoderKLFlux2.from_pretrained("JeonghyeokDo/GeoCore-9B", subfolder="vae")
```

CLIP and T5 text encoders are still downloaded from the Hub at runtime.

> [!IMPORTANT]
> Do **not** substitute `ae.safetensors` from `FLUX.2-dev`. It holds the same autoencoder weights,
> but under the FLUX Non-Commercial License v2.1, whose §4(a)(iii) forbids "surveillance purposes,
> including all research and development related to surveillance" — a clause that Earth-observation
> work should not have to argue about. The copy shipped here is Apache-2.0, and the two were
> verified identical by pairing every tensor on value: 250 of 251 pair one-to-one with a worst
> deviation of 7.802e-03 (bf16 rounding); the odd one out is a BatchNorm step counter. In bf16, the
> precision this model runs in, latents and reconstructions are bit-identical.

## Limitations

* Trained on 256x256 RGB optical imagery; other resolutions and sensor modalities require adaptation.
* Git-10M coverage is uneven across the globe, so generation quality varies by region.
* Geospatial conditioning reflects correlations in the training corpus and is not a substitute for
  real observations of a location.

## License and attribution

GeoCore-9B is released under Apache-2.0, and so is every weight needed to run it.

**`model-*.safetensors` (the 9.24B DiT)** — Copyright 2026 Jeonghyeok Do and Munchurl Kim,
Apache-2.0. Trained from scratch on Git-10M; not derived from any FLUX checkpoint.

**`vae/` (the frozen 84M autoencoder)** — Copyright Black Forest Labs, **Apache-2.0**. This is an
**unmodified, byte-identical redistribution** of
[`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
`vae/diffusion_pytorch_model.safetensors` and its `vae/config.json`, bundled here only so the model
is usable in one download. Verify it against upstream:

```
sha256  ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04  vae/diffusion_pytorch_model.safetensors
sha256  0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948  vae/config.json
```

Those digests are the upstream files' own. A copy of the Apache License 2.0 as distributed with
that model is included as [`LICENSE-FLUX2-VAE.md`](LICENSE-FLUX2-VAE.md). Black Forest Labs neither
endorses nor is affiliated with GeoCore-9B; the attribution above is a statement of origin, not of
sponsorship.

The [Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) training corpus is CC BY-NC-ND 4.0,
which restricts (re)training on that data to non-commercial use.

## Citation

```bibtex
@article{do2026geocore,
  title   = {GeoCore-9B: Towards Geo-Aware Generative Foundation Models in Earth Observation},
  author  = {Do, Jeonghyeok and Kim, Munchurl},
  year    = {2026}
}
```
