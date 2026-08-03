"""Text- and geo-conditioned sampling from GeoCore-9B.

Example:
    python inference.py \
        --ckpt /path/to/0300000.pt \
        --vae /path/to/ae.safetensors \
        --prompt "A satellite view of a highly dense urban city with towering skyscrapers" \
        --lon 126.97 --lat 37.56 --res 0.0 \
        --num-samples 4 --out samples/

Geospatial metadata is optional: any of --res / --lon / --lat left unset is passed as
-999.0, which routes through the model's learned null embedding for that field.
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file
from torchvision.utils import save_image

from models.flux2 import Flux2, GeoCore4BParams, GeoCore9BParams
from models.vae_flux2 import AutoEncoder, AutoEncoderParams
from models.text_encoder import TextEncoder
from samplers import euler_sampler_flux2

torch.set_float32_matmul_precision('high')

NULL_META = -999.0
PARAMS = {"9b": GeoCore9BParams, "4b": GeoCore4BParams}


def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def load_state_dict(path):
    """Accepts a training .pt (EMA preferred) or a converted safetensors file/directory."""
    if os.path.isdir(path):
        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            state_dict = {}
            for shard in shards:
                state_dict.update(load_file(os.path.join(path, shard)))
            return state_dict
        return load_file(os.path.join(path, "model.safetensors"))

    if path.endswith(".safetensors"):
        return load_file(path)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}


def load_models(args, device, weight_dtype):
    model = Flux2(PARAMS[args.model_size]()).to(device, dtype=weight_dtype)
    model.load_state_dict(load_state_dict(args.ckpt), strict=True)

    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()

    model.eval()

    vae = AutoEncoder(AutoEncoderParams()).to(device, dtype=weight_dtype)
    vae.load_state_dict(load_file(args.vae))
    vae.eval()

    text_encoder = TextEncoder(device=device, dtype=weight_dtype)
    text_encoder.encoder_clip.eval()
    text_encoder.encoder_t5.eval()

    return model, vae, text_encoder


@torch.no_grad()
def generate(model, vae, text_encoder, prompts, res, lon, lat, args, device, weight_dtype, generator):
    n = len(prompts)
    latent_size = args.resolution // 16
    img_ids = _prepare_latent_image_ids(latent_size, latent_size, device, weight_dtype)

    prompt_embeds, pooled_embeds, text_ids = text_encoder(prompts)

    def as_batch(value):
        return torch.full((n,), value, device=device, dtype=weight_dtype)

    cond_kwargs = {
        "x_ids": img_ids.unsqueeze(0).repeat(n, 1, 1),
        "ctx": prompt_embeds,
        "ctx_ids": text_ids,
        "y": pooled_embeds,
        "res": as_batch(res),
        "lon": as_batch(lon),
        "lat": as_batch(lat),
    }

    uncond_kwargs = text_encoder.get_null_info_flux2(n)
    uncond_kwargs["ctx_ids"] = text_ids
    uncond_kwargs["x_ids"] = cond_kwargs["x_ids"]

    latent_channels = AutoEncoderParams.z_channels * 4  # 2x2 patchify inside the VAE
    xT = torch.randn(
        (n, latent_channels, latent_size, latent_size),
        device=device, dtype=weight_dtype, generator=generator,
    )

    samples = euler_sampler_flux2(
        model, xT, cond_kwargs=cond_kwargs, uncond_kwargs=uncond_kwargs,
        num_steps=args.num_steps, cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low, guidance_high=args.guidance_high,
        path_type="linear", heun=False,
    ).to(weight_dtype)

    return ((vae.decode(samples) + 1) / 2.0).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="GeoCore-9B text-to-image sampling")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Training .pt checkpoint, converted .safetensors file, or a sharded directory")
    parser.add_argument("--vae", type=str, required=True, help="Path to the Flux2 VAE ae.safetensors")
    parser.add_argument("--lora", type=str, default=None, help="Optional LoRA adapter directory to merge")
    parser.add_argument("--model-size", type=str, default="9b", choices=list(PARAMS))

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--res", type=float, default=NULL_META,
                        help="Resolution conditioning: 17 minus the Google tile zoom level, so 0 is roughly "
                             "1.2 m/px at the equator and larger values are coarser. Unset means null.")
    parser.add_argument("--lon", type=float, default=NULL_META, help="Longitude in degrees; unset means null")
    parser.add_argument("--lat", type=float, default=NULL_META, help="Latitude in degrees; unset means null")

    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--out", type=str, default="samples")
    args = parser.parse_args()

    assert args.resolution % 16 == 0, "Resolution must be divisible by 16 (for the VAE encoder)."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    os.makedirs(args.out, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    model, vae, text_encoder = load_models(args, device, weight_dtype)

    # lon and lat share one null mask in the model, so they have to drop together
    lon, lat = args.lon, args.lat
    if lon == NULL_META or lat == NULL_META:
        lon = lat = NULL_META

    written = 0
    while written < args.num_samples:
        n = min(args.batch_size, args.num_samples - written)
        images = generate(model, vae, text_encoder, [args.prompt] * n,
                          args.res, lon, lat, args, device, weight_dtype, generator)
        for image in images:
            save_image(image.float(), os.path.join(args.out, f"sample_{written:04d}.png"))
            written += 1

    print(f"Saved {written} sample(s) to {args.out}")


if __name__ == "__main__":
    main()
