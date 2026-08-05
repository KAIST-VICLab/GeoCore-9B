"""Convert a GeoCore-9B training checkpoint into sharded safetensors for the Hub.

The training checkpoint bundles the DeepSpeed ZeRO-2 optimizer state alongside the
weights (~65 GB). This drops everything except the model tensors and writes bf16
safetensors shards plus the index and config.json needed to reload them (~18 GB).

    python scripts/convert_checkpoint.py \
        --ckpt exps/GeoCore-9B_new_rev/checkpoints/0300000.pt \
        --out huggingface/
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.flux2 import GeoCore4BParams, GeoCore9BParams

PARAMS = {"9b": GeoCore9BParams, "4b": GeoCore4BParams}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Layers the architecture zero-initialises. If they are still all-zero at export
# time, the state dict is untrained (e.g. an EMA that never received updates).
ZERO_INIT_PROBES = (
    "final_layer.linear.weight",
    "time_in.out_layer.weight",
    "double_stream_modulation_img.lin.weight",
)


def strip_wrapper_prefixes(name):
    """Drop leading 'module.' / '_orig_mod.' added by DDP/DeepSpeed/torch.compile."""
    while name.startswith(("module.", "_orig_mod.")):
        name = name.split(".", 1)[1]
    return name


def shard_state_dict(state_dict, max_shard_bytes):
    """Greedily pack tensors into shards of at most max_shard_bytes."""
    shards, current, current_bytes = [], {}, 0
    for key, tensor in state_dict.items():
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[key] = tensor
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def main():
    ap = argparse.ArgumentParser(description="Export GeoCore-9B weights as safetensors")
    ap.add_argument("--ckpt", required=True, help="Training checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--weights", default="model", choices=["ema", "model"],
                    help="Which state dict to export ('model' is the raw trained weights)")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--model-size", default="9b", choices=list(PARAMS))
    ap.add_argument("--max-shard-size-gb", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = DTYPES[args.dtype]

    print(f"Loading {args.ckpt} (mmap, weights only are materialised) ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    if args.weights not in ckpt:
        raise KeyError(f"'{args.weights}' not found in checkpoint; keys are {list(ckpt)}")

    state_dict = {
        strip_wrapper_prefixes(k): v.to(dtype).contiguous()
        for k, v in ckpt[args.weights].items()
    }
    steps = ckpt.get("steps")
    del ckpt

    dead = [k for k in ZERO_INIT_PROBES
            if k in state_dict and not state_dict[k].abs().any()]
    if dead:
        raise RuntimeError(
            f"'{args.weights}' state looks untrained: {dead} are still all-zero "
            f"(zero-init values). Refusing to export. See GitHub issue #2."
        )

    total_params = sum(v.numel() for v in state_dict.values())
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"{len(state_dict)} tensors | {total_params / 1e9:.2f}B params | {total_bytes / 1e9:.2f} GB {args.dtype}")

    shards = shard_state_dict(state_dict, int(args.max_shard_size_gb * 1e9))
    n = len(shards)

    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        name = "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, os.path.join(args.out, name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name
        print(f"  wrote {name} ({len(shard)} tensors)")

    if n > 1:
        with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)

    config = asdict(PARAMS[args.model_size]())
    config.update({
        "architecture": "Flux2",
        "model_size": args.model_size,
        "torch_dtype": args.dtype,
        "weights": args.weights,
        "training_steps": steps,
    })
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done. Wrote {n} shard(s) and config.json to {args.out}")


if __name__ == "__main__":
    main()
