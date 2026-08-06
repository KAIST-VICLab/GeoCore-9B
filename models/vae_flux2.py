"""Flux.2 autoencoder.

Adapted from the FLUX reference implementation by Black Forest Labs
(https://github.com/black-forest-labs/flux), licensed under Apache-2.0.

Used frozen in GeoCore-9B. The pre-trained VAE weights are distributed
separately by Black Forest Labs; obtain them from `FLUX.2-klein-base-4B`
(`vae/diffusion_pytorch_model.safetensors`), which is Apache-2.0 and ungated.
`load_vae_state_dict` reads that file directly -- see the note under
"Checkpoint loading" below for why the diffusers-format copy is preferred over
the identical weights shipped in the FLUX.2-dev repository.
"""
import math
import os
from dataclasses import dataclass, field

import torch
from einops import rearrange
from safetensors.torch import load_file
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec


# ---------------------------------------------------------------------------
# Checkpoint loading
#
# The same autoencoder is published twice by Black Forest Labs:
#
#   black-forest-labs/FLUX.2-dev :: ae.safetensors
#       BFL key names, fp32, 336 MB, FLUX Non-Commercial License v2.1, gated.
#       4(a)(iii) of that license forbids "surveillance purposes, including all
#       research and development related to surveillance", which is not a
#       question an Earth-observation model should have to carry.
#
#   black-forest-labs/FLUX.2-klein-base-4B :: vae/diffusion_pytorch_model.safetensors
#       diffusers key names, bf16, 168 MB, Apache-2.0, ungated.
#
# The two hold the same 251 tensors / 84,046,372 parameters and differ only in
# serialisation: key names, dtype, and eight attention projections stored as
# (512, 512) linears rather than (512, 512, 1, 1) 1x1 convolutions. The rename
# rules below were not written by hand from the naming conventions; they were
# derived by pairing every tensor of the two files ON VALUE, so no key can be
# silently mismatched. 250 of 251 tensors paired one-to-one with a worst
# absolute deviation of 7.802e-03, which is bf16 rounding of the source file.
# The single tensor that does not pair is `bn.num_batches_tracked`, a BatchNorm
# step counter that `normalize`/`inv_normalize` never read.
#
# GeoCore-9B is therefore usable end to end under Apache-2.0. Point `--vae-dir`
# at the Klein-4B file; the FLUX.2-dev copy still loads unchanged for anyone who
# already has it.
# ---------------------------------------------------------------------------

#: diffusers attention/normalisation leaf names -> BFL ones.
_ATTN_RENAMES = (
    ("to_out.0", "proj_out"),
    ("to_q", "q"),
    ("to_k", "k"),
    ("to_v", "v"),
    ("group_norm", "norm"),
)


def _rename_leaf(tail: str) -> str:
    for src, dst in _ATTN_RENAMES:
        if tail == src or tail.startswith(src + "."):
            return dst + tail[len(src):]
    return tail


def _split_index(rest: str, prefix: str) -> tuple[int, str]:
    """``resnets.2.conv1.weight`` with prefix ``resnets.`` -> ``(2, "conv1.weight")``."""
    idx, tail = rest[len(prefix):].split(".", 1)
    return int(idx), tail


def _diffusers_key_to_bfl(key: str, num_levels: int) -> str:
    """Translate one diffusers autoencoder key into the BFL name for it."""
    if key.startswith("bn."):
        return key
    if key.startswith("quant_conv."):
        return "encoder." + key
    if key.startswith("post_quant_conv."):
        return "decoder." + key

    stem, rest = key.split(".", 1)
    if stem not in ("encoder", "decoder"):
        raise KeyError(f"unrecognised autoencoder key: {key}")

    if rest.startswith("conv_norm_out."):
        rest = "norm_out." + rest[len("conv_norm_out."):]

    elif rest.startswith("mid_block."):
        body = rest[len("mid_block."):]
        if body.startswith("resnets."):
            i, tail = _split_index(body, "resnets.")
            body = f"block_{i + 1}.{tail}"
        elif body.startswith("attentions."):
            i, tail = _split_index(body, "attentions.")
            body = f"attn_{i + 1}.{_rename_leaf(tail)}"
        else:
            raise KeyError(f"unrecognised autoencoder key: {key}")
        rest = "mid." + body

    elif rest.startswith("down_blocks."):
        i, body = _split_index(rest, "down_blocks.")
        if body.startswith("resnets."):
            j, tail = _split_index(body, "resnets.")
            body = f"block.{j}.{tail.replace('conv_shortcut', 'nin_shortcut')}"
        elif body.startswith("downsamplers."):
            _, tail = _split_index(body, "downsamplers.")
            body = f"downsample.{tail}"
        else:
            raise KeyError(f"unrecognised autoencoder key: {key}")
        rest = f"down.{i}.{body}"

    elif rest.startswith("up_blocks."):
        # diffusers counts up_blocks from the lowest resolution, the BFL decoder
        # indexes `up` by level, so the order is reversed.
        i, body = _split_index(rest, "up_blocks.")
        level = num_levels - 1 - i
        if body.startswith("resnets."):
            j, tail = _split_index(body, "resnets.")
            body = f"block.{j}.{tail.replace('conv_shortcut', 'nin_shortcut')}"
        elif body.startswith("upsamplers."):
            _, tail = _split_index(body, "upsamplers.")
            body = f"upsample.{tail}"
        else:
            raise KeyError(f"unrecognised autoencoder key: {key}")
        rest = f"up.{level}.{body}"

    return f"{stem}.{rest}"


def convert_diffusers_vae_state_dict(
    state_dict: dict[str, Tensor], params: AutoEncoderParams | None = None
) -> dict[str, Tensor]:
    """Convert a diffusers-format Flux.2 autoencoder state dict to BFL names.

    Every produced key is checked against `AutoEncoder`'s own state dict, and a
    tensor is reshaped only when the target adds singleton dimensions -- the
    (512, 512) linear / (512, 512, 1, 1) convolution difference. Anything that
    does not line up exactly is an error rather than a silent partial load.
    """
    params = params or AutoEncoderParams()
    target = AutoEncoder(params).state_dict()
    num_levels = len(params.ch_mult)

    out: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        name = _diffusers_key_to_bfl(key, num_levels)
        if name not in target:
            raise KeyError(f"converted key {key!r} -> {name!r} is not in AutoEncoder")
        if name in out:
            raise KeyError(f"two source keys map to {name!r}")
        want = target[name].shape
        if value.shape != want:
            if tuple(value.shape) != tuple(s for s in want if s != 1):
                raise ValueError(f"{key!r} -> {name!r}: shape {tuple(value.shape)} != {tuple(want)}")
            value = value.reshape(want)
        out[name] = value.contiguous()

    missing = sorted(set(target) - set(out))
    # The step counter is absent from some exports; it is unused at eval time.
    for name in list(missing):
        if name.endswith("num_batches_tracked"):
            out[name] = target[name].clone()
            missing.remove(name)
    if missing:
        raise KeyError(f"conversion left {len(missing)} parameters unfilled: {missing[:5]}")
    return out


def load_vae_state_dict(
    path: str, params: AutoEncoderParams | None = None
) -> dict[str, Tensor]:
    """Load Flux.2 autoencoder weights in either the diffusers or BFL layout.

    `path` may be a safetensors file or a directory holding one (a downloaded
    `FLUX.2-klein-base-4B` snapshot, its `vae/` subfolder, or a bare
    `ae.safetensors`).
    """
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "vae", "diffusion_pytorch_model.safetensors"),
            os.path.join(path, "diffusion_pytorch_model.safetensors"),
            os.path.join(path, "ae.safetensors"),
        ]
        found = next((c for c in candidates if os.path.isfile(c)), None)
        if found is None:
            raise FileNotFoundError(f"no autoencoder safetensors under {path}")
        path = found

    state_dict = load_file(path)
    is_diffusers = any(k.startswith(("encoder.down_blocks.", "decoder.up_blocks.")) for k in state_dict)
    return convert_diffusers_vae_state_dict(state_dict, params) if is_diffusers else state_dict


def load_autoencoder(
    path: str,
    params: AutoEncoderParams | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AutoEncoder:
    """Build the frozen Flux.2 autoencoder from `path` and put it in eval mode."""
    params = params or AutoEncoderParams()
    vae = AutoEncoder(params)
    vae.load_state_dict(load_vae_state_dict(path, params), strict=True)
    return vae.to(device=device, dtype=dtype).eval()