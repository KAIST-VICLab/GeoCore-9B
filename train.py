import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.flux2 import Flux2, TerraNova4BParams, TerraNova9BParams
from loss import SILoss_Flux2
from utils import load_encoders

from data.dataset import Git10M_T2I
from safetensors.torch import load_file
from models.vae_flux2 import AutoEncoder, AutoEncoderParams
from models.text_encoder import TextEncoder

# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

import shutil
import glob
import re

torch.set_float32_matmul_precision('high')


def cleanup_old_states(checkpoint_dir, prefix="state-", keep_limit=1):
    folders = glob.glob(os.path.join(checkpoint_dir, f"{prefix}*"))

    def extract_step(path):
        match = re.search(rf"{prefix}(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    folders.sort(key=extract_step)

    if len(folders) > keep_limit:
        folders_to_delete = folders[:-keep_limit]
        for folder in folders_to_delete:
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Error: {folder}, {e}")


logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

DINO_SAT_MEAN = (0.430, 0.411, 0.296)
DINO_SAT_STD = (0.213, 0.156, 0.143)


def preprocess_raw_image(x, enc_type, resolution=256):
    resolution = x.shape[-1]
    if 'dinov3' in enc_type:
        x = x * 0.5 + 0.5
        x = Normalize(DINO_SAT_MEAN, DINO_SAT_STD)(x)
        x = torch.nn.functional.interpolate(x, 256 * (resolution // 256), mode='bicubic')
    elif 'dinov2' in enc_type:
        x = x * 0.5 + 0.5
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x * 0.5 + 0.5
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()

        for name, param in msd.items():
            clean_name = name.replace('_orig_mod.', '')

            if clean_name in esd:
                esd[clean_name].copy_(
                    esd[clean_name] * decay + param.detach().data * (1 - decay)
                )


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
    latent_size = args.resolution // 16
    img_ids = _prepare_latent_image_ids(latent_size, latent_size, device, weight_dtype)

    if args.enc_type != 'None':
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, dinov3_repo=args.dinov3_repo)
        encoders = [enc.to(dtype=weight_dtype) for enc in encoders]
        for enc in encoders:
            enc.eval()
            requires_grad(enc, False)
    else:
        encoders, encoder_types, architectures = [None], [None], [None]

    model = Flux2(TerraNova9BParams()).to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # FLUX 2.0 VAE
    vae_params = AutoEncoderParams()
    vae = AutoEncoder(vae_params)
    state_dict = load_file(args.vae_dir)
    vae.load_state_dict(state_dict)
    vae = vae.to(device, dtype=weight_dtype)
    vae.eval()
    requires_grad(vae, False)

    # T5 and CLIP text encoder
    text_encoder = TextEncoder(device=device, dtype=weight_dtype)
    text_encoder.encoder_clip.eval()
    text_encoder.encoder_t5.eval()
    requires_grad(text_encoder.encoder_clip, False)
    requires_grad(text_encoder.encoder_t5, False)

    # create loss function
    loss_fn = SILoss_Flux2(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data:
    train_dataset = Git10M_T2I(path=args.data_dir, cfg=(args.cfg_prob > 0), p_uncond=args.cfg_prob).train
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing = True
        logger.info("Gradient Checkpointing enabled manually.")

    ema.eval()  # EMA model should always be in eval mode

    if hasattr(torch, "compile"):
        if accelerator.is_main_process:
            logger.info("Start torch.compile")
        model = torch.compile(model)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    if args.resume_step > 0:
        resume_path = os.path.join(checkpoint_dir, f"state-{args.resume_step}")

        if os.path.exists(resume_path):
            if accelerator.is_main_process:
                logger.info(f"Resuming from checkpoint folder: {resume_path}")

            accelerator.load_state(resume_path)
            global_step = args.resume_step

            ckpt_name = str(args.resume_step).zfill(7) + '.pt'
            ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
            if os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                ema.load_state_dict(ckpt['ema'])
                if accelerator.is_main_process:
                    logger.info(f"EMA weights successfully loaded from {ckpt_file}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Checkpoint folder {resume_path} not found! Check your directory structure.")

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="GeoCore",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_xs, gt_zs, xT, sample_ys = None, None, None, None
    sample_kwargs = None
    null_kwargs = None

    for epoch in range(args.epochs):
        model.train()
        for x, cap, meta in train_dataloader:
            x = x.to(device, dtype=weight_dtype)
            prompt_embeds, pooled_embeds, text_ids = text_encoder(cap)
            meta_res = meta["res"].to(device, dtype=weight_dtype)
            meta_lon = meta["lon"].to(device, dtype=weight_dtype)
            meta_lat = meta["lat"].to(device, dtype=weight_dtype)

            if sample_kwargs is None:
                actual_num = min(sample_batch_size, x.shape[0])
                gt_xs = x[:actual_num].clone()
                sample_caps = cap[:actual_num]
                with torch.no_grad():
                    gt_zs = vae.encode(gt_xs)

                xT = torch.randn_like(gt_zs)
                sample_prompt, sample_pooled, sample_ids = text_encoder(sample_caps)

                sample_kwargs = {
                    "x_ids": img_ids.unsqueeze(0).repeat(actual_num, 1, 1),
                    "ctx": sample_prompt,
                    "ctx_ids": sample_ids,
                    "y": sample_pooled,
                    "res": meta_res[:actual_num].clone(),
                    "lon": meta_lon[:actual_num].clone(),
                    "lat": meta_lat[:actual_num].clone(),
                }

                null_kwargs = text_encoder.get_null_info_flux2(actual_num)
                null_kwargs["ctx_ids"] = sample_ids
                null_kwargs["x_ids"] = img_ids

            with torch.no_grad():
                z = vae.encode(x)
                prompt_embeds, pooled_embeds, text_ids = text_encoder(cap)
                z_repa = None
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        x_ = preprocess_raw_image(x, encoder_type, resolution=args.resolution)
                        z_repa = encoder.forward_features(x_.to(weight_dtype))
                        if 'dinov3' in encoder_type: z_repa = z_repa['x_norm_patchtokens']
                        if 'dinov2' in encoder_type: z_repa = z_repa['x_norm_patchtokens']

            with accelerator.accumulate(model):
                model_kwargs = {
                    "x_ids": img_ids.repeat(x.shape[0], 1, 1),
                    "ctx": prompt_embeds,
                    "ctx_ids": text_ids,
                    "y": pooled_embeds,
                    "res": meta_res,
                    "lon": meta_lon,
                    "lat": meta_lat
                }
                diff_loss, proj_loss = loss_fn(model, z, model_kwargs, z_repa_target=z_repa)
                diff_loss_mean = diff_loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = diff_loss_mean + proj_loss_mean * args.proj_coeff

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)  # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                accelerator.wait_for_everyone()

                checkpoint_state_dir = os.path.join(checkpoint_dir, f"state-{global_step}")
                accelerator.save_state(checkpoint_state_dir)

                if accelerator.is_main_process:
                    cleanup_old_states(checkpoint_dir, prefix="state-", keep_limit=1)
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "model": unwrapped_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler_flux2
                with torch.no_grad():
                    samples = euler_sampler_flux2(
                        model,
                        xT,
                        cond_kwargs=sample_kwargs,
                        uncond_kwargs=null_kwargs,
                        num_steps=50,
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = samples.to(weight_dtype)
                    samples = vae.decode(samples)
                    gt_samples = gt_xs
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            grad_norm_tensor = torch.tensor(grad_norm, device=device)
            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
                "diff_loss": accelerator.gather(diff_loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm_tensor).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--vae-dir", type=str, required=True, help="Path to the Flux2 VAE ae.safetensors")
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)

    # dataset
    parser.add_argument("--data-dir", type=str, required=True, help="Git-10M dataset root (HF datasets format)")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=1024)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=300000)
    parser.add_argument("--checkpointing-steps", type=int, default=15000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.001, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-15, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=16)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])  # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov3-vit-7b')
    parser.add_argument("--dinov3-repo", type=str, default=None,
                        help="Local clone of facebookresearch/dinov3 with sat493m weights (GSA teacher). Defaults to $DINOV3_REPO.")
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)