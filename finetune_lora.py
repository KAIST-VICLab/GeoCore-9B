"""LoRA fine-tuning of GeoCore-9B.

Two modes:
  * text-to-image (default) -- adapts the pre-trained model on a new T2I corpus.
  * image-conditioned (--cond-image) -- expands `img_in` with zero-initialised
    channels so a conditioning latent can be concatenated, which is how the
    downstream cloud-removal and SAR-to-optical adaptations in the paper are set up.
    Supply the paired dataset with --dataset-class (see README).

GSA is a pre-training-only objective, so the DINOv3-Sat teacher is never loaded here.
"""
import argparse
import copy
import importlib
import glob
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from models.flux2 import Flux2, TerraNova9BParams
from models.vae_flux2 import AutoEncoder, AutoEncoderParams
from models.text_encoder import TextEncoder
from loss import SILoss_Flux2_no_repa
from data.dataset_finetune import Git10M_T2I

torch.set_float32_matmul_precision('high')

logger = get_logger(__name__)

LORA_TARGET_MODULES = ["qkv", "proj", "linear1", "linear2"]


def expand_flux_img_in_zero_init(model, new_in_channels):
    """Widen `img_in` so a conditioning latent can be channel-concatenated.

    The extra columns start at zero, so the expanded model is initially identical
    to the pre-trained one.
    """
    old_img_in = model.img_in
    old_in_channels = old_img_in.in_features
    hidden_size = old_img_in.out_features

    new_img_in = nn.Linear(new_in_channels, hidden_size, bias=False)
    with torch.no_grad():
        new_img_in.weight[:, :old_in_channels] = old_img_in.weight.data.clone()
        nn.init.zeros_(new_img_in.weight[:, old_in_channels:])

    model.img_in = new_img_in
    model.in_channels = new_in_channels
    return model


def load_dataset_class(spec):
    """Resolve a "package.module:ClassName" string to the class object."""
    module_path, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"--dataset-class must look like 'my_pkg.my_module:MyDataset', got {spec!r}")
    return getattr(importlib.import_module(module_path), class_name)


def cleanup_old_states(checkpoint_dir, prefix="state-", keep_limit=2):
    folders = glob.glob(os.path.join(checkpoint_dir, f"{prefix}*"))

    def extract_step(path):
        match = re.search(rf"{prefix}(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    folders.sort(key=extract_step)
    for folder in folders[:-keep_limit]:
        shutil.rmtree(folder, ignore_errors=True)


def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def log_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")


def main(args):
    logging_dir = Path(args.output_dir, args.exp_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=os.path.join(args.output_dir, args.exp_name), logging_dir=logging_dir
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

    global logger
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
    latent_size = args.resolution // 16
    img_ids = _prepare_latent_image_ids(latent_size, latent_size, device, weight_dtype)

    model = Flux2(TerraNova9BParams()).to(device)

    ckpt = torch.load(args.base_ckpt, map_location='cpu', weights_only=False)
    if 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if accelerator.is_main_process:
        logger.info(f"Loaded base checkpoint from {args.base_ckpt}")

    base_in_channels = model.img_in.in_features
    if args.cond_image:
        model = expand_flux_img_in_zero_init(model, new_in_channels=base_in_channels * 2)
        model = model.to(device)

    requires_grad(model, False)
    if args.cond_image:
        model.img_in.weight.requires_grad = True

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        log_trainable_parameters(model)

    vae = AutoEncoder(AutoEncoderParams())
    vae.load_state_dict(load_file(args.vae_dir))
    vae = vae.to(device, dtype=weight_dtype)
    vae.eval()
    requires_grad(vae, False)

    text_encoder = TextEncoder(device=device, dtype=weight_dtype)
    text_encoder.encoder_clip.eval()
    text_encoder.encoder_t5.eval()
    requires_grad(text_encoder.encoder_clip, False)
    requires_grad(text_encoder.encoder_t5, False)

    loss_fn = SILoss_Flux2_no_repa(
        prediction=args.prediction,
        path_type=args.path_type,
        accelerator=accelerator,
        weighting=args.weighting,
    )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataset_class:
        train_dataset = load_dataset_class(args.dataset_class)(args.data_dir)
    else:
        train_dataset = Git10M_T2I(
            args.data_dir, cfg=True, p_uncond=args.cfg_prob, score_threshold=args.score_threshold
        ).train

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model.train()
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    if args.resume_step > 0:
        resume_path = os.path.join(checkpoint_dir, f"state-{args.resume_step}")
        accelerator.load_state(resume_path)
        global_step = args.resume_step

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="GeoCore-LoRA",
            config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def save_adapter(step):
        unwrapped_model = accelerator.unwrap_model(model)
        lora_save_path = os.path.join(checkpoint_dir, f"lora_{step:07d}")
        unwrapped_model.save_pretrained(lora_save_path)
        if args.cond_image:
            # img_in is trained outside LoRA, so it is not covered by save_pretrained
            torch.save(unwrapped_model.base_model.model.img_in.state_dict(),
                       os.path.join(lora_save_path, "img_in_weight.pt"))
        logger.info(f"Saved LoRA adapter to {lora_save_path}")

    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in train_dataloader:
            if args.cond_image:
                cond_img, target_img, cap, meta = batch
                cond_img = cond_img.to(device, dtype=weight_dtype)
            else:
                target_img, cap, meta = batch
                cond_img = None
            target_img = target_img.to(device, dtype=weight_dtype)

            with torch.no_grad():
                z_target = vae.encode(target_img)
                z_cond = vae.encode(cond_img) if cond_img is not None else None
                prompt_embeds, pooled_embeds, text_ids = text_encoder(cap)

            with accelerator.accumulate(model):
                model_kwargs = {
                    "x_ids": img_ids.repeat(target_img.shape[0], 1, 1),
                    "ctx": prompt_embeds,
                    "ctx_ids": text_ids,
                    "y": pooled_embeds,
                    "res": meta["res"].to(device, dtype=weight_dtype),
                    "lon": meta["lon"].to(device, dtype=weight_dtype),
                    "lat": meta["lat"].to(device, dtype=weight_dtype),
                }
                if z_cond is not None:
                    model_kwargs["z_cond"] = z_cond

                loss_mean = loss_fn(model, z_target, model_kwargs).mean()

                accelerator.backward(loss_mean)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": accelerator.gather(loss_mean).mean().detach().item()}
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    accelerator.save_state(os.path.join(checkpoint_dir, f"state-{global_step}"))
                    if accelerator.is_main_process:
                        cleanup_old_states(checkpoint_dir)
                        save_adapter(global_step)

                if global_step >= args.max_train_steps:
                    done = True
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_adapter(global_step)
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of GeoCore-9B")

    parser.add_argument("--output-dir", type=str, default="exps_lora")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--resume-step", type=int, default=0)

    parser.add_argument("--base-ckpt", type=str, required=True,
                        help="Pre-trained GeoCore-9B checkpoint (.pt); EMA weights are used when present")
    parser.add_argument("--vae-dir", type=str, required=True, help="Path to the Flux2 VAE ae.safetensors")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--cond-image", action="store_true",
                        help="Image-conditioned adaptation: widen img_in with zero-init channels and "
                             "concatenate a conditioning latent (cloud removal, SAR-to-optical, ...)")

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset-class", type=str, default=None,
                        help="'module.path:ClassName' of a custom Dataset taking the data root as its only "
                             "positional argument. Required with --cond-image; defaults to Git-10M otherwise.")
    parser.add_argument("--score-threshold", type=float, default=4.8,
                        help="Git-10M image-quality filter (ignored with --dataset-class)")
    parser.add_argument("--cfg-prob", type=float, default=0.1,
                        help="Classifier-free guidance dropout (ignored with --dataset-class)")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=50000)
    parser.add_argument("--checkpointing-steps", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--weighting", type=str, default="uniform")

    args = parser.parse_args(input_args)
    if args.cond_image and not args.dataset_class:
        parser.error("--cond-image requires --dataset-class returning (cond_img, target_img, caption, meta)")
    return args


if __name__ == "__main__":
    main(parse_args())
