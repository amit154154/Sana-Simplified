#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusers import (
    AutoencoderDC,
    SanaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
)

# --- your code (new local dataset + model) ---
from dataset.control_net import (
    ensure_mjhq_root,
    MJHQLocalCannyLatentTextDataset,
)
from models.control_net import SanaControlNetModel

import torch

# Enable TF32-style fast matmul on Tensor Cores (L40S etc.)
torch.set_float32_matmul_precision("medium")

# (optional but common if you're on CUDA)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# =========================
# CONFIG
# =========================

PRETRAINED_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"

IMAGE_SIZE = 512
CANNY_LOW = 100
CANNY_HIGH = 200

BATCH_SIZE = 4
acc_grad = 4
NUM_WORKERS = 4
MAX_STEPS = 50_000
LOG_EVERY_N_STEPS = 1

LR = 1e-4
WEIGHTING_SCHEME = "none"

OUTPUT_ROOT = Path("runs_sana_controlnet_mjhq_local")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_ROOT = OUTPUT_ROOT / "checkpoints"
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# logging / sampling
LOG_WANDB = True  # toggle wandb logging on/off
SAMPLE_EVERY = 400  # sample + log every N steps
NUM_INFERENCE_STEPS = 30  # sampling steps during logging
max_samples = None


# =========================
# DATA
# =========================

def build_dataloader():
    # ---- make sure MJHQ zip + meta are present and extracted ----
    imgs_root, meta_path = ensure_mjhq_root(CACHE_DIR)

    print("[sana] loading VAE & text pipeline for caching...")
    vae = AutoencoderDC.from_pretrained(PRETRAINED_MODEL, subfolder="vae").eval()
    text_pipe = SanaPipeline.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
    )

    # this will:
    #   * scan imgs_root (animals/art/...)
    #   * match images by hash with meta_data.json
    #   * build latents / canny / text-encoding caches under CACHE_DIR
    ds = MJHQLocalCannyLatentTextDataset(
        imgs_root=imgs_root,
        meta_path=meta_path,
        cache_dir=CACHE_DIR,
        vae=vae,
        text_pipe=text_pipe,
        image_size=IMAGE_SIZE,
        canny_low=CANNY_LOW,
        canny_high=CANNY_HIGH,
        max_sequence_length=300,
        max_samples=max_samples,  # or an int like 5_000 for faster experiments
    )

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    return dl, vae


# =========================
# MODEL
# =========================

def build_model(vae: AutoencoderDC):
    print("[sana] loading transformer + scheduler...")
    # train transformer in bf16 on GPU, fp32 otherwise
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    transformer = SanaTransformer2DModel.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="scheduler",
    )

    lit_model = SanaControlNetModel(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        lr=LR,
        weighting_scheme=WEIGHTING_SCHEME,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        ckpt_root=CKPT_ROOT,
        # logging / sampling
        log_wandb=LOG_WANDB,
        sample_every=SAMPLE_EVERY,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    return lit_model


# =========================
# TRAINER
# =========================

def build_trainer():
    logger = None
    if LOG_WANDB:
        logger = WandbLogger(
            project="sana_canny_mjhq",
            name="sana_canny_controlnet_local",
            save_dir=str(OUTPUT_ROOT),
        )

    trainer = pl.Trainer(
        default_root_dir=str(OUTPUT_ROOT),
        accelerator="auto",  # cuda / mps / cpu
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        max_steps=MAX_STEPS,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=None,  # no val for now
        logger=logger,
        accumulate_grad_batches=acc_grad,  # <-- or 16 if BATCH_SIZE=1

    )
    return trainer


# =========================
# MAIN
# =========================

def main():
    dl_train, vae = build_dataloader()
    lit_model = build_model(vae)
    trainer = build_trainer()
    trainer.fit(lit_model, train_dataloaders=dl_train)


if __name__ == "__main__":
    main()