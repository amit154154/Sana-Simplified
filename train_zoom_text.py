#!/usr/bin/env python3
from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusers import (
    AutoencoderDC,
    SanaTransformer2DModel,
    SanaPipeline,
    FlowMatchEulerDiscreteScheduler,
)

from peft import LoraConfig, get_peft_model  # <--- NEW

from dataset.zoom_siglip import ZoomLatentMultiClassDataset
from models.zoom_siglip import SanaZoomLoRA

# ---- FAST BUT RISKY: enable tf32 + looser matmul precision ----
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

# -----------------------
# Basic config
# -----------------------
PRETRAINED_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"

DATA_ROOT = Path("/home/ubuntu/AMIT/data/dataset_zoom_30mb_notfiltered_10")
LABELPATH = Path("/home/ubuntu/AMIT/data/zoom_30mb_notfiltered_labels.csv")

LATENT_CACHE_DIR = Path("runs_sana/zoom_latent_cache")
text_cache_dir = Path("runs_sana/zoom30mb_text_encoding_cache")
SIGLIP_CACHE_DIR = None

BATCH_SIZE = 8
EPOCH_SIZE_STEPS = 100_000  # how many *batches* per "epoch" via sampler
MAX_STEPS = 12500  # total training steps (global_step)
VAL_INTERVAL_STEPS = 1000  # run validation every N train steps

IMAGE_LOG_EVERY_STEPS = 250
SAMPLES_DIR = Path("samples_text_zoom_nolora")
OUTPUT_DIR = Path("runs_sana/samples_text_zoom_nolora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = OUTPUT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

LR = 1e-4
USE_LORA = False
SIGLIP_DIM = None  # only used if you actually provide siglip_cache_dir
NUM_WORKERS = 4
PRECISION = "bf16-mixed"  # or "16-mixed" / "fp32"

WANDB_PROJECT = "sana15_zoom_text"
WANDB_RUN_NAME = "sana15_zoom_text_nolora"

# LoRA config (tune as you like)
lora_freeze_steps = 0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ['linear', 'linear_1', 'linear_2', 'to_q', 'to_k', 'to_v', 'proj_out']

# -----------------------
# Device / misc
# -----------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"[device] {DEVICE} | torch={torch.__version__}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

pl.seed_everything(432422)

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    PRETRAINED_MODEL,
    subfolder="scheduler",
)


def main():
    # -----------------------
    # 1) Load VAE & Transformer
    # -----------------------
    print("[init] loading SANA1.5 VAE + transformer...")
    vae = AutoencoderDC.from_pretrained(PRETRAINED_MODEL, subfolder="vae")
    transformer = SanaTransformer2DModel.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="transformer",
    )

    # -----------------------
    # 1a) Attach LoRA (PEFT) to transformer
    # -----------------------
    if USE_LORA:
        print("[init] attaching LoRA adapters via PEFT...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            # init_lora_weights="gaussian",
            # task_type="UNET",   # works fine for diffusers-style models
        )
        transformer = get_peft_model(transformer, lora_config)
    else:
        print("[init] USE_LORA = False -> full transformer frozen in SanaZoomLoRA")

    # Move to device / dtype *after* wrapping with LoRA
    use_bf16 = str(PRECISION).startswith("bf16") and (DEVICE == "cuda")
    dtype_target = torch.bfloat16 if use_bf16 else torch.float32

    vae = vae.to(DEVICE, dtype=torch.float32).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    transformer = transformer.to(DEVICE, dtype=dtype_target).train()

    # -----------------------
    # 2) Dataset + DataLoaders
    # -----------------------
    print("[init] building ZoomLatentOneClassDataset...")
    text_pipe = SanaPipeline.from_pretrained(
        PRETRAINED_MODEL,
        vae=None,
        transformer=None,
        torch_dtype=torch.float32,
    )

    ds = ZoomLatentMultiClassDataset(
        root=DATA_ROOT,
        cache_dir=LATENT_CACHE_DIR,
        vae=vae,
        text_pipe=text_pipe,
        labels_path=LABELPATH,
        text_cache_dir=text_cache_dir,

        # siglip_cache_dir=None,      # set Path(...) if you actually use SigLIP
        always_zero_org=True,
        allow_same_zoom=False,
    )

    # RandomSampler: fixed number of *batches* per epoch, with replacement
    train_sampler = RandomSampler(
        ds,
        replacement=True,
        num_samples=EPOCH_SIZE_STEPS,  # this is "steps per epoch"
    )

    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    # For validation we can just iterate once over the dataset
    val_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    # -----------------------
    # 3) Lightning module
    # -----------------------
    print("[init] building SanaZoomLoRA (zoom token version)...")
    model = SanaZoomLoRA(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        lr=LR,
        do_lora=USE_LORA,
        lora_freeze_steps=lora_freeze_steps,
        example_dir=SAMPLES_DIR,
        save_examples_every_steps=IMAGE_LOG_EVERY_STEPS,
        siglip_dim=None,
        instance_prompt=None,
        num_sample_steps=30,
        weighting_scheme="none",
        ckpt_root=OUTPUT_DIR / "weights",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[trainable params] {trainable:,}")

    # -----------------------
    # 4) Trainer (step-based)
    # -----------------------
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    trainer = pl.Trainer(
        max_steps=MAX_STEPS,
        max_epochs=1_000_000,  # effectively ignored once max_steps is hit
        accelerator="gpu" if DEVICE == "cuda" else "auto",
        devices=1,
        precision=PRECISION,
        logger=wandb_logger,
        log_every_n_steps=2,
        # val_check_interval=VAL_INTERVAL_STEPS,
        enable_progress_bar=True,
    )

    # -----------------------
    # 5) Train
    # -----------------------
    trainer.fit(model, train_dataloaders=train_loader)  # val_dataloaders=val_loader)
    print("Done.")


if __name__ == "__main__":
    main()


