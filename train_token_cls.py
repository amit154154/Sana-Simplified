#!/usr/bin/env python3
"""
train_token_sana15.py

Token-style DreamBooth / Textual-Inversion for Sana-1.5:

- Dataset: instance images (your dog / concept) -> VAE latent cache.
- Text:
    * INSTANCE_PROMPT = "a <sks> dog"
- Model:
    * SanaDreamBoothToken: transformer is frozen, only a learnable token delta.
- Loss:
    * Flow-matching loss (same as HF official Sana trainer), but only updating the token.
"""

from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader, random_split, RandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusers import (
    AutoencoderDC,
    SanaTransformer2DModel,
    SanaPipeline,
    FlowMatchEulerDiscreteScheduler,
)

# <<< your model >>>
from models.token_cls import SanaDreamBoothToken
from dataset.token_cls import DreamBoothLatentDataset


# ---- FAST BUT RISKY: enable tf32 + looser matmul precision ----
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")


# -----------------------
# Basic config
# -----------------------
DATA_ROOT = Path("data/dreambooth/dog")  # folder of your concept images
OUTPUT_DIR = Path("runs/sana15_token_dreambooth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"

INSTANCE_PROMPT = "a <sks> dog"

# word/text exploration prompt:
WORD_PROMPT = (
    "a white poster with a single unknown word written in big bold black letters in the center, "
    "no other text, minimal design"
)
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
precision = "bf16-mixed"

VAL_FRACTION = 0.0
SEED = 69420

NUM_WORKERS = 4
EPOCH_SIZE = 10_000  # 0 -> full dataset per epoch

RESOLUTION = 1024
TOKEN_INDEX = 0
GUIDANCE_SCALE = 1.0

# Flow-matching weighting config (as in official)
WEIGHTING_SCHEME = "none"  # try "logit_normal" or "mode" later if you want
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29

# sampling / logging config
TOKEN_SAVE_EVERY_STEPS = 200
SAMPLE_EVERY_STEPS = 40            # generic samples (training prompt)
NUM_SAMPLE_IMAGES = 1
WORD_LOG_EVERY_STEPS = 100         # word-generation samples
# ---- WandB ----
LOG_WANDB = True
WANDB_PROJECT = "sana15-token-dreambooth"
WANDB_RUN_NAME = "sana15_<sks>_token"


# -----------------------
# Device / seed
# -----------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"[device] {DEVICE} | torch={torch.__version__}")
pl.seed_everything(SEED)


# -----------------------
# Text encoder: get base pe/pam for the instance prompt
# and also for the word prompt
# -----------------------
print("[init] loading Sana-1.5 text encoder pipeline...")
text_pipe = SanaPipeline.from_pretrained(
    PRETRAINED_MODEL,
    transformer=None,
    vae=None,
    torch_dtype=torch.float32,
)
text_pipe = text_pipe.to(DEVICE)

MAX_SEQUENCE_LENGTH = 128

with torch.no_grad():
    pe_base, pam_base, _, _ = text_pipe.encode_prompt(
        prompt=INSTANCE_PROMPT,
        do_classifier_free_guidance=False,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        complex_human_instruction=None,
    )
    pe_word, pam_word, _, _ = text_pipe.encode_prompt(
        prompt=WORD_PROMPT,
        do_classifier_free_guidance=False,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        complex_human_instruction=None,
    )

pe_base = pe_base.cpu().detach()                    # [1, T, D]
pam_base = pam_base.cpu().to(torch.bool).detach()   # [1, T]
pe_word = pe_word.cpu().detach()                    # [1, T, D]
pam_word = pam_word.cpu().to(torch.bool).detach()   # [1, T]

del text_pipe
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

print("[text] base pe shape:", pe_base.shape, "pam shape:", pam_base.shape)
print("[text] word pe shape:", pe_word.shape, "pam_word shape:", pam_word.shape)


# -----------------------
# VAE + dataset (latent cache)
# -----------------------
print("[init] loading Sana-1.5 VAE...")
vae = AutoencoderDC.from_pretrained(PRETRAINED_MODEL, subfolder="vae")
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)
vae = vae.to(DEVICE, dtype=torch.float32)

LATENT_CACHE_DIR = OUTPUT_DIR / "latent_cache"

print(f"[data] building DreamBoothLatentDataset from {DATA_ROOT}...")
full_ds = DreamBoothLatentDataset(
    root=DATA_ROOT,
    cache_dir=LATENT_CACHE_DIR,
    vae=vae,
    size=RESOLUTION,
    indices=None,
    device=DEVICE,
)

n_total = len(full_ds)
n_val = int(n_total * VAL_FRACTION)
n_train = n_total - n_val

if n_val > 0:
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
else:
    train_ds, val_ds = full_ds, None

# -----------------------
# Dataloaders with optional EPOCH_SIZE
# -----------------------
if EPOCH_SIZE > 0:
    print(f"[data] using RandomSampler with EPOCH_SIZE={EPOCH_SIZE}")
    train_sampler = RandomSampler(
        train_ds,
        replacement=True,
        num_samples=EPOCH_SIZE,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    effective_train_len = EPOCH_SIZE
else:
    print("[data] using full train_ds per epoch")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    effective_train_len = len(train_ds)

if val_ds is not None:
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
else:
    val_loader = None

print(
    f"[data] train={effective_train_len} items/epoch | "
    f"raw_train={len(train_ds)} | val={len(val_ds) if val_ds is not None else 0}"
)


# -----------------------
# Transformer (base) – passed into SanaDreamBoothToken
# -----------------------
print("[init] loading Sana-1.5 transformer...")
transformer = SanaTransformer2DModel.from_pretrained(
    PRETRAINED_MODEL,
    subfolder="transformer",
)

use_bf16 = str(precision).startswith("bf16")
dtype_target = torch.bfloat16 if (use_bf16 and DEVICE == "cuda") else torch.float32
transformer = transformer.to(DEVICE, dtype=dtype_target).train()


# -----------------------
# Scheduler (Sana-1.5 Flow-Matching)
# -----------------------
print("[init] loading Sana-1.5 scheduler...")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    PRETRAINED_MODEL,
    subfolder="scheduler",
)


# -----------------------
# Model: token-only DreamBooth (flow matching)
# -----------------------
model = SanaDreamBoothToken(
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    pe_base=pe_base[0],
    pam_base=pam_base[0],
    token_index=TOKEN_INDEX,
    lr=LEARNING_RATE,
    guidance_scale=GUIDANCE_SCALE,
    weighting_scheme=WEIGHTING_SCHEME,
    logit_mean=LOGIT_MEAN,
    logit_std=LOGIT_STD,
    mode_scale=MODE_SCALE,
    use_8bit_adam=True,
    # sampling
    sample_every_n_steps=SAMPLE_EVERY_STEPS,
    num_sample_images=NUM_SAMPLE_IMAGES,
    sample_num_inference_steps=30,
    sample_dir=OUTPUT_DIR / "samples",
    sample_seed=42,
    # word-gen logging
    pe_word=pe_word[0],
    pam_word=pam_word[0],
    log_word_every_n_steps=WORD_LOG_EVERY_STEPS,
    log_word_dir=OUTPUT_DIR / "word_samples",
    log_word_prompt=WORD_PROMPT,
    # NEW: token embedding snapshots
    save_token_every_n_steps=TOKEN_SAVE_EVERY_STEPS,
    token_save_dir=OUTPUT_DIR / "token_snapshots",
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[trainable params] {trainable:,} (should be just the token delta)")


# -----------------------
# Trainer with W&B logger
# -----------------------
accelerator = "gpu" if DEVICE == "cuda" else "cpu"
wandb_logger = (
    WandbLogger(project=WANDB_PROJECT, name=WANDB_RUN_NAME) if LOG_WANDB else False
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=accelerator,
    devices=1,
    precision=precision,
    logger=wandb_logger,
    enable_progress_bar=True,
    log_every_n_steps=5,
    accumulate_grad_batches=1,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("Done.")