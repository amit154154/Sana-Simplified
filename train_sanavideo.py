#!/usr/bin/env python3
"""
train_sana_tomjerry.py

- Loads SANA-Video.
- Encodes a single "Tom & Jerry" class prompt once and caches the embedding.
- If the embedding file already exists:
    * loads ONLY the transformer submodule (no full pipeline).
- Uses TomJerryLatentDataset (latents only) and trains the transformer with
  a Rectified Flow / Flow-Matching style velocity loss.
- Uses PyTorch Lightning, optional W&B logging, and LoRA on the transformer.
- Training is controlled by max_steps (not epochs).
"""

from __future__ import annotations
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusers import SanaVideoPipeline, SanaVideoTransformer3DModel
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict

from dataset.sana_video import TomJerryLatentDataset
import lovely_tensors as lt
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

lt.monkey_patch()

torch.set_float32_matmul_precision("medium")

# (optional but common if you're on CUDA)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==========================
# bitsandbytes (AdamW 8bit)
# ==========================
try:
    from bitsandbytes.optim import AdamW8bit

    HAS_BNB = True
    print("Using bitsandbytes AdamW8bit optimizer.")
except Exception:
    HAS_BNB = False
    from torch.optim import AdamW

    print("bitsandbytes not available, falling back to torch.optim.AdamW.")

# ==========================
# CONFIG
# ==========================

MODEL_ID = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"

# Latent cache folder (where clip_*.pt live)
CACHE_DIR = Path("/home/ubuntu/AMIT/video/data/full_video_latents")

# Where to store the cached prompt embedding
PROMPT_EMBEDS_PATH = Path("/home/ubuntu/AMIT/video/data/tom_jerry_prompt_embeds.pt")

DEVICE = "cuda"
DTYPE_MODEL = torch.bfloat16

BATCH_SIZE = 8
LR = 2e-4
MAX_STEPS = 10_000  # <--- control training length via steps
LOG_INTERVAL = 4

SAVE_DIR = Path("/home/ubuntu/AMIT/video/checkpoints_0")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# If not None and file exists, we will load these LoRA weights into the transformer
LORA_INIT_PATH = SAVE_DIR / "lora_step_000200.pt"  # change if you want a different file, or set to None

# how often to save LoRA weights (in steps).
# 0 → only save final LoRA at the end.
SAVE_LORA_EVERY = 50

# W&B options
DO_WANDB = True  # <--- set True if you want logging
WANDB_PROJECT = "sana-video-tomjerry"
WANDB_RUN_NAME = "sana_tomjerry_lora"

# LoRA options
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
# adjust if needed to match actual module names
LORA_TARGET_MODULES = ["proj_out", "to_q", "to_v", "to_k", "linear_2", "linear_1", "linear"]

# Single global class prompt
CLASS_PROMPT = (
    "A vintage slapstick 2D cartoon scene of a grey cat chasing a small brown mouse in a colorful house, "
    "Tom and Jerry style, bold outlines, limited color palette, exaggerated expressions, smooth character motion."
)


# ==========================
# STEP OFFSET FROM INIT LORA
# ==========================

def infer_initial_step_from_path(path: Path | None) -> int:
    """
    Infer numeric step from filename like 'lora_step_000200.pt'.
    Returns 0 if no path or cannot parse.
    """
    if path is None or (not path.exists()):
        return 0
    stem = path.stem  # e.g. 'lora_step_000200'
    m = re.search(r"(\d+)$", stem)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


INITIAL_STEP = infer_initial_step_from_path(LORA_INIT_PATH)
print(f"INITIAL_STEP inferred from LORA_INIT_PATH: {INITIAL_STEP}")


# ==========================
# HELPERS: PROMPT EMBEDS / TRANSFORMER
# ==========================

def prepare_prompt_embeds() -> torch.Tensor:
    """
    Prepare or load the cached prompt embeddings.

    Returns:
        prompt_embeds: [1, L, D] on DEVICE, dtype=DTYPE_MODEL
    """
    if PROMPT_EMBEDS_PATH.exists():
        print(f"Found cached prompt embeddings at: {PROMPT_EMBEDS_PATH}")
        prompt_embeds = torch.load(PROMPT_EMBEDS_PATH, map_location="cpu")
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        return prompt_embeds.to(DEVICE, dtype=DTYPE_MODEL)

    # No cache → load pipeline and encode once
    print("No cached prompt embeddings found. Loading full SANA-Video pipeline...")
    pipe = SanaVideoPipeline.from_pretrained(
        MODEL_ID,
        dtype=DTYPE_MODEL,
    )
    pipe.to(DEVICE)

    print("Encoding class prompt...")
    with torch.no_grad():
        prompt_embeds = pipe.encode_prompt(
            prompt=[CLASS_PROMPT],
            negative_prompt=None,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
            device=DEVICE,
        )[0]  # [1, L, D]

    PROMPT_EMBEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_embeds.cpu(), PROMPT_EMBEDS_PATH)
    print(f"Saved prompt embeddings to: {PROMPT_EMBEDS_PATH}")

    # Clean up heavy parts
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.transformer.to("cpu")
    del pipe
    torch.cuda.empty_cache()

    return prompt_embeds.to(DEVICE, dtype=DTYPE_MODEL)


def build_transformer_with_lora() -> torch.nn.Module:
    """
    Load SanaVideoTransformer3DModel and optionally wrap with LoRA.
    Optionally load existing LoRA weights from LORA_INIT_PATH.
    """
    print("Loading SanaVideoTransformer3DModel...")
    transformer = SanaVideoTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=DTYPE_MODEL,
    ).to(DEVICE)

    if USE_LORA:
        print("Applying LoRA to transformer...")
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=LORA_TARGET_MODULES,
        )
        transformer = get_peft_model(transformer, lora_cfg)
        transformer.print_trainable_parameters()

        # ---- Optional: load existing LoRA weights ----
        if LORA_INIT_PATH is not None and LORA_INIT_PATH.exists():
            print(f"Loading LoRA weights from: {LORA_INIT_PATH}")
            state = torch.load(LORA_INIT_PATH, map_location="cpu")
            # Use PEFT's helper instead of load_state_dict
            set_peft_model_state_dict(transformer, state)
            print("Loaded LoRA init weights into transformer.")
    else:
        print("Training full transformer (no LoRA).")

    transformer.train()
    return transformer


# ==========================
# LIGHTNING MODULE
# ==========================

class SanaTomJerryModule(pl.LightningModule):
    def __init__(self, lr: float, save_dir: str, save_lora_every: int, initial_step: int):
        super().__init__()
        # save_dir + save_lora_every + initial_step into hparams
        self.save_hyperparameters()

        self.transformer = build_transformer_with_lora()
        prompt_embeds = prepare_prompt_embeds()

        # Register as buffer so it's saved with checkpoints
        self.register_buffer(
            "prompt_embeds",
            prompt_embeds,  # [1, L, D]
            persistent=True,
        )

        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            print("Enabling gradient checkpointing...")
            self.transformer.enable_gradient_checkpointing()

    # ----------- core training logic -----------

    def flowmatching_step(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, C, F, H, W] on current device (x0)
        Rectified Flow / Flow-Matching:
          x1 ~ N(0,I), t~U(0,1),
          x_t = (1-t)x0 + t x1
          v_target = x1 - x0
          transformer predicts v_hat; loss = MSE(v_hat, v_target)
        """
        x0 = latents.to(self.device, dtype=DTYPE_MODEL)
        B = x0.shape[0]

        x1 = torch.randn_like(x0)
        t = torch.rand(B, 1, 1, 1, 1, device=self.device, dtype=DTYPE_MODEL)

        x_t = (1.0 - t) * x0 + t * x1
        v_target = x1 - x0

        encoder_hidden_states = self.prompt_embeds.expand(B, -1, -1)  # [B, L, D]
        timesteps = (t.view(B) * 1000.0).to(self.device, dtype=DTYPE_MODEL)

        out = self.transformer(
            hidden_states=x_t,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )
        v_pred = out.sample if hasattr(out, "sample") else out

        loss = F.mse_loss(v_pred, v_target)
        return loss

    def training_step(self, batch, batch_idx):
        latents = batch  # [B, C, F, H, W]
        loss = self.flowmatching_step(latents)

        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=latents.size(0),
        )

        # periodic LoRA saving with offset step
        save_every = self.hparams.save_lora_every
        offset = self.hparams.initial_step
        effective_step = offset + self.global_step + 1  # "real" step including previous run

        if save_every and save_every > 0:
            if effective_step % save_every == 0:
                ckpt_path = Path(self.hparams.save_dir) / f"lora_step_{effective_step:06d}.pt"
                self._save_lora(ckpt_path)
                print(f"[effective_step {effective_step}] Saved LoRA weights to: {ckpt_path}")

        return loss

    def configure_optimizers(self):
        # with LoRA, only adapter params require grad
        params = [p for p in self.transformer.parameters() if p.requires_grad]

        if HAS_BNB:
            optimizer = AdamW8bit(
                params,
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )
        else:
            optimizer = AdamW(
                params,
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )
        return optimizer

    # ----------- utility: save LoRA / transformer -----------

    def _save_lora(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if USE_LORA:
            state = get_peft_model_state_dict(self.transformer)
        else:
            state = self.transformer.state_dict()
        torch.save(state, path)


# ==========================
# MAIN
# ==========================

def main():
    # 1) Dataset + dataloader
    dataset = TomJerryLatentDataset(
        cache_dir=CACHE_DIR,
        dtype=DTYPE_MODEL,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: torch.stack(batch, dim=0),  # [B, C, F, H, W]
        persistent_workers=True,
    )

    # 2) Lightning module
    model = SanaTomJerryModule(
        lr=LR,
        save_dir=str(SAVE_DIR),
        save_lora_every=SAVE_LORA_EVERY,
        initial_step=INITIAL_STEP,
    )

    # 3) Logger (optional W&B)
    logger = None
    if DO_WANDB:
        logger = WandbLogger(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            log_model=False,
        )

    # 4) Trainer — controlled by max_steps
    trainer = pl.Trainer(
        max_steps=MAX_STEPS,
        max_epochs=-1,  # no effective epoch limit, use steps
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        default_root_dir=str(SAVE_DIR),
        gradient_clip_val=1.0,
        log_every_n_steps=2,  # W&B every 2 steps
        enable_checkpointing=False,
    )

    # 5) Train
    trainer.fit(model, train_dataloaders=train_loader)

    # Final LoRA save (always) with final effective step
    final_effective_step = INITIAL_STEP + trainer.global_step
    final_ckpt = SAVE_DIR / f"lora_step_{final_effective_step:06d}_final.pt"
    model._save_lora(final_ckpt)
    print(f"Saved final LoRA / transformer weights to: {final_ckpt}")


if __name__ == "__main__":
    main()