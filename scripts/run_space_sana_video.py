#!/usr/bin/env python3
"""
batch_generate_tomjerry_loras.py

Generate comparison videos for multiple LoRA checkpoints of SANA-Video.

- Loads SANA-Video_2B_480p_diffusers once.
- Wraps transformer with LoRA (same config as training).
- For each step in LORA_STEPS, loads lora_step_{step:06d}.pt
  from LORA_DIR and generates a video with fixed settings.
- Saves all MP4s into OUTPUT_DIR.
"""

from __future__ import annotations
from pathlib import Path
import time

import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video

from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict

from tqdm.auto import tqdm  # NEW: nicer progress bar

# ==========================
# CONFIG
# ==========================

MODEL_ID = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Folder with your LoRA checkpoints (same as training / Gradio)
LORA_DIR = Path("/home/ubuntu/AMIT/video/checkpoints_realtrain")

# Which steps to render
LORA_STEPS = [100, 1000, 2000, 5000, 7500, 10000]

# File pattern used during training: lora_step_{step:06d}.pt
LORA_FILENAME_TEMPLATE = "lora_step_{step:06d}.pt"

# Where to save the videos
OUTPUT_DIR = Path("tomjerry_lora_videos_lora_seed69420_cfg4")

# LoRA config (must match training)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["proj_out", "to_q", "to_v", "to_k", "linear_2", "linear_1", "linear"]

# Generation settings (same for all LoRAs)
HEIGHT = 224
WIDTH  = 224
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE      = 4.0
FPS   = 16
SEED  = 69420  # fixed seed to compare LoRAs fairly

PROMPT = (
    "A vintage slapstick 2D cartoon scene of a grey cat chasing a small brown mouse "
    "in a colorful house, Tom and Jerry style, bold outlines, limited color palette, "
    "exaggerated expressions, smooth character motion."
)

NEGATIVE_PROMPT = ""


# ==========================
# HELPERS
# ==========================

def load_lora_into_peft_model(peft_model: torch.nn.Module, lora_path: Path):
    """
    Load LoRA weights from `lora_path` into a PEFT-wrapped transformer.

    Training used torch.compile, so keys look like:
        "_orig_mod.base_model.model. ..."
    We strip the leading "_orig_mod." and optional "module." to match the inference model.
    """
    print(f"[INFO] Loading LoRA state dict from: {lora_path}")
    state = torch.load(lora_path, map_location="cpu")

    fixed_state = {}
    for k, v in state.items():
        new_k = k

        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        fixed_state[new_k] = v

    set_peft_model_state_dict(peft_model, fixed_state)

    # Debug: check LoRA norms on the PEFT model
    with torch.no_grad():
        lora_params = {n: p for n, p in peft_model.named_parameters() if "lora_" in n}
        total_abs = sum(p.abs().sum().item() for p in lora_params.values())
        max_abs = max((p.abs().max().item() for p in lora_params.values()), default=0.0)

    print(f"[DEBUG] After load, LoRA total |weights| = {total_abs:.4e}, max = {max_abs:.4e}")


def zero_lora_weights(peft_model: torch.nn.Module):
    """Set all LoRA weights to zero (for base comparison if needed)."""
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.zero_()
        lora_params = {n: p for n, p in peft_model.named_parameters() if "lora_" in n}
        total_abs = sum(p.abs().sum().item() for p in lora_params.values())
    print(f"[DEBUG] After zeroing, total |LoRA| = {total_abs:.4e}")


# ==========================
# MAIN
# ==========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load base pipeline once
    print("[INFO] Loading SANA-Video pipeline...")
    pipe = SanaVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    )

    # VAE in fp32 for stability
    pipe.vae.to(DEVICE, dtype=torch.float32)

    # Wrap transformer with LoRA
    print("[INFO] Wrapping transformer with LoRA adapter...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_cfg)

    # Start with LoRA zeroed (equivalent to base model)
    zero_lora_weights(pipe.transformer)

    pipe.to(DEVICE)
    pipe.transformer.to(DEVICE, dtype=DTYPE)
    pipe.text_encoder.to(DEVICE, dtype=DTYPE)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # Optional: also generate base model video for comparison
    print("[INFO] Generating base model video (LoRA zeroed)...")
    with torch.no_grad():
        base_out = pipe(
            prompt=[PROMPT],
            negative_prompt=[NEGATIVE_PROMPT],
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            height=HEIGHT,
            width=WIDTH,
            use_resolution_binning=False,
        )
    if hasattr(base_out, "frames"):
        base_frames = base_out.frames[0]
    else:
        base_frames = base_out.videos[0]

    base_path = OUTPUT_DIR / f"tomjerry_base_seed{SEED}.mp4"
    export_to_video(base_frames, str(base_path), fps=FPS)
    print(f"[INFO] Saved base model video to: {base_path}")

    # 2) Loop over LoRA steps with tqdm
    print("\n[INFO] Generating LoRA videos...")
    for step in tqdm(LORA_STEPS, desc="LoRA checkpoints", unit="ckpt"):
        ckpt_name = LORA_FILENAME_TEMPLATE.format(step=step)
        ckpt_path = LORA_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"[WARN] LoRA checkpoint not found for step {step}: {ckpt_path}")
            continue

        print("\n" + "=" * 60)
        print(f"[INFO] Generating video for LoRA step {step}  ({ckpt_name})")
        print("=" * 60)

        # Reload generator with same seed so noise is identical
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

        # Load LoRA weights into transformer
        load_lora_into_peft_model(pipe.transformer, ckpt_path)

        # Generate video
        with torch.no_grad():
            out = pipe(
                prompt=[PROMPT],
                negative_prompt=[NEGATIVE_PROMPT],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                height=HEIGHT,
                width=WIDTH,
                use_resolution_binning=False,
            )

        if hasattr(out, "frames"):
            video_frames = out.frames[0]
        else:
            video_frames = out.videos[0]

        out_path = OUTPUT_DIR / f"tomjerry_lora_step{step:06d}_seed{SEED}.mp4"
        export_to_video(video_frames, str(out_path), fps=FPS)
        print(f"[INFO] Saved LoRA step {step} video to: {out_path}")

    print("\n[INFO] Done. All videos are in:")
    print(f"       {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()