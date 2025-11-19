#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Tuple

import tempfile  # NEW
import numpy as np  # NEW
import cv2  # NEW

import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr
from diffusers import (
    AutoencoderDC,
    SanaTransformer2DModel,
    SanaPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoProcessor, AutoModel

# Your Lightning module class
from models.zoom_siglip  import SanaZoomLoRA
from peft import LoraConfig, get_peft_model

# --- LoRA config: MUST MATCH TRAINING ---
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0


# ============================================================
# CONFIG
# ============================================================

# --- Base SANA checkpoint (same as training) ---
PRETRAINED_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"

# --- Paths to your fine-tuned weights (EDIT THESE) ---
step = 30000
CKPT_ROOT      = Path(f"/home/ubuntu/AMIT/runs_sana/zoom_siglip_zoom_big_run/weights/step_0030000")  # ### EDIT ME ###
ZOOM_PROJ_PATH = CKPT_ROOT / "zoom_proj.pt"
SIGLIP_PROJ_PATH = CKPT_ROOT / "siglip_proj.pt"

LORA_TARGET_MODULES = ['linear', 'linear_1', 'linear_2', 'to_q', 'to_k', 'to_v', 'proj_out']

# LoRA weights file saved by on_fit_end()
LORA_WEIGHTS_SAFE = CKPT_ROOT / "lora" / "pytorch_lora_weights.safetensors"

# Where to cache text embeddings (pe/pam)
TEXT_CACHE_PATH = CKPT_ROOT / "text_cache.pt"

# Same instance prompt you used during training
INSTANCE_PROMPT = "a 3d game asset rendered in a white background"  # ### keep in sync with training ###

# SigLIP
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-256"
SIGLIP_DIM = 768  # matches training

# Sampling
NUM_SAMPLE_STEPS = 30
OUTPUT_RESOLUTION = 1024  # you trained at 1024x1024

# Device / dtype
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
USE_BF16 = False
WEIGHT_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

print(f"[device] {DEVICE} | weight dtype = {WEIGHT_DTYPE}")


# ============================================================
# GLOBALS – filled once at startup
# ============================================================

PE_CACHE: torch.Tensor | None = None   # [1, T, D]
PAM_CACHE: torch.Tensor | None = None  # [1, T]
SIGLIP_PROCESSOR: AutoProcessor | None = None
SIGLIP_MODEL: AutoModel | None = None
MODEL: SanaZoomLoRA | None = None


# ============================================================
# 1. TEXT ENCODING CACHE (pe / pam)
# ============================================================

@torch.no_grad()
def build_text_cache() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (or load) the text encoder outputs for the instance prompt:
      - pe:  [1, T, D]  (last hidden state of Gemma text encoder)
      - pam: [1, T]     (attention mask, bool)
    """
    TEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TEXT_CACHE_PATH.exists():
        cache = torch.load(TEXT_CACHE_PATH, map_location="cpu")
        pe = cache["pe"]          # [1, T, D]
        pam = cache["pam"].bool() # [1, T]
        print(f"[text-cache] loaded from {TEXT_CACHE_PATH}")
        return pe, pam

    print("[text-cache] building text cache for instance prompt...")
    # Load only tokenizer + text_encoder via SanaPipeline
    text_pipe = SanaPipeline.from_pretrained(
        PRETRAINED_MODEL,
        vae=None,
        transformer=None,
        torch_dtype=torch.bfloat16,  # Gemma likes bf16
    )

    tokenizer = text_pipe.tokenizer
    text_encoder = text_pipe.text_encoder.to(DEVICE, dtype=torch.bfloat16).eval()

    tokens = tokenizer(
        INSTANCE_PROMPT,
        max_length=300,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens.input_ids.to(DEVICE)
    attention_mask = tokens.attention_mask.to(DEVICE)

    out = text_encoder(input_ids=input_ids)
    hidden_states = out.last_hidden_state  # [1, T, D]

    pe = hidden_states.to(torch.float32).cpu()  # [1, T, D]
    pam = attention_mask.bool().cpu()          # [1, T]

    torch.save({"pe": pe, "pam": pam}, TEXT_CACHE_PATH)
    print(f"[text-cache] saved to {TEXT_CACHE_PATH}")

    # Free memory
    del text_pipe, text_encoder, tokenizer, tokens, out
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return pe, pam


# ============================================================
# 2. LOAD SIGLIP ENCODER
# ============================================================

def setup_siglip():
    global SIGLIP_PROCESSOR, SIGLIP_MODEL

    if SIGLIP_PROCESSOR is not None and SIGLIP_MODEL is not None:
        return

    print("[siglip] loading SigLIP model...")
    SIGLIP_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    SIGLIP_MODEL = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(DEVICE)
    SIGLIP_MODEL.eval()


@torch.no_grad()
def get_siglip_embedding_from_pil(img: Image.Image) -> torch.Tensor:
    """
    img: PIL.Image
    returns [1, D_siglip] L2-normalized tensor on DEVICE
    """
    setup_siglip()
    assert SIGLIP_PROCESSOR is not None
    assert SIGLIP_MODEL is not None

    img = img.convert("RGB")
    inputs = SIGLIP_PROCESSOR(images=img, return_tensors="pt").to(DEVICE)
    feats = SIGLIP_MODEL.get_image_features(**inputs)  # [1, D]
    feats = F.normalize(feats, p=2, dim=-1)            # [1, D]
    return feats  # [1, D]


# ============================================================
# 3. LOAD SANA + LORA + ZOOM/SIGLIP HEADS into SanaZoomLoRA
# ============================================================

def setup_model():
    """
    Build the SanaZoomLoRA LightningModule in pure inference mode.
    """
    global MODEL, PE_CACHE, PAM_CACHE

    if MODEL is not None and PE_CACHE is not None and PAM_CACHE is not None:
        return

    # ---- text cache (pe / pam) ----
    PE_CACHE, PAM_CACHE = build_text_cache()  # CPU

    # ---- load base VAE + transformer ----
    print("[model] loading base SANA1.5 VAE + transformer...")
    vae = AutoencoderDC.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="vae",
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="transformer",
    )

    # ---- attach LoRA via PEFT ----
    print("[model] attaching LoRA via PEFT...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    transformer = get_peft_model(transformer, lora_config)

    # ---- load LoRA weights ----
    if LORA_WEIGHTS_SAFE.exists():
        from safetensors.torch import load_file
        print(f"[model] loading LoRA weights from {LORA_WEIGHTS_SAFE}")
        raw_state = load_file(str(LORA_WEIGHTS_SAFE))
    else:
        raise FileNotFoundError(f"LoRA weights not found at {LORA_WEIGHTS_SAFE}")

    adapted_state = {}
    for k, v in raw_state.items():
        new_k = k
        if new_k.startswith("transformer."):
            new_k = new_k[len("transformer."):]
        new_k = new_k.replace(".lora_A.weight", ".lora_A.default.weight")
        new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")
        adapted_state[new_k] = v

    missing, unexpected = transformer.load_state_dict(adapted_state, strict=False)
    print(f"[model] LoRA load: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (first 10):", missing[:10])
    if unexpected:
        print("  unexpected keys (first 10):", unexpected[:10])

    # ---- move to device / dtype and freeze ----
    vae = vae.to(DEVICE, dtype=torch.float32).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    transformer = transformer.to(DEVICE, dtype=WEIGHT_DTYPE).eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    # ---- scheduler ----
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="scheduler",
    )

    # ---- build SanaZoomLoRA ----
    print("[model] building SanaZoomLoRA (inference mode)...")
    model = SanaZoomLoRA(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        lr=1e-4,                # unused at inference
        do_lora=False,          # transformer already has LoRA, we just run it
        example_dir=CKPT_ROOT / "samples_space",
        save_examples_every_steps=0,
        save_pt_every_steps=0,
        lora_freeze_steps=0,
        siglip_dim=SIGLIP_DIM,
        instance_prompt=INSTANCE_PROMPT,
        num_sample_steps=NUM_SAMPLE_STEPS,
        weighting_scheme="none",
        ckpt_root=CKPT_ROOT,
    )

    # ---- load zoom_proj / siglip_proj ----
    if ZOOM_PROJ_PATH.exists():
        print(f"[model] loading zoom_proj from {ZOOM_PROJ_PATH}")
        zoom_state = torch.load(ZOOM_PROJ_PATH, map_location=DEVICE)
        model.zoom_proj.load_state_dict(zoom_state)
    else:
        print(f"[WARN] {ZOOM_PROJ_PATH} not found – zoom token head will be random")

    if SIGLIP_PROJ_PATH.exists() and model.siglip_proj is not None:
        print(f"[model] loading siglip_proj from {SIGLIP_PROJ_PATH}")
        sig_state = torch.load(SIGLIP_PROJ_PATH, map_location=DEVICE)
        model.siglip_proj.load_state_dict(sig_state)
    else:
        print(f"[WARN] {SIGLIP_PROJ_PATH} not found or siglip_proj=None – SigLIP head will be random")

    # --- move the Lightning module to device (preserve dtypes of submodules) ---
    model = model.to(DEVICE)

    # --- IMPORTANT: if we run in bf16, also cast zoom/siglip heads to bf16 ---
    if USE_BF16:
        # transformer is already bf16 (WEIGHT_DTYPE)
        if hasattr(model, "zoom_proj") and model.zoom_proj is not None:
            model.zoom_proj = model.zoom_proj.to(dtype=WEIGHT_DTYPE)
        if getattr(model, "siglip_proj", None) is not None:
            model.siglip_proj = model.siglip_proj.to(dtype=WEIGHT_DTYPE)

    model.eval()
    MODEL = model
    print("[model] setup complete.")


# ============================================================
# 4. INFERENCE: IMAGE + ZOOM -> GENERATED IMAGE
# ============================================================

@torch.no_grad()
def generate_zoomed_image(
    image: Image.Image,
    zoom_level: float,
    latents_init: torch.Tensor | None = None,  # NEW
) -> Image.Image:
    """
    Main inference function.

    If latents_init is provided, we use the same initial noise for all calls
    (e.g. for video frames). Otherwise we sample fresh noise.
    """
    setup_model()
    assert MODEL is not None
    assert PE_CACHE is not None and PAM_CACHE is not None

    model = MODEL
    dtype = model.transformer.dtype

    # --- text conditioning (clone cached) ---
    pe = PE_CACHE.clone().to(DEVICE, dtype=dtype)   # [1, T, D]
    pam = PAM_CACHE.clone().to(DEVICE)              # [1, T]

    z_norm = float(zoom_level)

    zo = torch.zeros((1, 1), device=DEVICE, dtype=torch.float32)      # original zoom (0)
    zt = torch.tensor([[z_norm]], device=DEVICE, dtype=torch.float32) # target zoom

    # Append zoom token
    pe, pam = model._append_zoom_token(pe, pam, zo, zt)

    # --- SigLIP token from the input image ---
    siglip_emb = get_siglip_embedding_from_pil(image)   # [1, D_siglip], float32
    # match dtype of pe / transformer (especially important in bf16 mode)
    if siglip_emb.dtype != pe.dtype:
        siglip_emb = siglip_emb.to(pe.dtype)

    batch = {"siglip_org_embed": siglip_emb}
    pe, pam = model._apply_siglip_token(pe, pam, batch)
    # --- Scheduler setup ---
    infer_scheduler = deepcopy(model.infer_scheduler)
    infer_scheduler.set_timesteps(NUM_SAMPLE_STEPS, device=DEVICE)
    timesteps = infer_scheduler.timesteps  # [num_steps]

    # --- Latent init (Sana 1024px: 32x32x32) ---
    if latents_init is None:
        latents = torch.randn(
            (1, 32, 32, 32),
            device=DEVICE,
            dtype=dtype,
        )
    else:
        # reuse the same base noise for all frames
        latents = latents_init.clone().to(device=DEVICE, dtype=dtype)

    # --- Flow-matching sampling loop ---
    for t in timesteps:
        t_batch = t.repeat(1).to(device=DEVICE, dtype=dtype)  # [1]

        model_pred = model.transformer(
            hidden_states=latents,
            encoder_hidden_states=pe,
            encoder_attention_mask=pam,
            timestep=t_batch,
            return_dict=False,
        )[0]

        latents = infer_scheduler.step(
            model_output=model_pred,
            timestep=t,
            sample=latents,
            return_dict=False,
        )[0]

    # Decode to image
    img_tensor = model._decode_latents(latents)[0]  # [H,W,3] uint8 on CPU
    img_pil = Image.fromarray(img_tensor.numpy())
    return img_pil


# ============================================================
# 4b. VIDEO GENERATION: IMAGE + START/END ZOOM -> MP4
# ============================================================

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation helper."""
    return a + (b - a) * t


def generate_zoom_video(
    image: Image.Image,
    start_zoom: float,
    end_zoom: float,
    num_frames: int,
    fps: int,
    title: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """
    Generate an MP4 video by interpolating zoom between start_zoom and end_zoom.

    All frames share the same initial noise (latents_init), so the video is
    temporally coherent. The seed/noise is constant within a single call.
    """
    if image is None:
        raise ValueError("Please upload an image.")

    num_frames = max(1, int(num_frames))
    fps = max(1, int(fps))

    # Prepare temporary file for video
    tmpdir = Path(tempfile.mkdtemp())
    video_path = tmpdir / "zoom_video.mp4"

    # Precompute zoom values
    if num_frames == 1:
        zooms = [start_zoom]
    else:
        zooms = [
            lerp(start_zoom, end_zoom, i / (num_frames - 1))
            for i in range(num_frames)
        ]

    progress(0, desc="Initializing video generation...")

    # --- NEW: sample one base latent tensor and reuse it for all frames ---
    # This gives you "same seed" for the whole video.
    base_latents = torch.randn(
        (1, 32, 32, 32),
        device=DEVICE,
        dtype=torch.float32,
    )

    writer = None
    try:
        for idx, z in enumerate(zooms):
            # Update progress bar
            progress(
                (idx + 1) / num_frames,
                desc=f"Rendering frame {idx + 1}/{num_frames} (zoom={z:.2f})",
            )

            # Generate frame at this zoom, using SAME initial noise
            frame_pil = generate_zoomed_image(
                image,
                z,
                latents_init=base_latents,  # NEW: shared noise
            )

            # Convert PIL -> OpenCV BGR
            frame_np = np.array(frame_pil)  # RGB, uint8
            h, w, _ = frame_np.shape

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # --- Overlays: title top-center, zoom counter top-left ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_title = 1.0
            font_scale_zoom = 0.7
            thickness_title = 2
            thickness_zoom = 2

            # Title (top center)
            if title:
                text_size, _ = cv2.getTextSize(title, font, font_scale_title, thickness_title)
                text_w, text_h = text_size
                x_title = (w - text_w) // 2
                y_title = 40 + text_h
                cv2.putText(
                    frame_bgr, title, (x_title + 2, y_title + 2),
                    font, font_scale_title, (0, 0, 0), thickness_title + 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame_bgr, title, (x_title, y_title),
                    font, font_scale_title, (255, 255, 255), thickness_title, cv2.LINE_AA
                )

            # Zoom counter (top-left)
            zoom_text = f"Zoom: {z:.2f}"
            zx, zy = 20, 40
            cv2.putText(
                frame_bgr, zoom_text, (zx + 2, zy + 2),
                font, font_scale_zoom, (0, 0, 0), thickness_zoom + 1, cv2.LINE_AA
            )
            cv2.putText(
                frame_bgr, zoom_text, (zx, zy),
                font, font_scale_zoom, (0, 255, 255), thickness_zoom, cv2.LINE_AA
            )

            writer.write(frame_bgr)

        progress(1, desc="Video generation complete.")
    finally:
        if writer is not None:
            writer.release()

    return str(video_path)


# ============================================================
# 5. GRADIO SPACE
# ============================================================

def gradio_infer(image: Image.Image, zoom: float) -> Image.Image:
    if image is None:
        raise ValueError("Please upload an image.")
    return generate_zoomed_image(image, zoom)


def gradio_infer_video(
    image: Image.Image,
    start_zoom: float,
    end_zoom: float,
    num_frames: int,
    fps: int,
    title: str,
    progress: gr.Progress = gr.Progress(),  # NEW: progress bar in UI
) -> str:
    return generate_zoom_video(
        image=image,
        start_zoom=start_zoom,
        end_zoom=end_zoom,
        num_frames=num_frames,
        fps=fps,
        title=title,
        progress=progress,
    )


title = "Sana 1.5 Zoom LoRA + SigLIP"
description = (
    "Upload an image and choose a zoom level or a zoom trajectory.\n\n"
    "Image tab: single zoomed frame.\n"
    "Video tab: generate a zoom video with start/end zoom, configurable number of frames and FPS. "
    "The video has a title at the top and a live zoom counter at the top-left."
)

# --- Single image interface ---
image_demo = gr.Interface(
    fn=gradio_infer,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="Zoom level (−2 to 2)"),
    ],
    outputs=gr.Image(type="pil", label="Generated image"),
    title=title,
    description=description,
)

# --- Video interface (NEW) ---
video_demo = gr.Interface(
    fn=gradio_infer_video,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="Start zoom (−2 to 2)"),
        gr.Slider(-2.0, 2.0, value=1.5, step=0.1, label="End zoom (−2 to 2)"),
        gr.Slider(10, 240, value=60, step=5, label="Number of frames"),
        gr.Slider(1, 60, value=24, step=1, label="FPS"),
        gr.Textbox(value="Sana Zoom LoRA", label="Video title (top overlay)"),
    ],
    outputs=gr.Video(label="Zoom video"),
    title=f"{title} — Video Mode",
    description="Generate a zoom video with overlays and a progress bar while rendering.",
)

# Combine into tabs
demo = gr.TabbedInterface(
    [image_demo, video_demo],
    ["Image (single frame)", "Video (zoom animation)"],
)

if __name__ == "__main__":
    demo.launch(share=True)