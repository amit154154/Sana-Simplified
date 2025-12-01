#!/usr/bin/env python3
"""
Tom & Jerry SANA-Video LoRA – Gradio Space

- Loads SANA-Video_2B_480p_diffusers once.
- Wraps the transformer with a LoRA adapter (same config as training).
- Lets you pick which LoRA checkpoint (.pt) to load from a folder.
- You can switch LoRA at any time; only one LoRA is active at once.
- Includes a tqdm-like progress bar in the Gradio UI.
- V2 mode can pull a random cached scene from the latents+labels cache,
  fill the structured text fields, and show the original cached clip
  next to the generated one.
"""

from __future__ import annotations
from pathlib import Path
import os
import time
import random
import re

import torch
import gradio as gr
from diffusers import SanaVideoPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict

import pandas as pd  # NEW: for scene labels CSV

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_XFORMERS"] = "1"  # optional, avoids xformers issues

# ==========================
# MODE SWITCH
# ==========================
# If True → structured "scene description" V2 mode.
# If False → original single prompt + negative prompt.
V2 = True
#V2 = False

# ==========================
# CONFIG
# ==========================

MODEL_ID = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Directory with your LoRA checkpoints
LORA_DIR = Path("/home/ubuntu/AMIT/video/checkpoints_realtrain_v2_lowlr_warmup")
#LORA_DIR = Path("/home/ubuntu/AMIT/video/checkpoints_realtrain")

# Directory with cached latents + labels (from your builder script)
CACHE_DIR = Path("/home/ubuntu/AMIT/video/data/full_video_latents")
SCENE_LABELS_CSV = CACHE_DIR / "tomjerry_scene_labels_qwen3vl_vllm.csv"

# LoRA config must match training
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["proj_out", "to_q", "to_v", "to_k", "linear_2", "linear_1", "linear"]

# ---------- V1 default prompts ----------
DEFAULT_PROMPT = "classic tom and jerry"
DEFAULT_NEGATIVE_PROMPT = ""

# ---------- V2 base prefix + default section texts ----------
V2_BASE_PREFIX = "A classic Tom and Jerry cartoon scene."

V2_DEFAULT_ENVIRONMENT = (
    "- Inside a cozy living room with a wooden floor and a small carpet.\n"
    "- Simple painted walls with a framed picture and a doorway in the back.\n"
    "- Warm, even indoor lighting with a playful, cartoonish mood."
)

V2_DEFAULT_CHARACTERS = (
    "- Tom: a tall grey cat with white paws and a white muzzle, looking mischievous and confident.\n"
    "- Jerry: a small brown mouse with big ears, curious and slightly nervous."
)

V2_DEFAULT_PROPS = (
    "- A framed sign on the wall.\n"
    "- A doorway in the background.\n"
    "- A rug on the floor."
)

V2_DEFAULT_ACTION = (
    "- Jerry is peeking out cautiously while Tom leans in with a smug grin.\n"
    "- Tom appears to taunt or corner Jerry without actually grabbing him yet."
)

V2_DEFAULT_CAMERA = (
    "- Medium shot focusing on both Tom and Jerry.\n"
    "- Static camera angle at eye level, classic side view framing."
)

# Resolution you used for Tom & Jerry
HEIGHT = 224
WIDTH = 224

# Special label for “base model only”
BASE_OPTION = "Base SANA-Video (no LoRA)"


def list_lora_files() -> list[str]:
    """List available LoRA checkpoints in LORA_DIR."""
    if not LORA_DIR.exists():
        return []
    files = []
    for f in LORA_DIR.glob("*.pt"):
        if f.is_file():
            files.append(f.name)
    files.sort()
    return files


AVAILABLE_LORAS = list_lora_files()
LORA_OPTIONS = [BASE_OPTION] + AVAILABLE_LORAS
DEFAULT_LORA_SELECTION = AVAILABLE_LORAS[-1] if AVAILABLE_LORAS else BASE_OPTION

# ==========================
# GLOBAL PIPELINE & LABEL CACHE
# ==========================

pipe: SanaVideoPipeline | None = None
CURRENT_LORA_NAME: str | None = None
LORA_WRAPPED: bool = False  # whether transformer is already wrapped with LoRA adapters

SCENE_DF: pd.DataFrame | None = None  # lazy-loaded scene labels


# ==========================
# SCENE LABEL HELPERS
# ==========================

def load_scene_df() -> pd.DataFrame:
    """Lazy-load the scene labels CSV."""
    global SCENE_DF
    if SCENE_DF is None:
        if not SCENE_LABELS_CSV.exists():
            raise FileNotFoundError(f"Scene labels CSV not found: {SCENE_LABELS_CSV}")
        SCENE_DF = pd.read_csv(SCENE_LABELS_CSV)
        if "scene_description" not in SCENE_DF.columns:
            raise RuntimeError("Expected 'scene_description' column in labels CSV.")
    return SCENE_DF


def parse_scene_description(desc: str) -> dict:
    """
    Parse a single 'scene_description' string into sections:

    ENVIRONMENT / CHARACTERS / PROPS / ACTION / CAMERA & FRAMING

    Assumes the Qwen output format from the labeler script:
        ENVIRONMENT:
        ...
        CHARACTERS:
        ...
        PROPS:
        ...
        ACTION:
        ...
        CAMERA & FRAMING:
        ...
    """
    sections = {
        "ENVIRONMENT": "",
        "CHARACTERS": "",
        "PROPS": "",
        "ACTION": "",
        "CAMERA & FRAMING": "",
    }

    current = None
    lines = desc.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ENVIRONMENT:"):
            current = "ENVIRONMENT"
            continue
        elif stripped.startswith("CHARACTERS:"):
            current = "CHARACTERS"
            continue
        elif stripped.startswith("PROPS:"):
            current = "PROPS"
            continue
        elif stripped.startswith("ACTION:"):
            current = "ACTION"
            continue
        elif stripped.startswith("CAMERA & FRAMING:"):
            current = "CAMERA & FRAMING"
            continue

        if current is not None:
            if sections[current]:
                sections[current] += "\n" + line
            else:
                sections[current] = line

    for k in sections:
        sections[k] = sections[k].strip()

    return sections


def decode_cached_clip_from_pt(clip_pt_path: str, fps: int = 16) -> str:
    """
    Decode a cached clip from its latents .pt file into an MP4,
    using a fresh SANA-Video VAE (same logic as the working notebook script).
    """
    pt_path = Path(clip_pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Cached clip .pt not found: {pt_path}")

    # ---- 1) Load latents from .pt on CPU ----
    data = torch.load(pt_path, map_location="cpu")
    if "latents" not in data:
        raise KeyError(
            f"'latents' not found in {pt_path}. "
            "Adjust decode_cached_clip_from_pt() if your key name differs."
        )

    latents = data["latents"]  # e.g. [C, F, H, W] or [F, C, H, W]
    print(f"[decode_cached_clip_from_pt] latents shape: {latents.shape}, dtype: {latents.dtype}")

    # ---- 2) Load a minimal pipeline just for the VAE (exactly like notebook) ----
    vae_pipe = SanaVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    vae_pipe.vae.to(DEVICE, dtype=torch.float32)
    vae_pipe.vae.eval()

    # Move latents to device + VAE dtype
    latents = latents.to(DEVICE, dtype=vae_pipe.vae.dtype)

    # Ensure batch dimension
    if latents.dim() == 4:
        # treat as single batch
        latents = latents.unsqueeze(0)

    # ---- 3) Decode latents with VAE ----
    with torch.no_grad():
        decoded = vae_pipe.vae.decode(latents).sample  # [B, C, F, H, W] or [B, F, C, H, W]

    print(
        "[decode_cached_clip_from_pt] decoded shape:",
        decoded.shape,
        "dtype:", decoded.dtype,
        "min/max:", decoded.min().item(), decoded.max().item(),
    )

    if decoded.dim() != 5:
        raise RuntimeError(f"Unexpected decoded shape: {decoded.shape}")

    # ---- 4) Reorder to [F, H, W, C] exactly like the notebook ----
    if decoded.shape[1] in (3, 4):
        # [B, C, F, H, W] -> [B, F, H, W, C]
        video = decoded.permute(0, 2, 3, 4, 1)
    else:
        # [B, F, C, H, W] -> [B, F, H, W, C]
        video = decoded.permute(0, 1, 3, 4, 2)

    video = video[0]  # [F, H, W, C]

    # ---- 5) Normalize from [-1, 1] -> [0, 255] uint8 (same as your notebook logic) ----
    video = (video / 2 + 0.5).clamp(0, 1)           # [F, H, W, C] in [0,1]
    video_np = video.detach().cpu().numpy()

    # ---- 6) Save MP4 ----
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/original_cached_{pt_path.stem}.mp4"
    export_to_video(video_np, out_path, fps=int(fps))

    # Optional: free the tiny VAE pipeline to not keep extra VRAM
    vae_pipe.to("cpu")
    del vae_pipe
    torch.cuda.empty_cache()

    return out_path


# ==========================
# HELPER: LOAD LORA WITH PREFIX FIX
# ==========================

def load_lora_into_peft_model(peft_model: torch.nn.Module, lora_path: Path):
    """
    Load LoRA weights from `lora_path` into a PEFT-wrapped transformer.

    Training used torch.compile, so keys look like:
        "_orig_mod.base_model.model. ..."
    We strip the leading "_orig_mod." and optional "module." to match the inference model.
    """
    print(f"Loading LoRA state dict from: {lora_path}")
    state = torch.load(lora_path, map_location="cpu")

    fixed_state = {}
    for k, v in state.items():
        new_k = k

        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]

        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        fixed_state[new_k] = v

    # Actually load into PEFT model
    set_peft_model_state_dict(peft_model, fixed_state)

    # Debug: check LoRA norms on the PEFT model
    with torch.no_grad():
        lora_params = {n: p for n, p in peft_model.named_parameters() if "lora_" in n}
        total_abs = sum(p.abs().sum().item() for p in lora_params.values())
        max_abs = max((p.abs().max().item() for p in lora_params.values()), default=0.0)

    print(f"[DEBUG] After load, LoRA total |weights| = {total_abs:.4e}, max = {max_abs:.4e}")


def _wrap_transformer_with_lora(base_pipe: SanaVideoPipeline):
    """Wrap base_pipe.transformer with a PEFT LoRA adapter (only once)."""
    global LORA_WRAPPED

    if LORA_WRAPPED:
        return

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    base_pipe.transformer = get_peft_model(base_pipe.transformer, lora_cfg)
    base_pipe.transformer.to(DEVICE, dtype=DTYPE)

    # Initialize LoRA weights to zero ⇒ same as base model until you load a checkpoint
    with torch.no_grad():
        for name, param in base_pipe.transformer.named_parameters():
            if "lora_" in name:
                param.zero_()

    LORA_WRAPPED = True
    print("Transformer wrapped with LoRA adapter (weights initialized to zero).")

    # Debug after zeroing
    with torch.no_grad():
        lora_params = {n: p for n, p in base_pipe.transformer.named_parameters() if "lora_" in n}
        total_abs = sum(p.abs().sum().item() for p in lora_params.values())
    print(f"[DEBUG] After initial zeroing, total |LoRA| = {total_abs:.4e}")


def load_pipeline(selected_lora: str | None) -> SanaVideoPipeline:
    """
    Load base SANA-Video pipeline once, wrap with LoRA once,
    and then swap LoRA weights depending on `selected_lora`.

    selected_lora:
      - None            => zero LoRA weights (base behavior)
      - filename in dir => load that .pt into the LoRA adapter
    """
    global pipe, CURRENT_LORA_NAME

    # 1) Create pipeline once
    if pipe is None:
        print("Loading SANA-Video pipeline...")
        base_pipe = SanaVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        )

        # VAE in fp32 for stability
        base_pipe.vae.to(DEVICE, dtype=torch.float32)

        # Text encoder + transformer in DTYPE
        base_pipe.text_encoder.to(DEVICE, dtype=DTYPE)
        base_pipe.transformer.to(DEVICE, dtype=DTYPE)

        # Wrap transformer with LoRA structure (zero-initialized)
        _wrap_transformer_with_lora(base_pipe)

        base_pipe.to(DEVICE)
        base_pipe.set_progress_bar_config(disable=False)
        base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_pipe.scheduler.config, flow_shift=8.0
        )

        pipe = base_pipe
        CURRENT_LORA_NAME = None

    # 2) Handle "base model" option ⇒ zero out LoRA weights
    if selected_lora is None or selected_lora == BASE_OPTION:
        if CURRENT_LORA_NAME is not None:
            print("Switching to base model (no LoRA) by zeroing LoRA weights.")
        with torch.no_grad():
            for name, param in pipe.transformer.named_parameters():
                if "lora_" in name:
                    param.zero_()
            lora_params = {n: p for n, p in pipe.transformer.named_parameters() if "lora_" in n}
            total_abs = sum(p.abs().sum().item() for p in lora_params.values())
        print(f"[DEBUG] After zeroing, total |LoRA| = {total_abs:.4e}")

        CURRENT_LORA_NAME = None
        return pipe

    # 3) If the same LoRA is already active, just reuse it
    if CURRENT_LORA_NAME == selected_lora:
        return pipe

    # 4) Load new LoRA checkpoint and apply to adapter
    lora_path = LORA_DIR / selected_lora
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    print(f"Loading LoRA from: {lora_path}")
    load_lora_into_peft_model(pipe.transformer, lora_path)
    CURRENT_LORA_NAME = selected_lora
    print(f"LoRA '{selected_lora}' loaded & applied to transformer.")

    return pipe


# ==========================
# CORE INFERENCE FUNCTION
# ==========================

def _generate_video_core(
    prompt: str,
    negative_prompt: str,
    lora_choice: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_frames: int,
    fps: int,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Core generation logic. Assumes `prompt` and `negative_prompt` are already finalized.
    """
    progress(0, desc="Preparing pipeline & LoRA")

    # Load / switch LoRA
    selected_lora = None if lora_choice == BASE_OPTION else lora_choice
    pipe = load_pipeline(selected_lora)

    # Fallback to defaults only in V1-style case (when caller passes empty)
    if not prompt or prompt.strip() == "":
        prompt = DEFAULT_PROMPT

    if not negative_prompt or negative_prompt.strip() == "":
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Seed logic: seed <= 0 => random
    if seed is None or int(seed) <= 0:
        seed_value = torch.randint(0, 2**31 - 1, (1,)).item()
    else:
        seed_value = int(seed)

    generator = torch.Generator(device=DEVICE).manual_seed(seed_value)

    progress(0.1, desc="Sampling video")

    with torch.no_grad():
        out = pipe(
            prompt=[prompt],
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            height=HEIGHT,
            width=WIDTH,
            use_resolution_binning=False,
        )

    progress(0.8, desc="Saving MP4")

    # Depending on diffusers version, output can be .frames or .videos
    if hasattr(out, "frames"):
        video_frames = out.frames[0]  # [T, H, W, 3], float32 [0,1]
    else:
        video_frames = out.videos[0]  # fallback

    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/tomjerry_{int(time.time())}_{seed_value}.mp4"
    export_to_video(video_frames, filename, fps=int(fps))

    progress(1.0, desc="Done")
    return filename


# ==========================
# V1 & V2 WRAPPERS
# ==========================

def generate_video_v1(
    prompt: str,
    negative_prompt: str,
    lora_choice: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_frames: int,
    fps: int,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Original mode: free-form prompt + negative prompt.
    """
    return _generate_video_core(
        prompt=prompt,
        negative_prompt=negative_prompt,
        lora_choice=lora_choice,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        progress=progress,
    )


def build_v2_prompt(
    env_text: str,
    characters_text: str,
    props_text: str,
    action_text: str,
    camera_text: str,
) -> str:
    """
    Build the structured scene prompt string like your example.
    """
    env_text = env_text.strip()
    characters_text = characters_text.strip()
    props_text = props_text.strip()
    action_text = action_text.strip()
    camera_text = camera_text.strip()

    prompt = (
        f"{V2_BASE_PREFIX}\n"
        f"ENVIRONMENT:\n{env_text}\n\n"
        f"CHARACTERS:\n{characters_text}\n\n"
        f"PROPS:\n{props_text}\n\n"
        f"ACTION:\n{action_text}\n\n"
        f"CAMERA & FRAMING:\n{camera_text}"
    )
    return prompt


def generate_video_v2(
    env_text: str,
    characters_text: str,
    props_text: str,
    action_text: str,
    camera_text: str,
    lora_choice: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_frames: int,
    fps: int,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    V2 mode:
    - negative prompt is always "" (zero).
    - positive prompt is built from structured ENV/CHARACTERS/PROPS/ACTION/CAMERA sections.
    """
    final_prompt = build_v2_prompt(
        env_text=env_text,
        characters_text=characters_text,
        props_text=props_text,
        action_text=action_text,
        camera_text=camera_text,
    )
    final_negative = ""  # forced zero
    return _generate_video_core(
        prompt=final_prompt,
        negative_prompt=final_negative,
        lora_choice=lora_choice,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        progress=progress,
    )


# ==========================
# RANDOM SCENE LOADER (V2)
# ==========================

def load_random_cached_scene(fps: int = 16):
    """
    Pick a random scene from the labels CSV, parse its description into
    ENV/CHARACTERS/PROPS/ACTION/CAMERA, decode the cached clip from latents,
    and return everything to fill the UI.

    Returns:
      env_text, characters_text, props_text, action_text, camera_text,
      info_markdown, original_video_path
    """
    df = load_scene_df()
    if len(df) == 0:
        raise RuntimeError("Scene labels CSV is empty.")

    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx]

    desc = str(row["scene_description"])
    sections = parse_scene_description(desc)

    clip_id = str(row.get("clip_id", f"row_{idx}"))
    clip_pt_path = row.get("clip_pt_path", None)
    if clip_pt_path is None:
        # Fallback: construct from clip_id if path column missing
        clip_pt_path = str(CACHE_DIR / f"{clip_id}.pt")

    original_video_path = decode_cached_clip_from_pt(clip_pt_path, fps=fps)

    info_md = (
        f"**Random cached scene**: index **{idx}**, "
        f"clip_id: `{clip_id}`\n\n"
        f"Loaded from: `{clip_pt_path}`"
    )

    env = sections.get("ENVIRONMENT", V2_DEFAULT_ENVIRONMENT) or V2_DEFAULT_ENVIRONMENT
    characters = sections.get("CHARACTERS", V2_DEFAULT_CHARACTERS) or V2_DEFAULT_CHARACTERS
    props = sections.get("PROPS", V2_DEFAULT_PROPS) or V2_DEFAULT_PROPS
    action = sections.get("ACTION", V2_DEFAULT_ACTION) or V2_DEFAULT_ACTION
    camera = sections.get("CAMERA & FRAMING", V2_DEFAULT_CAMERA) or V2_DEFAULT_CAMERA

    return env, characters, props, action, camera, info_md, original_video_path


# ==========================
# GRADIO UI
# ==========================

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# Sana-Video Tom & Jerry – LoRA Playground

This space wraps **SANA-Video 2B** with your **Tom & Jerry LoRAs**.

- Default prompt = class prompt used for training (V1 mode)  
- In V2 mode, you provide a **structured scene description**:
  ENVIRONMENT / CHARACTERS / PROPS / ACTION / CAMERA & FRAMING.
- In V2, you can also **load a random cached scene** from your latents+labels cache,
  auto-fill the fields, and see the original clip next to the generated video.
            """
        )

        if not V2:
            # ==========================
            # V1 UI (original free-form prompt)
            # ==========================
            with gr.Row():
                with gr.Column(scale=1):
                    lora_dropdown = gr.Dropdown(
                        label="LoRA checkpoint",
                        choices=LORA_OPTIONS,
                        value=DEFAULT_LORA_SELECTION,
                    )

                    prompt_in = gr.Textbox(
                        label="Prompt",
                        value=DEFAULT_PROMPT,
                        lines=4,
                    )
                    negative_in = gr.Textbox(
                        label="Negative prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=4,
                    )

                    steps_in = gr.Slider(
                        label="Sampling steps",
                        minimum=5,
                        maximum=80,
                        value=50,
                        step=1,
                    )
                    cfg_in = gr.Slider(
                        label="CFG scale",
                        minimum=1.0,
                        maximum=15.0,
                        value=6.0,
                        step=0.5,
                    )
                    frames_in = gr.Slider(
                        label="Number of frames (model uses its internal default length)",
                        minimum=16,
                        maximum=96,
                        value=81,
                        step=1,
                    )
                    fps_in = gr.Slider(
                        label="FPS",
                        minimum=4,
                        maximum=30,
                        value=16,
                        step=1,
                    )
                    seed_in = gr.Number(
                        label="Seed (≤0 = random)",
                        value=0,
                        precision=0,
                    )

                    generate_btn = gr.Button("Generate video 🚀")

                with gr.Column(scale=1):
                    video_out = gr.Video(label="Generated video (MP4)")

                generate_btn.click(
                    fn=generate_video_v1,
                    inputs=[
                        prompt_in,
                        negative_in,
                        lora_dropdown,
                        steps_in,
                        cfg_in,
                        frames_in,
                        fps_in,
                        seed_in,
                    ],
                    outputs=video_out,
                )

        else:
            # ==========================
            # V2 UI (structured + random cached scene)
            # ==========================
            with gr.Row():
                with gr.Column(scale=1):
                    lora_dropdown = gr.Dropdown(
                        label="LoRA checkpoint",
                        choices=LORA_OPTIONS,
                        value=DEFAULT_LORA_SELECTION,
                    )

                    steps_in = gr.Slider(
                        label="Sampling steps",
                        minimum=5,
                        maximum=80,
                        value=50,
                        step=1,
                    )
                    cfg_in = gr.Slider(
                        label="CFG scale",
                        minimum=1.0,
                        maximum=15.0,
                        value=6.0,
                        step=0.5,
                    )
                    frames_in = gr.Slider(
                        label="Number of frames (model uses its internal default length)",
                        minimum=16,
                        maximum=96,
                        value=81,
                        step=1,
                    )
                    fps_in = gr.Slider(
                        label="FPS",
                        minimum=4,
                        maximum=30,
                        value=16,
                        step=1,
                    )
                    seed_in = gr.Number(
                        label="Seed (≤0 = random)",
                        value=0,
                        precision=0,
                    )

                    gr.Markdown("### Structured scene description (V2 mode)")

                    env_in = gr.Textbox(
                        label="ENVIRONMENT",
                        value=V2_DEFAULT_ENVIRONMENT,
                        lines=4,
                    )
                    characters_in = gr.Textbox(
                        label="CHARACTERS",
                        value=V2_DEFAULT_CHARACTERS,
                        lines=4,
                    )
                    props_in = gr.Textbox(
                        label="PROPS",
                        value=V2_DEFAULT_PROPS,
                        lines=4,
                    )
                    action_in = gr.Textbox(
                        label="ACTION",
                        value=V2_DEFAULT_ACTION,
                        lines=4,
                    )
                    camera_in = gr.Textbox(
                        label="CAMERA & FRAMING",
                        value=V2_DEFAULT_CAMERA,
                        lines=4,
                    )

                    random_btn = gr.Button("🎲 Load random cached scene from cache")
                    generate_btn = gr.Button("Generate video 🚀 (V2)")

                with gr.Column(scale=1):
                    random_info = gr.Markdown(
                        "Cached scene info will appear here when you load a random scene."
                    )
                    original_video_out = gr.Video(
                        label="Original cached clip (from latents cache)"
                    )
                    video_out = gr.Video(label="Generated video (MP4)")

                # Wire up random scene loader
                random_btn.click(
                    fn=lambda fps: load_random_cached_scene(fps=fps),
                    inputs=[fps_in],
                    outputs=[
                        env_in,
                        characters_in,
                        props_in,
                        action_in,
                        camera_in,
                        random_info,
                        original_video_out,
                    ],
                )

                # Wire up generation
                generate_btn.click(
                    fn=generate_video_v2,
                    inputs=[
                        env_in,
                        characters_in,
                        props_in,
                        action_in,
                        camera_in,
                        lora_dropdown,
                        steps_in,
                        cfg_in,
                        frames_in,
                        fps_in,
                        seed_in,
                    ],
                    outputs=video_out,
                )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)