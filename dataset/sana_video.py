#!/usr/bin/env python3
"""
tomjerry_data.py

1) Precompute SANA-Video latents for Tom & Jerry MP4s (cache builder).
2) Provide a Dataset that reads cached latents for training.

- Uses one global prompt for the whole dataset (NOT stored per-item).
- Dataset __getitem__ returns ONLY the latents tensor.

Usage:
  - First, run as a script to build / extend the cache (resumable):
        python tomjerry_data.py

  - Then, in your training code:
        from tomjerry_data import TomJerryLatentDataset
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Set, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video
from tqdm import tqdm

from diffusers import SanaVideoPipeline


# ==========================
# CONFIG (EDIT THESE)
# ==========================

# Folder with your 224x224 Tom & Jerry mp4 episodes
RAW_VIDEO_DIR = Path("/home/ubuntu/AMIT/video/data/tomandjerry_224p")

# Where to store cached latents (.pt files)
CACHE_DIR = Path("/home/ubuntu/AMIT/video/data/full_video_latents")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SANA-Video model id
MODEL_ID = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"

# Clip + FPS configuration
FRAMES_PER_CLIP = 81   # number of frames per training clip
FPS_TARGET      = 16   # resample videos to this fps

# Device / dtype for caching
DEVICE        = "cuda"
VAE_DTYPE     = torch.float32    # safer for encode
LATENT_DTYPE  = torch.bfloat16   # store latents in bf16 to save disk

# Batch size for VAE encoding (tune based on VRAM)
CACHE_BATCH_SIZE = 4


# ==========================
# CACHE BUILDING
# ==========================

def _load_vae_only() -> torch.nn.Module:
    """
    Load SANA-Video pipeline and keep only the VAE on GPU.
    Transformer and text encoder are moved off GPU.
    """
    print("Loading SANA-Video pipeline (for VAE)...")
    pipe = SanaVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # put VAE on GPU in fp32 for better numeric stability
    pipe.vae.to(DEVICE, dtype=VAE_DTYPE)
    pipe.vae.eval()

    # free GPU from heavy parts we don't need for caching
    pipe.transformer.to("cpu")
    pipe.text_encoder.to("cpu")

    return pipe.vae


def _preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: [T, H, W, 3] uint8 from read_video
    returns: [1, C, F, H, W] in [-1, 1] for VAE

    We assume target size is 224x224, but resize anyway to be safe.
    """
    # [T, H, W, C] -> [T, C, H, W]
    frames = frames.permute(0, 3, 1, 2).contiguous()  # T, C, H, W

    resize = T.Resize((224, 224), antialias=True)
    frames = resize(frames)  # T, C, 224, 224

    # to [0,1] then [-1,1]
    frames = frames.float() / 255.0
    frames = frames * 2.0 - 1.0

    # [T, C, H, W] -> [1, C, F, H, W]
    frames = frames.unsqueeze(0)             # 1, T, C, H, W
    frames = frames.permute(0, 2, 1, 3, 4)   # 1, C, F, H, W
    return frames


def _scan_existing_cache() -> Tuple[int, Set[Tuple[str, int, int]]]:
    """
    Look at existing clip_*.pt files and:
      - Find max clip index (for naming new files)
      - Collect a set of (source_video, start_frame, end_frame) that already exist

    Returns:
        next_clip_index: int
        existing_keys: set of (video_name, start_frame, end_frame)
    """
    existing_files = sorted(CACHE_DIR.glob("clip_*.pt"))
    if not existing_files:
        print("No existing cached clips found – starting fresh.")
        return 0, set()

    print(f"Found {len(existing_files)} existing cached clips in {CACHE_DIR}")

    existing_keys: Set[Tuple[str, int, int]] = set()
    max_index = -1

    for f in tqdm(existing_files, desc="Scanning existing cache"):
        # parse index from filename: clip_000123.pt
        stem = f.stem  # "clip_000123"
        try:
            idx = int(stem.split("_")[-1])
            if idx > max_index:
                max_index = idx
        except Exception:
            # ignore weird filenames
            pass

        # load metadata to know which segments are already done
        try:
            data = torch.load(f, map_location="cpu")
            src = str(data.get("source_video", ""))
            start = int(data.get("start_frame", -1))
            end = int(data.get("end_frame", -1))
            existing_keys.add((src, start, end))
        except Exception as e:
            print(f"Warning: failed to read metadata from {f}: {e}")

    next_index = max_index + 1
    print(f"Will start new clips from index: {next_index:06d}")
    return next_index, existing_keys


def build_latent_cache():
    """
    Walk RAW_VIDEO_DIR, slice videos into clips of FRAMES_PER_CLIP,
    encode with VAE in batches of CACHE_BATCH_SIZE,
    and save latents to CACHE_DIR as clip_XXXXXX.pt.

    This process is **resumable**:
      - Existing clips (same video + start_frame + end_frame) are skipped.
      - New clips get new indices after the last existing one.

    Each .pt file has:
        - "latents": [C, F, H', W'] (bfloat16, CPU)
        - "source_video": str
        - "start_frame": int
        - "end_frame": int
    """
    # Scan existing cache so we can resume
    clip_index, existing_keys = _scan_existing_cache()

    vae = _load_vae_only()

    video_paths: List[Path] = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if not video_paths:
        raise RuntimeError(f"No .mp4 files found in {RAW_VIDEO_DIR}")

    print(f"Found {len(video_paths)} videos in {RAW_VIDEO_DIR}")
    new_clips = 0

    # global buffers so we can batch across videos too
    batch_videos: List[torch.Tensor] = []
    batch_meta: List[tuple] = []

    def flush_batch():
        nonlocal clip_index, batch_videos, batch_meta, new_clips
        if not batch_videos:
            return

        # [B, C, F, H, W]
        batch_tensor = torch.cat(batch_videos, dim=0).to(DEVICE, VAE_DTYPE)

        with torch.no_grad():
            posterior = vae.encode(batch_tensor)
            # AutoencoderKLWan has no `scaling_factor` in config; we keep raw latents.
            latents = posterior.latent_dist.sample()  # [B, C, F', H', W']

        latents = latents.to(LATENT_DTYPE).cpu()

        for b_idx, (source_video, start, end) in enumerate(batch_meta):
            clip_latents = latents[b_idx]  # [C, F', H', W']
            out_path = CACHE_DIR / f"clip_{clip_index:06d}.pt"
            torch.save(
                {
                    "latents": clip_latents,
                    "source_video": source_video,
                    "start_frame": int(start),
                    "end_frame": int(end),
                },
                out_path,
            )
            clip_index += 1
            new_clips += 1

        batch_videos = []
        batch_meta = []

    for vid_path in tqdm(video_paths, desc="Videos"):
        # read_video: returns (video, audio, info)
        vframes, _, info = read_video(str(vid_path), pts_unit="sec")
        fps = float(info.get("video_fps", FPS_TARGET))

        # Temporal downsample to FPS_TARGET
        step = max(1, int(round(fps / FPS_TARGET)))
        vframes = vframes[::step]  # [T', H, W, C]

        total_frames = vframes.shape[0]
        if total_frames < FRAMES_PER_CLIP:
            print(f"  Skipping {vid_path.name}: not enough frames ({total_frames}).")
            continue

        # Non-overlapping clips of FRAMES_PER_CLIP
        clip_starts = range(0, total_frames - FRAMES_PER_CLIP + 1, FRAMES_PER_CLIP)

        for start in tqdm(clip_starts, desc=f"{vid_path.name} clips", leave=False):
            end = start + FRAMES_PER_CLIP
            key = (vid_path.name, int(start), int(end))

            # If we already have this segment cached, skip
            if key in existing_keys:
                continue

            clip_frames = vframes[start:end]  # [FRAMES_PER_CLIP, H, W, C]

            video_tensor = _preprocess_frames(clip_frames)  # [1, C, F, H, W]
            batch_videos.append(video_tensor)               # keep batch on CPU for now
            batch_meta.append(key)

            # If batch is full, move to GPU and encode
            if len(batch_videos) >= CACHE_BATCH_SIZE:
                flush_batch()

    # leftover clips
    flush_batch()

    print(f"Done. New cached clips this run: {new_clips}")
    print(f"Total clips (existing + new): {len(existing_keys) + new_clips}")


# ==========================
# DATASET
# ==========================

class TomJerryLatentDataset(Dataset):
    """
    Dataset over cached Tom & Jerry latents.

    Each cache file:
        {
          "latents": [C, F, H', W']  (bfloat16 or float32)
          ... (metadata ignored by dataset)
        }

    __getitem__ returns ONLY the latents tensor:
        latents: [C, F, H', W']
    """

    def __init__(self, cache_dir: Path, dtype: torch.dtype = torch.bfloat16):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("clip_*.pt"))
        if not self.files:
            raise RuntimeError(f"No cached clips found in {self.cache_dir}")
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = torch.load(self.files[idx], map_location="cpu")
        latents = data["latents"]  # [C, F, H', W']
        return latents.to(self.dtype)


# ==========================
# CLI ENTRY
# ==========================

if __name__ == "__main__":
    build_latent_cache()