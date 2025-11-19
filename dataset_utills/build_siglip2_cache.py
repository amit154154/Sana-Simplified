#!/usr/bin/env python3
"""
Build SigLIP / SigLIP2 image embeddings for every zoom image in a dataset.

Assumes folder structure:

    root/
      obj_name_1/
        zoom_0.png
        zoom_1.png
        ...
      obj_name_2/
        zoom_0.png
        zoom_1.png
        ...

Embeddings are saved as:

    SIGLIP_CACHE_DIR / "{gi:07d}.pt"  with key "embedding"

The enumeration order (gi) matches ZoomLatentTextPairDataset:
  - sorted object dirs
  - sorted files per dir
"""

from pathlib import Path
from typing import Sequence, Dict, Any, List
import re

import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# ---------------- EDIT THESE ----------------
DATA_ROOT = Path(
    "/Users/mac/PycharmProjects/rotation_image_generation/"
    "data_zoom/dataset_zoom_10"
)
SIGLIP_CACHE_DIR = Path(
    "/Users/mac/PycharmProjects/rotation_image_generation/"
    "data_zoom/dataset_zoom_10_siglip"
)

SIGLIP_MODEL_NAME = "google/siglip-base-patch16-256"

ZOOM_PATTERN = r"zoom_(\d+)"   # must match dataset zoom regex
EXTS: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp")
# -------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[device] {DEVICE} | torch={torch.__version__}")


def main():
    SIGLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zoom_re = re.compile(ZOOM_PATTERN)
    exts_set = {e.lower() for e in EXTS}

    # 1) Enumerate all zoom images exactly like ZoomLatentTextPairDataset.items_flat
    items_flat: List[Dict[str, Any]] = []
    gi_counter = 0

    obj_dirs = [d for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()]

    for obj_dir in obj_dirs:
        obj_name = obj_dir.name
        for p in sorted(obj_dir.iterdir()):
            if p.suffix.lower() not in exts_set:
                continue
            m = zoom_re.search(p.stem)
            if not m:
                continue
            z = float(m.group(1))
            items_flat.append(
                {
                    "gi": gi_counter,
                    "path": p,
                    "zoom": z,
                    "obj_name": obj_name,
                }
            )
            gi_counter += 1

    if not items_flat:
        raise ValueError(f"No zoom images found under {DATA_ROOT}")

    print(f"[siglip-cache] found {len(items_flat)} images to consider.")

    def _embed_path(gi: int) -> Path:
        return SIGLIP_CACHE_DIR / f"{gi:07d}.pt"

    # 2) Filter missing (resumable)
    missing = [it for it in items_flat if not _embed_path(it["gi"]).exists()]
    if not missing:
        print(
            f"[siglip-cache] all {len(items_flat)} embeddings already exist in "
            f"{SIGLIP_CACHE_DIR}"
        )
        return

    print(f"[siglip-cache] building {len(missing)} embeddings -> {SIGLIP_CACHE_DIR}")

    # 3) Load model & processor (official HF pattern)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(DEVICE)
    model.eval()

    with torch.no_grad():
        for item in tqdm(missing):
            gi = item["gi"]
            path: Path = item["path"]

            out_path = _embed_path(gi)
            # Resumable within the same run as well
            if out_path.exists():
                continue

            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)

            # For SigLIP/SigLIP2, recommended way to get image embeddings
            # is model.get_image_features(**inputs) (see HF docs).
            image_features = model.get_image_features(**inputs)  # [1, D]
            emb = image_features[0]  # [D]

            # L2 normalize to match typical usage
            emb = F.normalize(emb, p=2, dim=0)

            payload = {"embedding": emb.cpu().half()}  # float16 to save disk
            torch.save(payload, out_path)

    print("[siglip-cache] done.")


if __name__ == "__main__":
    main()