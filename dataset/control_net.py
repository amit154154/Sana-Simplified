#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import gc
import zipfile

import numpy as np
import cv2  # pip install opencv-python

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderDC, SanaPipeline
from huggingface_hub import hf_hub_download

# ---------------- DEVICE ----------------
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"[device] {DEVICE} | torch={torch.__version__}")


# ============================================================
# helpers: download & unpack MJHQ-30K locally
# ============================================================

MJHQ_CATEGORIES = [
    "animals",
    "art",
    "fashion",
    "food",
    "indoor",
    "landscape",
    "logo",
    "people",
    "plants",
    "vehicles",
]


def ensure_mjhq_root(cache_dir: Path) -> tuple[Path, Path]:
    """
    Ensure `mjhq30k_imgs.zip` and `meta_data.json` are downloaded and extracted.

    Returns:
      imgs_root:  Path to folder with category subdirs (animals, art, ...)
      meta_path:  Path to meta_data.json
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- download zip ----
    zip_path = hf_hub_download(
        repo_id="playgroundai/MJHQ-30K",
        filename="mjhq30k_imgs.zip",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    zip_path = Path(zip_path)

    # ---- extract if needed ----
    imgs_root = cache_dir / "mjhq30k_imgs"
    if not imgs_root.exists():
        print(f"[mjhq] extracting {zip_path} -> {imgs_root} ...")
        imgs_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(imgs_root)
        print("[mjhq] extraction done.")
    else:
        print(f"[mjhq] found existing image root at: {imgs_root}")

    # ---- download meta_data.json ----
    meta_path = hf_hub_download(
        repo_id="playgroundai/MJHQ-30K",
        filename="meta_data.json",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    meta_path = Path(meta_path)
    print(f"[mjhq] meta_data.json at: {meta_path}")

    return imgs_root, meta_path


# ============================================================
# Dataset
# ============================================================

class MJHQLocalCannyLatentTextDataset(Dataset):
    """
    MJHQ-30K dataset built from the *original* zip + meta_data.json.

    We rely on the hash filename to match image ↔ prompt exactly.

    Folder structure after unzip:
        root/
          animals/
            <hash>.png
            ...
          art/
          ...
          vehicles/

    meta_data.json:
        "<hash>": {
            "category": "<category_name>",
            "prompt": "<midjourney prompt>"
        }

    Per item returns:
      - latents:      [C,H',W']   float16  (Sana latent)
      - pe:           [T,D]       float32  (per-item text encoding)
      - pam:          [T]         int64/bool
      - canny_image:  [1,H_img,W_img] float32 in [0,1]
      - text:         str         (MJHQ prompt)
    """

    def __init__(
        self,
        imgs_root: Path,
        meta_path: Path,
        cache_dir: Path,
        vae: AutoencoderDC,
        text_pipe: SanaPipeline,
        image_size: int = 256,
        canny_low: int = 100,
        canny_high: int = 200,
        max_sequence_length: int = 300,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.imgs_root = Path(imgs_root)
        self.cache_dir = Path(cache_dir)
        self.latent_cache_dir = self.cache_dir / "latents"
        self.canny_cache_dir = self.cache_dir / "canny"
        self.text_cache_dir = self.cache_dir / "text"
        self.latent_cache_dir.mkdir(parents=True, exist_ok=True)
        self.canny_cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)

        self.image_size = image_size
        self.canny_low = canny_low
        self.canny_high = canny_high

        # ---------------- load meta_data.json ----------------
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # ---------------- build item list ----------------
        self.label_names = list(MJHQ_CATEGORIES)
        items: List[Dict[str, Any]] = []

        for cat in MJHQ_CATEGORIES:
            cat_dir = self.imgs_root / cat
            if not cat_dir.exists():
                print(f"[warn] category folder missing: {cat_dir}")
                continue

            # collect all PNGs (you can add jpg/webp if needed)
            for img_path in sorted(cat_dir.glob("*.png")):
                key = img_path.stem  # hash string
                info = meta.get(key)
                if info is None:
                    # no metadata for this image; skip
                    continue

                prompt = str(info.get("prompt", "")).strip()
                category = str(info.get("category", "")).strip()

                # sanity: make sure category matches folder
                if category and category != cat:
                    # if mismatch, still allowed but warn once per category
                    print(f"[meta] category mismatch for {img_path}: {category} vs {cat}")
                items.append(
                    {
                        "path": img_path,
                        "category": cat,
                        "prompt": prompt if prompt else f"A highly aesthetic {cat} image.",
                    }
                )

        if max_samples is not None:
            items = items[: max_samples]

        self.items = items
        print(f"[MJHQLocalCannyLatentTextDataset] using {len(self.items)} items")

        # deterministic preprocessing
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.image_size),
            ]
        )

        # 1) build latent cache
        self._ensure_latent_cache(vae)

        # 2) build canny cache
        self._ensure_canny_cache()

        # 3) build per-item text encoding cache
        self._ensure_text_cache(text_pipe, max_sequence_length)

    # ---------------- path helpers ----------------

    def _latent_path(self, idx: int) -> Path:
        return self.latent_cache_dir / f"{idx:07d}.pt"

    def _canny_path(self, idx: int) -> Path:
        return self.canny_cache_dir / f"{idx:07d}.pt"

    def _text_path(self, idx: int) -> Path:
        return self.text_cache_dir / f"{idx:07d}.pt"

    # ---------------- latent cache ----------------

    @torch.no_grad()
    def _ensure_latent_cache(self, vae: AutoencoderDC):
        missing = [i for i in range(len(self.items)) if not self._latent_path(i).exists()]
        if not missing:
            print(f"[latent-cache] found all {len(self.items)} latents in {self.latent_cache_dir.name}")
            return

        print(f"[latent-cache] building {len(missing)} latents -> {self.latent_cache_dir}")
        vae = vae.to(DEVICE, dtype=torch.float32).eval()
        scaling_factor = float(vae.config.scaling_factor)

        for i in tqdm(missing, desc="[latent-cache]"):
            img_path = self.items[i]["path"]
            img = Image.open(img_path).convert("RGB")

            img = self.img_transform(img)
            x = transforms.ToTensor()(img)  # [3,H,W], 0..1
            x = transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])(x)  # [-1,1]
            x = x.unsqueeze(0).to(DEVICE, dtype=torch.float32)

            lat = vae.encode(x).latent.float() * scaling_factor  # [1,C,H',W']
            payload = {"latent": lat[0].to(torch.bfloat16 if torch.cuda.is_available() else torch.float16).cpu()}
            torch.save(payload, self._latent_path(i))

        print("[latent-cache] done.")

    # ---------------- canny cache ----------------

    def _ensure_canny_cache(self):
        missing = [i for i in range(len(self.items)) if not self._canny_path(i).exists()]
        if not missing:
            print(f"[canny-cache] found all {len(self.items)} canny maps in {self.canny_cache_dir.name}")
            return

        print(f"[canny-cache] building {len(missing)} canny maps -> {self.canny_cache_dir}")

        for i in tqdm(missing, desc="[canny-cache]"):
            img_path = self.items[i]["path"]
            img = Image.open(img_path).convert("RGB")
            img = self.img_transform(img)

            img_np = np.array(img)          # [H,W,3], uint8
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)

            canny = torch.from_numpy(edges).float() / 255.0  # [H,W]
            canny = canny.unsqueeze(0)  # [1,H,W]

            payload = {"canny": canny}
            torch.save(payload, self._canny_path(i))

        print("[canny-cache] done.")

    # ---------------- text encoding cache ----------------

    @torch.no_grad()
    def _ensure_text_cache(
        self,
        text_pipe: SanaPipeline,
        max_sequence_length: int,
    ):
        missing = [i for i in range(len(self.items)) if not self._text_path(i).exists()]
        if not missing:
            print(f"[text-cache] found all {len(self.items)} text encodings in {self.text_cache_dir.name}")
            return

        print(f"[text-cache] building {len(missing)} text encodings -> {self.text_cache_dir}")
        text_pipe = text_pipe.to(DEVICE)

        for i in tqdm(missing, desc="[text-cache]"):
            prompt = self.items[i]["prompt"]

            prompt_embeds, prompt_attention_mask, _, _ = text_pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                max_sequence_length=max_sequence_length,
            )

            pe = prompt_embeds[0].cpu()          # [T,D]
            pam = prompt_attention_mask[0].cpu() # [T]

            payload = {"pe": pe, "pam": pam, "text": prompt}
            torch.save(payload, self._text_path(i))

        # free heavy stuff
        try:
            del text_pipe.text_encoder, text_pipe.tokenizer
            del text_pipe.transformer, text_pipe.vae
        except Exception:
            pass
        gc.collect()
        print("[text-cache] done.")

    # ---------------- Dataset API ----------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lat_pack = torch.load(self._latent_path(idx), map_location="cpu")
        canny_pack = torch.load(self._canny_path(idx), map_location="cpu")
        text_pack = torch.load(self._text_path(idx), map_location="cpu")

        latent = lat_pack["latent"]          # [C,H',W']
        canny = canny_pack["canny"]          # [1,H_img,W_img]
        pe = text_pack["pe"]                 # [T,D]
        pam = text_pack["pam"]               # [T]
        text = text_pack["text"]             # str

        return {
            "latents": latent,
            "pe": pe,
            "pam": pam,
            "canny_image": canny,
            "text": text,
        }


# ============================================================
# main: quick smoke test on playgroundai/MJHQ-30K
# ============================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (good for Mac / CLI)
    import matplotlib.pyplot as plt

    # ---------------- CONFIG ----------------
    PRETRAINED_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    MAX_SAMPLES = 20          # how many items to actually cache / use

    OUTPUT_DIR = Path("mjhq_canny_latent_test")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = OUTPUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------- NEW: use zip + meta_data.json instead of HF Parquet --------
    # from the new file where you defined them:
    # from dataset.controlnet.mjhq_local import ensure_mjhq_root, MJHQLocalCannyLatentTextDataset
    imgs_root, meta_path = ensure_mjhq_root(CACHE_DIR)

    # ------------- load VAE & Sana pipeline -------------
    print("[sana] loading VAE and pipeline...")
    vae = AutoencoderDC.from_pretrained(PRETRAINED_MODEL, subfolder="vae").eval()
    text_pipe = SanaPipeline.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
    )

    # ------------- build dataset (this also builds caches) -------------
    ds = MJHQLocalCannyLatentTextDataset(
        imgs_root=imgs_root,
        meta_path=meta_path,
        cache_dir=CACHE_DIR,
        vae=vae,
        text_pipe=text_pipe,
        image_size=IMAGE_SIZE,
        canny_low=100,
        canny_high=200,
        max_sequence_length=300,
        max_samples=MAX_SAMPLES,
    )

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dl))

    print("\n[batch] keys:", list(batch.keys()))
    print("latents:", batch["latents"].shape)          # [B,C,H',W']
    print("pe:", batch["pe"].shape)                    # [B,T,D]
    print("pam:", batch["pam"].shape)                  # [B,T]
    print("canny_image:", batch["canny_image"].shape)  # [B,1,H_img,W_img]
    print("text example:", batch["text"][0])

    # ------------- decode latents for visual sanity-check -------------
    with torch.no_grad():
        vae = vae.to(DEVICE, dtype=torch.float32).eval()
        scaling_factor = float(vae.config.scaling_factor)

        lat = batch["latents"].to(DEVICE).float() / scaling_factor
        img_rec = vae.decode(lat, return_dict=False)[0]   # [-1,1]
        img_rec = (img_rec.clamp(-1, 1) + 1) / 2.0        # [0,1]
        img_rec = img_rec.cpu()

    B = img_rec.shape[0]
    texts = batch["text"]  # list of strings length B

    fig, axes = plt.subplots(2, B, figsize=(4 * B, 8))
    if B == 1:
        axes = axes.reshape(2, 1)

    for i in range(B):
        # reconstructed image from latents
        axes[0, i].imshow(img_rec[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        short_txt = texts[i]
        if len(short_txt) > 80:
            short_txt = short_txt[:77] + "..."
        axes[0, i].set_title(short_txt, fontsize=8)

        # canny edge map
        canny_i = batch["canny_image"][i, 0].numpy()
        axes[1, i].imshow(canny_i, cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("canny", fontsize=8)

    plt.tight_layout()
    out_img_path = OUTPUT_DIR / "decoded_batch_with_canny.png"
    fig.savefig(out_img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved decoded batch + canny preview to: {out_img_path}")
    print("Done.")