#!/usr/bin/env python3
"""
dataset/dreambooth/dataset.py

Simple DreamBooth-style dataset for Sana/Sana-1.5:

- Takes a folder of instance images (your dog / concept).
- Builds a latent cache once via the Sana DC-AE (AutoencoderDC).
- At train time returns only cached latents.

Returned item:
    {
        "latents": [C, H, W]  (float16 on CPU; move to device in training_step)
    }
"""

from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import torch
from torch.utils.data import Dataset
from diffusers import AutoencoderDC
from torchvision import transforms
from torchvision.transforms.functional import crop
from PIL import Image
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    """
    Basic image-folder dataset with random crop + flip, normalized to [-1, 1].
    """

    def __init__(self, root: Path, size: int = 1024):
        root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in exts]
        if not self.paths:
            raise ValueError(f"No images found in: {root}")
        self.size = size

        self.resize = transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.randcrop = transforms.RandomCrop(size)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ),  # -> [-1,1]
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.resize(img)
        y1, x1, h, w = self.randcrop.get_params(img, (self.size, self.size))
        img = crop(img, y1, x1, h, w)
        img = self.flip(img)
        return self.to_tensor(img)  # [3,H,W], float32 in [-1,1]


class LatentCacheDataset(Dataset):
    """
    One-time FP32 VAE encode -> save latents as float16 .pt files.

    At train time we just load the latents from disk and return them
    (still on CPU; moved to device in the Lightning module).
    """

    def __init__(
        self,
        base_ds: ImageFolderDataset,
        indices: Sequence[int],
        cache_dir: Path,
        vae: Optional[AutoencoderDC] = None,
        device: str = "cuda",
    ):
        self.base = base_ds
        self.indices = list(indices)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._ensure_cache(vae)

    def _path_for(self, gi: int) -> Path:
        stem = self.base.paths[gi].stem
        return self.cache_dir / f"{gi:07d}_{stem}.pt"

    @torch.no_grad()
    def _ensure_cache(self, vae: Optional[AutoencoderDC]):
        missing = [gi for gi in self.indices if not self._path_for(gi).exists()]
        if not missing:
            print(
                f"[latent-cache] found {len(self.indices)} latents in {self.cache_dir.name}"
            )
            return
        assert vae is not None, "VAE required to build latent cache."

        print(f"[latent-cache] building {len(missing)} latents -> {self.cache_dir}")
        vae = vae.to(self.device, dtype=torch.float32).eval()
        sf = float(getattr(vae.config, "scaling_factor", 1.0))

        for gi in tqdm(missing):
            x = self.base[gi].unsqueeze(0).to(self.device, dtype=torch.float32)
            lat = vae.encode(x).latent.float() * sf  # [1,C,H',W']
            torch.save({"latent": lat[0].half().cpu()}, self._path_for(gi))

        print("[latent-cache] done.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        gi = self.indices[i]
        pack = torch.load(self._path_for(gi), map_location="cpu")
        return pack["latent"]  # [C,H,W] float16


class DreamBoothLatentDataset(Dataset):
    """
    Thin wrapper that exposes cached latents as dicts:

        item = { "latents": [C,H,W] }

    so the training code can remain clean.
    """

    def __init__(
        self,
        root: Path,
        cache_dir: Path,
        vae: AutoencoderDC,
        size: int = 1024,
        indices: Optional[Sequence[int]] = None,
        device: str = "cuda",
    ):
        base_ds = ImageFolderDataset(root=root, size=size)
        if indices is None:
            indices = list(range(len(base_ds)))

        self.latent_ds = LatentCacheDataset(
            base_ds=base_ds,
            indices=indices,
            cache_dir=cache_dir,
            vae=vae,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.latent_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lat = self.latent_ds[idx]
        return {"latents": lat}