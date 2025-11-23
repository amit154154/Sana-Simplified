from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from diffusers import AutoencoderDC
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class SegmentFrameLatentDataset(Dataset):
    """
    Dataset for a folder that contains segment subfolders, e.g.:

        root/
          segment_0000/
            frame_000.jpg
            frame_001.jpg
            ...
          segment_0001/
            frame_000.jpg
            frame_001.jpg
            ...

    One-time VAE encode -> cache latents as .pt files (float16).
    At train time returns dict:

        {
            "latents":      [C, H', W'],  float16 on CPU
            "segment_idx":  int,
            "frame_idx":    int,   # local to the segment (0..N-1)
        }

    `target_size` is (H, W) in pixels for the *image fed into the VAE*.
    With Sana DC-AE downscale factor 32, target_size=(1024, 1920)
    ⇒ latents of shape [B, C, 32, 60].
    """

    def __init__(
        self,
        root: Path,
        cache_dir: Path,
        vae: Optional[AutoencoderDC] = None,
        target_size: tuple[int, int] = (1024, 1024),  # (H, W)
        indices: Optional[Sequence[int]] = None,
        device: str = "cuda",
        cache_batch_size: int = 2,   # NEW: batch size for caching
    ):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.target_size = target_size  # (H, W)
        self.cache_batch_size = cache_batch_size

        exts = {".jpg", ".jpeg", ".png", ".webp"}

        # ------------------------------------------------------------------
        # Collect all frames with (segment_idx, frame_idx) metadata
        # ------------------------------------------------------------------
        self.items: List[Tuple[Path, int, int]] = []  # (path, seg_idx, frame_idx)

        seg_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not seg_dirs:
            raise ValueError(f"No segment subfolders found in: {self.root}")

        for seg_idx, seg_dir in enumerate(seg_dirs):
            frame_paths = [
                p for p in sorted(seg_dir.iterdir())
                if p.suffix.lower() in exts
            ]
            if not frame_paths:
                continue
            for frame_idx, p in enumerate(frame_paths):
                self.items.append((p, seg_idx, frame_idx))

        if not self.items:
            raise ValueError(f"No images found under: {self.root}")

        if indices is None:
            self.indices = list(range(len(self.items)))
        else:
            self.indices = list(indices)

        # Preprocessing: resize to target_size (H, W) -> tensor -> normalize to [-1,1]
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    self.target_size,  # (H, W) non-square allowed, e.g. (1024, 1920)
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # -> [-1,1]
            ]
        )

        # Build cache if needed (batched)
        self._ensure_cache(vae)

    # ----------------------------------------------------------------------
    # Cache helpers
    # ----------------------------------------------------------------------
    def _cache_path_for(self, item_idx: int) -> Path:
        path, seg_idx, frame_idx = self.items[item_idx]
        return self.cache_dir / f"seg{seg_idx:04d}_frame{frame_idx:03d}.pt"

    @torch.no_grad()
    def _ensure_cache(self, vae: Optional[AutoencoderDC]):
        missing = [i for i in self.indices if not self._cache_path_for(i).exists()]
        if not missing:
            print(f"[segment-latent-cache] found {len(self.indices)} latents in {self.cache_dir}")
            return

        assert vae is not None, "VAE required to build latent cache."

        print(f"[segment-latent-cache] building {len(missing)} latents -> {self.cache_dir}")
        vae = vae.to(self.device, dtype=torch.float32).eval()
        sf = float(getattr(vae.config, "scaling_factor", 1.0))

        bs = self.cache_batch_size

        with tqdm(total=len(missing), desc="Encoding segments (batched)") as pbar:
            for start in range(0, len(missing), bs):
                batch_indices = missing[start:start + bs]

                batch_imgs = []
                batch_meta = []  # (item_idx, seg_idx, frame_idx)

                # Load & preprocess images on CPU
                for item_idx in batch_indices:
                    img_path, seg_idx, frame_idx = self.items[item_idx]
                    with Image.open(img_path) as im:
                        img = im.convert("RGB")
                    x = self.preprocess(img)  # [3,H,W] float32
                    batch_imgs.append(x)
                    batch_meta.append((item_idx, seg_idx, frame_idx))

                x_batch = torch.stack(batch_imgs, dim=0).to(self.device, dtype=torch.float32)  # [B,3,H,W]

                # Encode whole batch
                lat_batch = vae.encode(x_batch).latent.float() * sf  # [B,C,H',W']

                # Save each latent separately
                for b_idx, (item_idx, seg_idx, frame_idx) in enumerate(batch_meta):
                    torch.save(
                        {
                            "latent":      lat_batch[b_idx].half().cpu(),
                            "segment_idx": seg_idx,
                            "frame_idx":   frame_idx,
                        },
                        self._cache_path_for(item_idx),
                    )

                pbar.update(len(batch_indices))

        print("[segment-latent-cache] done.")

    # ----------------------------------------------------------------------
    # PyTorch Dataset API
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_idx = self.indices[idx]
        _, seg_idx, frame_idx = self.items[item_idx]

        pack = torch.load(self._cache_path_for(item_idx), map_location="cpu")
        lat = pack["latent"]

        return {
            "latents": lat,          # [C,H',W'] float16 (CPU)
            "segment_idx": seg_idx,  # int
            "frame_idx": frame_idx,  # int, local to segment
        }


if __name__ == "__main__":
    from diffusers import AutoencoderDC
    from pathlib import Path
    import lovely_tensors as lt
    import torch
    from torchvision.transforms.functional import to_pil_image

    lt.monkey_patch()

    device = "mps"  # or "cuda" / "cpu"

    vae = AutoencoderDC.from_pretrained(
        "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        subfolder="vae",
    ).to(device)

    root = Path("/Users/mac/PycharmProjects/Sana_Simplified/data_3d_render/top_100_walking_segments")
    cache_dir = Path("/Users/mac/PycharmProjects/Sana_Simplified/data_3d_render/cache_latents_top100_walking")

    ds = SegmentFrameLatentDataset(
        root=root,
        cache_dir=cache_dir,
        vae=vae,
        target_size=(1024, 1920),  # (H, W) -> 1024x1920 input ⇒ 32x60 latent
        device=device,
        cache_batch_size=2,        # try 4–8; increase if memory allows
    )

    sample = ds[0]
    latents = sample["latents"]         # [C, H', W'], float16 on CPU
    print("latent shape:", latents.shape)
    print("segment_idx, frame_idx:", sample["segment_idx"], sample["frame_idx"])

    # -----------------------------
    # Decode the latent
    # -----------------------------
    with torch.no_grad():
        # [1, C, H', W']
        lat = latents.unsqueeze(0).to(device, dtype=torch.float32)

        # Undo scaling factor used during encoding
        sf = float(getattr(vae.config, "scaling_factor", 1.0))
        lat = lat / sf

        # Decode: output in [-1, 1]
        dec = vae.decode(lat).sample  # [1, 3, H, W]

        # Map to [0,1] and convert to PIL
        img = (dec.clamp(-1, 1) + 1) / 2.0  # [1,3,H,W]
        img_pil = to_pil_image(img[0].cpu())

    # -----------------------------
    # Show or save the image
    # -----------------------------
    try:
        from IPython.display import display
        display(img_pil)
    except Exception:
        out_path = cache_dir / f"debug_segment{sample['segment_idx']}_frame{sample['frame_idx']}.png"
        img_pil.save(out_path)
        print(f"Saved decoded image to: {out_path}")