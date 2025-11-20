#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple
import random
import re
import gc

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers import AutoencoderDC, SanaPipeline
import pandas as pd  # add near the top of your file if not already imported


DEVICE = "mps" if torch.backends.mps.is_available() else "cuda"
print(f"[device] {DEVICE} | torch={torch.__version__}")


# ============================================================
# 1) Basic image + latent cache datasets (unchanged)
# ============================================================

class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, size: int = 512):
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in exts]
        if not self.paths:
            raise ValueError(f"No images found in: {root}")
        self.size = size
        self.resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.randcrop = transforms.RandomCrop(size)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]),  # -> [-1,1]
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.resize(img)
        y1, x1, h, w = self.randcrop.get_params(img, (self.size, self.size))
        img = transforms.functional.crop(img, y1, x1, h, w)
        img = self.flip(img)
        return self.to_tensor(img)  # [3,H,W], float32 in [-1,1]


class LatentCacheDataset(Dataset):
    """
    One-time FP32 VAE encode -> save latents as float16 .pt files.

    At train time we just load the latents to memory and move to device.
    """
    def __init__(
        self,
        base_ds: ImageFolderDataset,
        indices: Sequence[int],
        cache_dir: Path,
        vae: Optional[AutoencoderDC] = None,
    ):
        self.base = base_ds
        self.indices = list(indices)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_cache(vae)

    def _path_for(self, gi: int) -> Path:
        stem = self.base.paths[gi].stem
        return self.cache_dir / f"{gi:07d}_{stem}.pt"

    @torch.no_grad()
    def _ensure_cache(self, vae: Optional[AutoencoderDC]):
        missing = [gi for gi in self.indices if not self._path_for(gi).exists()]
        if not missing:
            print(f"[latent-cache] found {len(self.indices)} in {self.cache_dir.name}")
            return
        assert vae is not None, "VAE required to build latent cache."
        print(f"[latent-cache] building {len(missing)} latents -> {self.cache_dir.name}")
        vae = vae.to(DEVICE, dtype=torch.float32).eval()
        for gi in tqdm(missing):
            x = self.base[gi].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            lat = vae.encode(x).latent.float() * float(vae.config.scaling_factor)
            torch.save({"latent": lat[0].half().cpu()}, self._path_for(gi))
        print("[latent-cache] done.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        gi = self.indices[i]
        pack = torch.load(self._path_for(gi), map_location="cpu")
        return pack["latent"]  # [C,H,W] float16 (will be moved/cast in step)


# ============================================================
# 2) Zoom + *single class prompt* dataset for SANA1.5
# ============================================================

class ZoomLatentOneClassDataset(Dataset):
    """
    Folder layout:
        root/
          obj_1/
            zoom_0.png
            zoom_1.png
            ...
          obj_2/
            zoom_0.png
            zoom_3.png
            ...

    Differences vs old version:
      * No CSV, no per-object label text.
      * One global INSTANCE_PROMPT is used for all objects.
      * Text embeddings are precomputed once via SanaPipeline.encode_prompt.

    Returns per item:
      - latent_org_zoom       [C,H,W] (float16)
      - latent_target_zoom    [C,H,W] (float16)
      - zoom_org              [1] float32 (normalized)
      - zoom_target           [1] float32 (normalized)
      - processed_text_condition: str (the shared INSTANCE_PROMPT)
      - text_encoding_pe      [1,T,D] float32
      - text_encoding_pam     [1,T]   int64/bool
      - (optional) siglip_org_embed, siglip_target_embed if siglip_cache_dir is set
    """

    def __init__(
        self,
        root: Path,
        cache_dir: Path,
        vae: AutoencoderDC,
        text_pipe: SanaPipeline,
        instance_prompt: str,
        siglip_cache_dir: Optional[Path] = None,
        zoom_pattern: str = r"zoom_(\d+)",
        exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp"),
        max_sequence_length: int = 300,
        always_zero_org: bool = True,
        allow_same_zoom: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.instance_prompt = instance_prompt
        self._zoom_re = re.compile(zoom_pattern)
        self.exts = {e.lower() for e in exts}
        self.always_zero_org = always_zero_org
        self.allow_same_zoom = allow_same_zoom

        # Optional SigLIP cache (same contract as before)
        if siglip_cache_dir is not None:
            self.siglip_cache_dir = Path(siglip_cache_dir)
            self.siglip_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.siglip_cache_dir = None

        # ---------- 1) scan zoom dirs ----------
        self.objects: List[Dict[str, Any]] = []
        self.items_flat: List[Dict[str, Any]] = []   # used for latent & siglip cache
        gi_counter = 0

        obj_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        for obj_dir in obj_dirs:
            obj_name = obj_dir.name

            zoom_items: List[Tuple[float, int]] = []  # (zoom, gi)

            for p in sorted(obj_dir.iterdir()):
                if p.suffix.lower() not in self.exts:
                    continue
                m = self._zoom_re.search(p.stem)
                if not m:
                    continue
                z = float(m.group(1))  # zoom level from filename

                self.items_flat.append(
                    {
                        "gi": gi_counter,
                        "path": p,
                        "zoom": z,
                        "obj_name": obj_name,
                    }
                )
                zoom_items.append((z, gi_counter))
                gi_counter += 1

            if len(zoom_items) == 0:
                continue

            zoom_items.sort(key=lambda x: x[0])
            if self.always_zero_org:
                levels = [z for z, _ in zoom_items]
                if 0.0 not in levels:
                    # require zoom_0 for this object
                    continue

            self.objects.append(
                {
                    "name": obj_name,
                    "items": zoom_items,  # list[(zoom, gi)]
                }
            )

        if not self.objects:
            raise ValueError(f"No valid objects with zoom_* images in {self.root}")

        print(
            f"[ZoomLatentOneClassDataset] objects={len(self.objects)}, "
            f"single-images={len(self.items_flat)}"
        )

        # ---------- 2) latent cache ----------
        self._ensure_latent_cache(vae)

        # ---------- 3) encode the single class prompt once ----------
        self._encode_instance_prompt(text_pipe, max_sequence_length)

        # Optional: sanity check SigLIP cache
        if self.siglip_cache_dir is not None:
            missing = [
                item for item in self.items_flat
                if not self._siglip_path(item["gi"]).exists()
            ]
            if missing:
                print(
                    f"[warning] {len(missing)} SigLIP embeddings missing under "
                    f"{self.siglip_cache_dir}"
                )
            else:
                print(
                    f"[siglip-cache] found all {len(self.items_flat)} embeddings in "
                    f"{self.siglip_cache_dir.name}"
                )

    def __len__(self) -> int:
        # one index per object; each call samples a random (org,target) zoom pair
        return len(self.objects)

    # ----- latent cache helpers -----

    def _latent_path(self, gi: int) -> Path:
        return self.cache_dir / f"{gi:07d}.pt"

    @torch.no_grad()
    def _ensure_latent_cache(self, vae: AutoencoderDC):
        missing = [item for item in self.items_flat if not self._latent_path(item["gi"]).exists()]
        if not missing:
            print(f"[latent-cache] found all {len(self.items_flat)} latents in {self.cache_dir.name}")
            return

        print(f"[latent-cache] building {len(missing)} latents -> {self.cache_dir}")
        vae = vae.to(DEVICE, dtype=torch.float32).eval()

        for item in tqdm(missing):
            gi = item["gi"]
            path: Path = item["path"]

            img = Image.open(path).convert("RGB")
            x = transforms.ToTensor()(img)
            x = transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])(x)
            x = x.unsqueeze(0).to(DEVICE, dtype=torch.float32)

            lat = vae.encode(x).latent.float() * float(vae.config.scaling_factor)
            payload = {"latent": lat[0].half().cpu()}
            torch.save(payload, self._latent_path(gi))

        print("[latent-cache] done.")

    # ----- SigLIP cache helper (unchanged) -----

    def _siglip_path(self, gi: int) -> Path:
        assert self.siglip_cache_dir is not None
        return self.siglip_cache_dir / f"{gi:07d}.pt"

    # ----- text encodings: ONE class prompt -----

    @torch.no_grad()
    def _encode_instance_prompt(
        self,
        text_pipe: SanaPipeline,
        max_sequence_length: int,
    ):
        """
        Use SanaPipeline.encode_prompt once for the single INSTANCE_PROMPT.

        We store:
          self.pe  : [1,T,D]
          self.pam : [1,T]
        """
        print("[text-encode] encoding INSTANCE_PROMPT once with SanaPipeline...")
        text_pipe = text_pipe.to(DEVICE)

        # We only need positive prompt, no classifier-free guidance
        prompt_embeds, prompt_attention_mask, _, _ = text_pipe.encode_prompt(
            prompt=self.instance_prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        # Move to CPU for dataset storage
        self.pe = prompt_embeds.cpu()              # [1,T,D]
        self.pam = prompt_attention_mask.cpu()     # [1,T]

        # Optionally free heavy parts
        try:
            del text_pipe.text_encoder, text_pipe.tokenizer, text_pipe.transformer, text_pipe.vae
        except Exception:
            pass
        gc.collect()
        print("[text-encode] done.")

    # ----- choose zoom pair -----

    def _choose_zoom_pair(
        self,
        zoom_items: List[Tuple[float, int]],
    ) -> Tuple[float, int, float, int]:
        levels = [z for z, _ in zoom_items]
        z_to_gi = {z: gi for z, gi in zoom_items}

        if self.always_zero_org:
            zoom_org = 0.0
            gi_org = z_to_gi[zoom_org]
            if self.allow_same_zoom:
                zoom_target = random.choice(levels)
            else:
                candidates = [z for z in levels if z != zoom_org]
                zoom_target = random.choice(candidates)
            gi_target = z_to_gi[zoom_target]
        else:
            if self.allow_same_zoom:
                zoom_org = random.choice(levels)
                zoom_target = random.choice(levels)
                gi_org = z_to_gi[zoom_org]
                gi_target = z_to_gi[zoom_target]
            else:
                z1, z2 = random.sample(levels, 2)
                zoom_org, zoom_target = z1, z2
                gi_org = z_to_gi[zoom_org]
                gi_target = z_to_gi[zoom_target]

        return zoom_org, gi_org, zoom_target, gi_target

    # ----- __getitem__ -----

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.objects[idx]
        zoom_items = obj["items"]

        zoom_org, gi_org, zoom_target, gi_target = self._choose_zoom_pair(zoom_items)

        pack_org = torch.load(self._latent_path(gi_org), map_location="cpu")
        pack_target = torch.load(self._latent_path(gi_target), map_location="cpu")

        latent_org = pack_org["latent"]          # [C,H,W] float16
        latent_target = pack_target["latent"]    # [C,H,W] float16

        # normalize zooms e.g. /5 like before
        zoom_org_t = torch.tensor([zoom_org], dtype=torch.float32) / 9.0
        zoom_target_t = torch.tensor([zoom_target], dtype=torch.float32) / 9.0

        out = {
            "latent_org_zoom": latent_org,
            "latent_target_zoom": latent_target,
            "zoom_org": zoom_org_t,
            "zoom_target": zoom_target_t,
            "processed_text_condition": self.instance_prompt,
            "text_encoding_pe": self.pe,    # [1,T,D]
            "text_encoding_pam": self.pam,  # [1,T]
        }

        if self.siglip_cache_dir is not None:
            siglip_org = torch.load(self._siglip_path(gi_org), map_location="cpu")["embedding"]
            siglip_tgt = torch.load(self._siglip_path(gi_target), map_location="cpu")["embedding"]
            out["siglip_org_embed"] = siglip_org
            out["siglip_target_embed"] = siglip_tgt

        return out

class ZoomLatentMultiClassDataset(Dataset):
    """
    Multi-class version of ZoomLatentOneClassDataset.

    Differences:
      * Takes `labels_path` (CSV) instead of a single `instance_prompt`.
      * Expects a per-object text prompt in the CSV.
      * Builds a text-encoding cache directory with one file per object.
      * __getitem__ returns the text encoding for the correct object.

    Expected CSV schema:
      - column 'object' : folder name under `root`
      - column 'label'  : text prompt for that object
        (if 'label' is missing, will try 'prompt')

    Folder layout:
        root/
          obj_1/
            zoom_0.png
            zoom_1.png
            ...
          obj_2/
            zoom_0.png
            zoom_3.png
            ...
    """

    def __init__(
        self,
        root: Path,
        cache_dir: Path,
        vae: AutoencoderDC,
        text_pipe: SanaPipeline,
        labels_path: Path,
        text_cache_dir: Path,
        siglip_cache_dir: Optional[Path] = None,
        zoom_pattern: str = r"zoom_(\d+)",
        exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp"),
        max_sequence_length: int = 300,
        always_zero_org: bool = True,
        allow_same_zoom: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.labels_path = Path(labels_path)
        self.text_cache_dir = Path(text_cache_dir)
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)

        self._zoom_re = re.compile(zoom_pattern)
        self.exts = {e.lower() for e in exts}
        self.always_zero_org = always_zero_org
        self.allow_same_zoom = allow_same_zoom

        # -------- 0) load labels CSV --------
        df = pd.read_csv(self.labels_path)
        if "object" not in df.columns:
            raise ValueError(f"CSV {self.labels_path} must contain an 'object' column.")
        if "label" in df.columns:
            label_col = "label"
        elif "prompt" in df.columns:
            label_col = "prompt"
        else:
            raise ValueError(
                f"CSV {self.labels_path} must contain either 'label' or 'prompt' column."
            )

        self.obj_to_prompt = {
            str(row["object"]): str(row[label_col]).strip()
            for _, row in df.iterrows()
            if pd.notna(row["object"]) and pd.notna(row[label_col])
        }

        # Optional SigLIP cache
        if siglip_cache_dir is not None:
            self.siglip_cache_dir = Path(siglip_cache_dir)
            self.siglip_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.siglip_cache_dir = None

        # ---------- 1) scan zoom dirs ----------
        self.objects: List[Dict[str, Any]] = []
        self.items_flat: List[Dict[str, Any]] = []
        gi_counter = 0

        obj_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        for obj_dir in obj_dirs:
            obj_name = obj_dir.name

            # Skip objects with no label
            if obj_name not in self.obj_to_prompt:
                print(f"[ZoomLatentMultiClassDataset] skipping '{obj_name}' (no label in CSV)")
                continue

            zoom_items: List[Tuple[float, int]] = []  # (zoom, gi)

            for p in sorted(obj_dir.iterdir()):
                if p.suffix.lower() not in self.exts:
                    continue
                m = self._zoom_re.search(p.stem)
                if not m:
                    continue
                z = float(m.group(1))

                self.items_flat.append(
                    {
                        "gi": gi_counter,
                        "path": p,
                        "zoom": z,
                        "obj_name": obj_name,
                    }
                )
                zoom_items.append((z, gi_counter))
                gi_counter += 1

            if len(zoom_items) == 0:
                continue

            zoom_items.sort(key=lambda x: x[0])
            if self.always_zero_org:
                levels = [z for z, _ in zoom_items]
                if 0.0 not in levels:
                    # require zoom_0 for this object
                    continue

            self.objects.append(
                {
                    "name": obj_name,
                    "items": zoom_items,  # list[(zoom, gi)]
                }
            )

        if not self.objects:
            raise ValueError(
                f"No valid objects (with zoom_* images and labels) in {self.root}"
            )

        print(
            f"[ZoomLatentMultiClassDataset] objects={len(self.objects)}, "
            f"single-images={len(self.items_flat)}"
        )

        # ---------- 2) latent cache ----------
        self._ensure_latent_cache(vae)

        # ---------- 3) per-object text encodings ----------
        self._ensure_text_cache(text_pipe, max_sequence_length)

        # Optional: sanity check SigLIP cache
        if self.siglip_cache_dir is not None:
            missing = [
                item for item in self.items_flat
                if not self._siglip_path(item["gi"]).exists()
            ]
            if missing:
                print(
                    f"[warning] {len(missing)} SigLIP embeddings missing under "
                    f"{self.siglip_cache_dir}"
                )
            else:
                print(
                    f"[siglip-cache] found all {len(self.items_flat)} embeddings in "
                    f"{self.siglip_cache_dir.name}"
                )

    def __len__(self) -> int:
        return len(self.objects)  # one index per object

    # ----- latent cache helpers (same pattern as OneClass) -----

    def _latent_path(self, gi: int) -> Path:
        return self.cache_dir / f"{gi:07d}.pt"

    @torch.no_grad()
    def _ensure_latent_cache(self, vae: AutoencoderDC):
        from torchvision import transforms  # make sure it's imported at top

        missing = [
            item for item in self.items_flat
            if not self._latent_path(item["gi"]).exists()
        ]
        if not missing:
            print(
                f"[latent-cache] found all {len(self.items_flat)} latents in "
                f"{self.cache_dir.name}"
            )
            return

        print(f"[latent-cache] building {len(missing)} latents -> {self.cache_dir}")
        vae = vae.to(DEVICE, dtype=torch.float32).eval()

        for item in tqdm(missing, desc="[latent-cache] encoding"):
            gi = item["gi"]
            path: Path = item["path"]

            img = Image.open(path).convert("RGB")
            x = transforms.ToTensor()(img)
            x = transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])(x)
            x = x.unsqueeze(0).to(DEVICE, dtype=torch.float32)

            lat = vae.encode(x).latent.float() * float(vae.config.scaling_factor)
            payload = {"latent": lat[0].half().cpu()}
            torch.save(payload, self._latent_path(gi))

        print("[latent-cache] done.")

    # ----- SigLIP cache helper -----

    def _siglip_path(self, gi: int) -> Path:
        assert self.siglip_cache_dir is not None
        return self.siglip_cache_dir / f"{gi:07d}.pt"

    # ----- text encoding cache helpers -----

    def _text_path(self, obj_name: str) -> Path:
        # one file per object
        safe = obj_name.replace("/", "_")
        return self.text_cache_dir / f"{safe}.pt"

    @torch.no_grad()
    def _ensure_text_cache(
        self,
        text_pipe: SanaPipeline,
        max_sequence_length: int,
    ):
        """
        Use SanaPipeline.encode_prompt once per object's label and cache to disk.

        Each file stores:
          - "pe"   : [1,T,D] float32
          - "pam"  : [1,T]   attention mask
          - "text" : original prompt string
        """
        missing = []
        for obj in self.objects:
            obj_name = obj["name"]
            if not self._text_path(obj_name).exists():
                missing.append(obj_name)

        if not missing:
            print(
                f"[text-cache] found encodings for all {len(self.objects)} objects "
                f"in {self.text_cache_dir.name}"
            )
            return

        print(f"[text-cache] building encodings for {len(missing)} objects...")
        text_pipe = text_pipe.to(DEVICE)

        for obj_name in tqdm(missing, desc="[text-cache] encoding"):
            prompt = self.obj_to_prompt[obj_name]

            prompt_embeds, prompt_attention_mask, _, _ = text_pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                max_sequence_length=max_sequence_length,
            )

            payload = {
                "pe": prompt_embeds.cpu(),          # [1,T,D]
                "pam": prompt_attention_mask.cpu(), # [1,T]
                "text": prompt,
            }
            torch.save(payload, self._text_path(obj_name))

        # Free heavy parts from pipeline to save RAM
        try:
            del text_pipe.text_encoder, text_pipe.tokenizer
            del text_pipe.transformer, text_pipe.vae
        except Exception:
            pass
        gc.collect()
        print("[text-cache] done.")

    # ----- choose zoom pair (same as OneClass) -----

    def _choose_zoom_pair(
        self,
        zoom_items: List[Tuple[float, int]],
    ) -> Tuple[float, int, float, int]:
        levels = [z for z, _ in zoom_items]
        z_to_gi = {z: gi for z, gi in zoom_items}

        if self.always_zero_org:
            zoom_org = 0.0
            gi_org = z_to_gi[zoom_org]
            if self.allow_same_zoom:
                zoom_target = random.choice(levels)
            else:
                candidates = [z for z in levels if z != zoom_org]
                zoom_target = random.choice(candidates)
            gi_target = z_to_gi[zoom_target]
        else:
            if self.allow_same_zoom:
                zoom_org = random.choice(levels)
                zoom_target = random.choice(levels)
                gi_org = z_to_gi[zoom_org]
                gi_target = z_to_gi[zoom_target]
            else:
                z1, z2 = random.sample(levels, 2)
                zoom_org, zoom_target = z1, z2
                gi_org = z_to_gi[zoom_org]
                gi_target = z_to_gi[zoom_target]

        return zoom_org, gi_org, zoom_target, gi_target

    # ----- __getitem__ -----

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.objects[idx]
        obj_name = obj["name"]
        zoom_items = obj["items"]

        zoom_org, gi_org, zoom_target, gi_target = self._choose_zoom_pair(zoom_items)

        pack_org = torch.load(self._latent_path(gi_org), map_location="cpu")
        pack_target = torch.load(self._latent_path(gi_target), map_location="cpu")

        latent_org = pack_org["latent"]
        latent_target = pack_target["latent"]

        # normalized zooms (0–9 -> [0,1] if you keep /9.0)
        zoom_org_t = torch.tensor([zoom_org], dtype=torch.float32) / 9.0
        zoom_target_t = torch.tensor([zoom_target], dtype=torch.float32) / 9.0

        # load per-object text encoding
        text_pack = torch.load(self._text_path(obj_name), map_location="cpu")
        pe = text_pack["pe"]   # [1,T,D]
        pam = text_pack["pam"] # [1,T]
        prompt = text_pack["text"]

        out = {
            "latent_org_zoom": latent_org,
            "latent_target_zoom": latent_target,
            "zoom_org": zoom_org_t,
            "zoom_target": zoom_target_t,
            "processed_text_condition": prompt,
            "text_encoding_pe": pe,
            "text_encoding_pam": pam,
        }

        if self.siglip_cache_dir is not None:
            siglip_org = torch.load(self._siglip_path(gi_org), map_location="cpu")["embedding"]
            siglip_tgt = torch.load(self._siglip_path(gi_target), map_location="cpu")["embedding"]
            out["siglip_org_embed"] = siglip_org
            out["siglip_target_embed"] = siglip_tgt

        return out

# ============================================================
# 3) Simple __main__ to test dataset with SANA1.5_1.6B
# ============================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DATA_ROOT = Path(
        "/Users/mac/PycharmProjects/rotation_image_generation/data_zoom/dataset_zoom_10"
    )
    OUTPUT_DIR = Path("zoom_dataset_test_sana15")
    PRETRAINED_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
    SIGLIP_CACHE_DIR = Path(
        "/Users/mac/PycharmProjects/rotation_image_generation/data_zoom/dataset_zoom_10_siglip"
    )
    INSTANCE_PROMPT = "a 3d game asset rendered in a white background"
    BATCH_SIZE = 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATENT_CACHE_DIR = OUTPUT_DIR / "latent_cache"

    # --- load VAE & text pipeline from SANA1.5 diffusers repo ---
    vae = AutoencoderDC.from_pretrained(PRETRAINED_MODEL, subfolder="vae").eval()
    text_pipe = SanaPipeline.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
    )

    ds = ZoomLatentOneClassDataset(
        root=DATA_ROOT,
        cache_dir=LATENT_CACHE_DIR,
        vae=vae,
        text_pipe=text_pipe,
        instance_prompt=INSTANCE_PROMPT,
        siglip_cache_dir=SIGLIP_CACHE_DIR,  # or None if you don't want SigLIP
        always_zero_org=True,
        allow_same_zoom=False,
    )

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dl))

    print("\n[batch] keys:", list(batch.keys()))
    print("latent_org_zoom:", batch["latent_org_zoom"].shape)
    print("latent_target_zoom:", batch["latent_target_zoom"].shape)
    if "siglip_org_embed" in batch:
        print("siglip_org_embed:", batch["siglip_org_embed"].shape)
        print("siglip_target_embed:", batch["siglip_target_embed"].shape)
    print("zoom_org:", batch["zoom_org"])
    print("zoom_target:", batch["zoom_target"])
    print("text_encoding_pe:", batch["text_encoding_pe"].shape)
    print("text_encoding_pam:", batch["text_encoding_pam"].shape)
    print("processed_text_condition example:", batch["processed_text_condition"][0])

    # -------------------------------------------------------
    # Decode latents and plot org/target images for the batch
    # -------------------------------------------------------
    with torch.no_grad():
        vae = vae.to(DEVICE, dtype=torch.float32).eval()
        scaling_factor = float(vae.config.scaling_factor)

        lat_org = batch["latent_org_zoom"].to(DEVICE).float() / scaling_factor
        lat_tgt = batch["latent_target_zoom"].to(DEVICE).float() / scaling_factor

        img_org = vae.decode(lat_org, return_dict=False)[0]   # [-1,1]
        img_tgt = vae.decode(lat_tgt, return_dict=False)[0]   # [-1,1]

        img_org = (img_org.clamp(-1, 1) + 1) / 2.0
        img_tgt = (img_tgt.clamp(-1, 1) + 1) / 2.0

        img_org = img_org.cpu()
        img_tgt = img_tgt.cpu()

    B = img_org.shape[0]
    fig, axes = plt.subplots(2, B, figsize=(4 * B, 8))
    if B == 1:
        axes = axes.reshape(2, 1)

    for i in range(B):
        axes[0, i].imshow(img_org[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        z_org = float(batch["zoom_org"][i].item() * 5.0)
        axes[0, i].set_title(f"org (z={z_org:.1f})")

        axes[1, i].imshow(img_tgt[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        z_tgt = float(batch["zoom_target"][i].item() * 5.0)
        axes[1, i].set_title(f"target (z={z_tgt:.1f})")

    plt.tight_layout()
    out_img_path = OUTPUT_DIR / "decoded_batch.png"
    fig.savefig(out_img_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved decoded batch preview to: {out_img_path}")
    print("Done.")