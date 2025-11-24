#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusers import AutoencoderDC, SanaTransformer2DModel
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

import lovely_tensors as lt

lt.monkey_patch()

# from muon import Muon   # ❌ remove this

try:
    from bitsandbytes.optim import AdamW8bit
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# ============================================================
# 1) Small Canny ControlNet for Sana (uses timestep)
#    - now with deeper hint_block
#    - and sequential per-block token updates
# ============================================================

class SanaCannyControlNet(nn.Module):
    """
    Lighter ControlNet-style branch for Sana.

    - Reuses base_model's:
        * patch_embed
        * time_embed
        * caption_projection
        * caption_norm
    - Deep-copies ONLY the first `control_depth` transformer blocks.
    - Runs only on the Canny hint (noisy latents used just for shape/device).
    - For each cloned transformer block index i > 0, produces a residual (B, N, D),
      passed to base Sana via `controlnet_block_samples[i-1]`.

    This keeps memory much lower than cloning the full stack.
    """

    def __init__(
        self,
        base_model: SanaTransformer2DModel,
        conditioning_channels: int = 1,
        conditioning_scale: float = 1.0,
        control_depth: int = 8,   # <--- NEW: how many transformer blocks to copy
    ):
        super().__init__()

        self.config = base_model.config
        self.conditioning_scale = conditioning_scale

        self.latent_channels = self.config.in_channels
        self.patch_size = self.config.patch_size

        # inner_dim is the token dim = proj_out.in_features in diffusers Sana
        inner_dim = base_model.proj_out.in_features
        full_num_layers = len(base_model.transformer_blocks)

        # clamp control_depth so we don't exceed the real depth
        self.control_depth = max(2, min(control_depth, full_num_layers))

        # -------- 1) Small hint block on Canny (image space -> latent space) --------
        # Canny: [B, 1, Hc, Wc] -> [B, latent_channels, H', W']
        self.hint_block = nn.Sequential(
            nn.Conv2d(conditioning_channels, self.latent_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.latent_channels, self.latent_channels, 3, padding=1),
            nn.SiLU(),
        )

        # -------- 2) REUSE base_model embeddings / projections --------
        # These are shared with the frozen base model (no extra param memory).
        self.patch_embed = base_model.patch_embed          # shared
        self.time_embed = base_model.time_embed            # shared
        self.caption_projection = base_model.caption_projection  # shared
        self.caption_norm = base_model.caption_norm        # shared

        # -------- 3) Deep-copy ONLY the first `control_depth` blocks --------
        full_blocks = list(base_model.transformer_blocks)
        self.transformer_blocks = nn.ModuleList(
            copy.deepcopy(full_blocks[i]) for i in range(self.control_depth)
        )

        # -------- 4) Per-block zero-conv heads (depth-1 of them) --------
        # matches base Sana injection pattern:
        #   if 0 < index_block <= len(controlnet_block_samples):
        #       hidden_states = hidden_states + controlnet_block_samples[index_block-1]
        self.zero_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(inner_dim),
                    nn.Linear(inner_dim, inner_dim),
                )
                for _ in range(self.control_depth - 1)
            ]
        )

        # zero-init last Linear in each head (ControlNet trick)
        for block in self.zero_convs:
            lin = block[-1]
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        # make sure newly added params (hint_block + copied blocks + zero_convs) are trainable
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, C, H, W] (noisy latents, used for shape/device)
        canny_image: torch.Tensor,               # [B, 1 or 3, Hc, Wc]
        timestep: torch.Tensor,                  # [B] or scalar
        encoder_hidden_states: torch.Tensor,     # [B, T, caption_channels]
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, T] or already bias
        guidance: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, ...] | Dict[str, Tuple[torch.Tensor, ...]]:

        device = hidden_states.device
        out_dtype = hidden_states.dtype  # dtype we want to hand back to base Sana

        B, C, H, W = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = H // p, W // p

        # ---- 1) Prepare Canny -> latent-like features ----
        x = canny_image.to(device=device, dtype=out_dtype)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.hint_block(x)  # [B, latent_channels, H, W]

        # ---- 2) Patch embed (shared with base model) ----
        tokens = self.patch_embed(x)  # [B, N, D]

        # ---- 3) Time embedding (shared with base model) ----
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device)

        if guidance is not None:
            timestep_emb, embedded_timestep = self.time_embed(
                timestep,
                guidance=guidance,
                hidden_dtype=tokens.dtype,
            )
        else:
            timestep_emb, embedded_timestep = self.time_embed(
                timestep,
                batch_size=B,
                hidden_dtype=tokens.dtype,
            )

        # ---- 4) Text / caption projection & norm (shared) ----
        encoder_hidden_states_proj = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states_proj = encoder_hidden_states_proj.view(B, -1, tokens.shape[-1])
        encoder_hidden_states_proj = self.caption_norm(encoder_hidden_states_proj)

        # ---- 5) Convert masks to bias (same as diffusers code) ----
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(tokens.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(tokens.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # ---- 6) Run *only* the cloned transformer blocks, collect residuals ----
        controlnet_block_samples: List[torch.Tensor] = []

        for i, block in enumerate(self.transformer_blocks):
            tokens = block(
                tokens,
                attention_mask,
                encoder_hidden_states_proj,
                encoder_attention_mask,
                timestep_emb,
                post_patch_height,
                post_patch_width,
            )

            # skip block 0 → residuals start from block 1
            if i > 0:
                delta = self.zero_convs[i - 1](tokens)      # [B, N, D]
                residual = delta * self.conditioning_scale
                residual = residual.to(out_dtype)           # match base transformer dtype
                controlnet_block_samples.append(residual)

        # We return length = control_depth - 1.
        # Base Sana will add them only to the first `len(...)` blocks:
        #   if 0 < index_block <= len(controlnet_block_samples):
        #       hidden_states += controlnet_block_samples[index_block-1]

        if not return_dict:
            return tuple(controlnet_block_samples)

        return {"controlnet_block_samples": tuple(controlnet_block_samples)}
# ============================================================
# 2) Lightning wrapper: Sana + small Canny ControlNet
# ============================================================

class SanaControlNetModel(pl.LightningModule):
    """
    LightningModule wrapper for Sana + small Canny ControlNet in Flow-Matching training.

    Expected batch:
      - "latents":      [B,C,H',W']
      - "pe":           [B,T,D_text]  (text encoder output from SanaPipeline.encode_prompt)
      - "pam":          [B,T]         (attention mask)
      - "canny_image":  [B,1,H_img,W_img]
      - "text":         list[str]     (optional, for logging)
    """

    def __init__(
            self,
            transformer: SanaTransformer2DModel,
            vae: AutoencoderDC,
            scheduler,  # FlowMatchEulerDiscreteScheduler
            lr: float = 1e-4,
            weighting_scheme: str = "none",
            logit_mean: float = 0.0,
            logit_std: float = 1.0,
            mode_scale: float = 1.29,
            ckpt_root: Optional[Path] = None,
            log_wandb: bool = False,
            sample_every: int = 500,
            num_inference_steps: int = 20,
    ):
        super().__init__()

        self.transformer = transformer
        self.vae = vae

        # 1) Build small ControlNet FIRST (copies config, no backbone clone)
        self.control_net = SanaCannyControlNet(
            base_model=self.transformer,
            conditioning_channels=1,
            conditioning_scale=1.5,  # you can play 1.0–3.0
            control_depth=8,  # <-- try 6, 8, or 10
        )
        # match dtype with base transformer, e.g. bfloat16
        self.control_net.to(dtype=self.transformer.dtype)

        # ensure all ControlNet params are trainable
        for p in self.control_net.parameters():
            p.requires_grad_(True)

        # 2) Freeze the base transformer (we only train ControlNet)
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self.transformer.eval()

        self.lr = lr

        # separate schedulers for train vs inference
        self.train_scheduler = copy.deepcopy(scheduler)
        self.infer_scheduler = copy.deepcopy(scheduler)

        self.weighting_scheme = weighting_scheme
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.mode_scale = float(mode_scale)

        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path("checkpoints")
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        self.log_wandb = log_wandb
        self.sample_every = int(sample_every)
        self.num_inference_steps = int(num_inference_steps)
        self.example_dir = self.ckpt_root / "samples"
        self.example_dir.mkdir(parents=True, exist_ok=True)

        self.save_hyperparameters(
            ignore=[
                "transformer",
                "vae",
                "scheduler",
                "control_net",
                "train_scheduler",
                "infer_scheduler",
            ]
        )

        # VAE: frozen, only used for decoding samples
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    # ------------------------------------------------
    # helper: sigmas + stats
    # ------------------------------------------------
    def _get_sigmas(self, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @staticmethod
    def _compute_controlnet_stats(controlnet_block_samples: List[torch.Tensor]) -> dict:
        """
        Aggregate distribution stats over all controlnet_block_samples:
          - mean_abs_all
          - std_abs_all
          - max_abs_all
          - mean_abs_first_block
          - mean_abs_last_block
        """
        if not controlnet_block_samples:
            return {}

        device = controlnet_block_samples[0].device
        sum_abs = torch.zeros((), device=device, dtype=torch.float32)
        sum_sq = torch.zeros((), device=device, dtype=torch.float32)
        total_elems = 0
        max_abs = torch.zeros((), device=device, dtype=torch.float32)

        first_mean = None
        last_mean = None

        num_blocks = len(controlnet_block_samples)

        for idx, b in enumerate(controlnet_block_samples):
            bf = b.float()
            abs_b = bf.abs()

            if idx == 0:
                first_mean = abs_b.mean()
            if idx == num_blocks - 1:
                last_mean = abs_b.mean()

            total_elems += bf.numel()
            sum_abs += abs_b.sum()
            sum_sq += (bf ** 2).sum()
            max_abs = torch.maximum(max_abs, abs_b.max())

        if total_elems == 0:
            return {}

        mean_abs = sum_abs / total_elems
        var = sum_sq / total_elems - mean_abs ** 2
        std_abs = torch.sqrt(torch.clamp(var, min=0.0))

        stats = {
            "mean_abs_all": mean_abs.detach().cpu(),
            "std_abs_all": std_abs.detach().cpu(),
            "max_abs_all": max_abs.detach().cpu(),
        }

        if first_mean is not None:
            stats["mean_abs_first_block"] = first_mean.detach().cpu()
        if last_mean is not None:
            stats["mean_abs_last_block"] = last_mean.detach().cpu()

        return stats

    # ------------------------------------------------
    # core flow-matching step
    # ------------------------------------------------
    def _forward_flow_matching(
            self,
            x0: torch.Tensor,
            pe: torch.Tensor,
            pam: torch.Tensor,
            canny_image: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[dict]]:

        B = x0.shape[0]
        device = self.device

        x0 = x0.to(device=device, dtype=self.transformer.dtype)
        noise = torch.randn_like(x0)

        # Flow-Matching time sampling
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=B,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        ).to(device)

        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        indices_cpu = indices.to(self.train_scheduler.timesteps.device)
        timesteps = self.train_scheduler.timesteps[indices_cpu].to(device=device)  # [B]

        sigmas = self._get_sigmas(timesteps, n_dim=x0.ndim, dtype=x0.dtype)

        # noisy latent
        z_t = (1.0 - sigmas) * x0 + sigmas * noise

        pe_in = pe.to(device=device)
        pam_in = pam

        # --- ControlNet branch (full cloned transformer) ---
        controlnet_block_samples = self.control_net(
            hidden_states=z_t,  # noisy latents (for shape/device)
            canny_image=canny_image,  # [B, 1, H_img, W_img]
            timestep=timesteps,  # int timesteps [B]
            encoder_hidden_states=pe_in.to(self.transformer.dtype),
            encoder_attention_mask=pam_in,
            guidance=None,  # no guidance embeddings for now
            return_dict=False,
        )

        control_stats = None
        if self.log_wandb:
            control_stats = self._compute_controlnet_stats(list(controlnet_block_samples))

        # --- Base Sana with control injection ---
        model_pred = self.transformer(
            hidden_states=z_t,
            encoder_hidden_states=pe_in.to(self.transformer.dtype),
            encoder_attention_mask=pam_in,
            timestep=timesteps.to(self.transformer.dtype),
            return_dict=False,
            controlnet_block_samples=controlnet_block_samples,
        )[0]

        model_pred = model_pred.float()
        x0_f = x0.float()
        noise_f = noise.float()

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas,
        ).float()

        target = (noise_f - x0_f)

        loss = torch.mean(
            (weighting * (model_pred - target) ** 2).reshape(B, -1),
            dim=1,
        ).mean()

        return loss, control_stats

    # ------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------
    def training_step(self, batch, batch_idx):
        x0 = batch["latents"].to(self.device)
        pe = batch["pe"].to(self.device)
        pam = batch["pam"].to(self.device)
        canny_image = batch["canny_image"].to(self.device)

        loss, control_stats = self._forward_flow_matching(x0, pe, pam, canny_image)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if control_stats is not None:
            for k, v in control_stats.items():
                self.log(f"control/{k}", v, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def _sample_and_log_from_batch(self, batch, batch_idx):
        if self.sample_every <= 0:
            return
        if not self.log_wandb:
            return
        if not isinstance(self.logger, WandbLogger):
            return

        import numpy as np
        import wandb

        x0 = batch["latents"].to(self.device, dtype=self.transformer.dtype)
        pe = batch["pe"].to(self.device)
        pam = batch["pam"].to(self.device)
        canny = batch["canny_image"].to(self.device)
        texts = batch.get("text", None)

        B = x0.shape[0]
        if B == 0:
            return
        B_s = min(2, B)

        x0 = x0[:B_s]
        pe = pe[:B_s]
        pam = pam[:B_s]
        canny = canny[:B_s]

        if isinstance(texts, (list, tuple)):
            texts = list(texts)[:B_s]
        else:
            texts = None

        scheduler = self.infer_scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps.to(self.device)

        latents = torch.randn_like(x0)

        for t in timesteps:
            t_scalar = t.to(self.device) if torch.is_tensor(t) else torch.tensor(t, device=self.device)
            t_batch = t_scalar.expand(latents.shape[0])
            t_batch_model = t_batch.to(self.transformer.dtype)

            controlnet_block_samples = self.control_net(
                hidden_states=latents,
                canny_image=canny,
                timestep=t_batch,  # int64 [B]
                encoder_hidden_states=pe.to(self.transformer.dtype),
                encoder_attention_mask=pam,
                guidance=None,
                return_dict=False,
            )

            model_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=pe.to(self.transformer.dtype),
                encoder_attention_mask=pam,
                timestep=t_batch_model,
                return_dict=False,
                controlnet_block_samples=controlnet_block_samples,
            )[0]

            step_out = scheduler.step(
                model_output=model_pred,
                timestep=t_scalar,
                sample=latents,
                return_dict=True,
            )
            latents = step_out.prev_sample

        vae = self.vae.to(self.device, dtype=torch.float32).eval()
        scaling_factor = float(vae.config.scaling_factor)

        gen_dec = latents.float() / scaling_factor
        gt_dec = x0.float() / scaling_factor

        img_gen = vae.decode(gen_dec, return_dict=False)[0]
        img_gt = vae.decode(gt_dec, return_dict=False)[0]

        img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0
        img_gt = (img_gt.clamp(-1, 1) + 1) / 2.0

        img_gen = img_gen.detach().cpu()
        img_gt = img_gt.detach().cpu()
        canny_cpu = canny.detach().cpu()

        images = []
        for i in range(B_s):
            gen_np = img_gen[i].permute(1, 2, 0).numpy()
            gt_np = img_gt[i].permute(1, 2, 0).numpy()
            canny_np = canny_cpu[i, 0].numpy()
            canny_rgb = np.stack([canny_np] * 3, axis=-1)

            txt = texts[i] if texts is not None and i < len(texts) else ""
            cap_prefix = f"text: {txt} | step {int(self.global_step)}"

            images.append(wandb.Image(gen_np, caption=f"{cap_prefix} | gen[{i}]"))
            images.append(wandb.Image(gt_np, caption=f"{cap_prefix} | gt[{i}]"))
            images.append(wandb.Image(canny_rgb, caption=f"{cap_prefix} | canny[{i}]"))

        self.logger.experiment.log(
            {
                "samples": images,
                "global_step": int(self.global_step),
            }
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every > 0 and (self.global_step + 1) % self.sample_every == 0:
            self._sample_and_log_from_batch(batch, batch_idx)

    def configure_optimizers(self):
        """
        Use a modern, efficient optimizer:

        - If CUDA & bitsandbytes are available: AdamW8bit (8-bit AdamW, SOTA-ish & memory efficient)
        - Otherwise: standard torch.optim.AdamW

        LR stays self.lr (e.g. 3e-4 or 1e-4, as you set in the training script).
        """

        params = self.control_net.parameters()

        if torch.cuda.is_available() and _HAS_BNB:
            # 8-bit AdamW (super memory-efficient, widely used)
            optimizer = AdamW8bit(
                params,
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            print("[optimizer] Using bitsandbytes AdamW8bit")
        else:
            # fallback: standard AdamW
            optimizer = torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            print("[optimizer] Using torch.optim.AdamW")

        return optimizer

    def on_fit_end(self):
        out_dir = self.ckpt_root / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.control_net.state_dict(), out_dir / "controlnet.pt")