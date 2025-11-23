import copy
from pathlib import Path
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusers import SanaTransformer2DModel, AutoencoderDC, SanaPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft.utils import get_peft_model_state_dict
from copy import deepcopy
from tqdm import tqdm
import lovely_tensors as lt

lt.monkey_patch()


class SanaOneClassLoRA(pl.LightningModule):
    """
    Super-simple one-class LoRA fine-tuning for Sana 1.5 / Sana Sprint.

    - No zoom conditioning.
    - Optional learned frame-id token (per-frame conditioning).
    - No text encoder: uses *fixed* text embeddings (pe, pam) provided at init.
    - Dataset only needs to provide latents x0 and frame_idx (if frame token is used).

    Expected dataset batch keys:
      - "latents":   [B, C, H, W]   (VAE latents x0 = data, already scaled by scaling_factor)
      - "frame_idx": [B] int        (0..frames_per_segment-1), if use_frame_token=True
      - (optional) "processed_text_condition": list[str] or tensor/str (for logging only)
    """

    def __init__(
        self,
        transformer: SanaTransformer2DModel,
        vae: AutoencoderDC,
        scheduler,  # FlowMatchEulerDiscreteScheduler
        pe: torch.Tensor,          # base text embeddings [1,T,D] or [T,D]
        pam: Optional[torch.Tensor] = None,  # base attn mask [1,T] or [T]
        lr: float = 1e-4,
        do_lora: bool = True,
        example_dir: Path = Path("samples_one_class"),
        save_examples_every_steps: int = 1000,
        save_pt_every_steps: int = 1000,
        lora_freeze_steps: int = 0,
        instance_prompt: Optional[str] = None,
        num_sample_steps: int = 30,
        # flow-matching weighting config (matches Sana/SD3 trainer style)
        weighting_scheme: str = "none",  # "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        ckpt_root: Path = Path("one_class_lora"),
        # NEW: optional frame-id token
        use_frame_token: bool = False,
        max_frame_idx: int = 150,   # used to normalize idx: idx / max_frame_idx
    ):
        super().__init__()

        self.ckpt_root = Path(ckpt_root)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        self.transformer = transformer
        self.vae = vae
        self.lr = lr
        self.do_lora = do_lora
        self.example_dir = example_dir
        self.save_every_steps = int(save_examples_every_steps)
        self.save_pt_every_steps = int(save_pt_every_steps)
        self.lora_freeze_steps = int(lora_freeze_steps)
        self._lora_unfrozen = (self.lora_freeze_steps == 0)
        self.instance_prompt = instance_prompt
        self.num_sample_steps = int(num_sample_steps)

        self.use_frame_token = bool(use_frame_token)
        self.max_frame_idx = int(max_frame_idx)

        # -----------------------
        # flow-matching schedulers
        # -----------------------
        self.train_scheduler = copy.deepcopy(scheduler)
        self.infer_scheduler = copy.deepcopy(scheduler)

        self.weighting_scheme = weighting_scheme
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.mode_scale = float(mode_scale)

        # -----------------------
        # Fixed text conditioning (pe, pam)
        # -----------------------
        # Normalize shapes: pe -> [1,T,D], pam -> [1,T]
        if pe.dim() == 2:  # [T,D]
            pe = pe.unsqueeze(0)
        assert pe.dim() == 3, f"pe must be [1,T,D] or [T,D], got {pe.shape}"
        self.register_buffer("base_pe", pe.clone().detach(), persistent=False)

        if pam is None:
            pam = torch.ones(pe.shape[:2], dtype=torch.bool)  # [1,T]
        else:
            if pam.dim() == 1:  # [T]
                pam = pam.unsqueeze(0)
            assert pam.dim() == 2, f"pam must be [1,T] or [T], got {pam.shape}"
            pam = pam.to(torch.bool)
        self.register_buffer("base_pam", pam.clone().detach(), persistent=False)

        text_hidden_dim = self.base_pe.shape[-1]

        # -----------------------
        # OPTIONAL: frame-id projection -> frame token
        # -----------------------
        if self.use_frame_token:
            self.frame_proj = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, text_hidden_dim),
                nn.LayerNorm(text_hidden_dim),
            )
            # start as "no-op": last linear zero-init
            last_linear = self.frame_proj[2]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        else:
            self.frame_proj = None

        # -----------------------
        # LoRA params only
        # -----------------------
        self.lora_params = []
        if self.do_lora:
            # Freeze everything by default; only LoRA layers trainable
            for name, p in self.transformer.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad_(True)
                    self.lora_params.append(p)
                else:
                    p.requires_grad_(False)
        else:
            # No LoRA: fully freeze transformer (this class is mainly for LoRA)
            for _, p in self.transformer.named_parameters():
                p.requires_grad_(False)

        # Optional: LoRA warmup
        if self.do_lora and self.lora_freeze_steps > 0:
            for p in self.lora_params:
                p.requires_grad_(False)

        # VAE always frozen
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        # For logging samples
        self._ex_text = instance_prompt or "one-class"
        self._ex_latent_shape = None
        self._logged_ref_image = False

        self.save_hyperparameters(
            ignore=[
                "transformer",
                "vae",
                "train_scheduler",
                "infer_scheduler",
                "pe",
                "pam",
            ]
        )

    # --------------------------
    # helpers: frame token
    # --------------------------

    def _append_frame_token(
        self,
        pe: torch.Tensor,           # [B,T,D]
        pam: torch.Tensor,          # [B,T]
        frame_norm: torch.Tensor,   # [B,1] normalized idx (idx / max_frame_idx)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append a learned frame-id token to the text sequence.

        frame_norm: [B,1] values in roughly [0, 1].
        """
        if self.frame_proj is None:
            return pe, pam

        tok = self.frame_proj(frame_norm)      # [B,D_text]
        tok = tok.unsqueeze(1)                 # [B,1,D_text]

        pe = torch.cat([pe, tok], dim=1)       # [B,T+1,D]
        pam = F.pad(pam, (0, 1), value=True)   # [B,T+1]
        return pe, pam

    # --------------------------
    # scheduler helpers (Sana style)
    # --------------------------

    def _get_sigmas(self, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # --------------------------
    # flow-matching core
    # --------------------------

    def _forward_flow_matching(
        self,
        x0: torch.Tensor,  # [B,C,H,W]
        pe: torch.Tensor,  # [B,T,D]
        pam: torch.Tensor,  # [B,T]
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = self.device

        x0 = x0.to(device=device, dtype=self.transformer.dtype)
        noise = torch.randn_like(x0)

        # sample timesteps using the official density trick
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

        z_t = (1.0 - sigmas) * x0 + sigmas * noise

        pe_in = pe.to(self.transformer.dtype)
        model_pred = self.transformer(
            hidden_states=z_t,
            encoder_hidden_states=pe_in,
            encoder_attention_mask=pam,
            timestep=timesteps.to(self.transformer.dtype),
            return_dict=False,
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

        return loss

    # --------------------------
    # decode helper
    # --------------------------

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B,C,H,W] in VAE latent scale (already multiplied by scaling_factor)
        returns [B,H,W,3] uint8 on CPU
        """
        vae = self.vae
        sf = float(vae.config.scaling_factor)

        lat = latents.to(self.device, dtype=vae.dtype) / sf
        out = vae.decode(lat, return_dict=False)[0]  # [-1,1]
        img = (out.clamp(-1, 1) + 1.0) * 0.5  # [0,1]
        img = (img.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1))
        return img.cpu()

    # --------------------------
    # simple sampling (with or without frame token)
    # --------------------------

    @torch.no_grad()
    def _sample_images(self, num_samples: int = 4):
        """
        Generate a small batch of samples with fixed pe/pam and log to W&B.

        If use_frame_token:
          - Use a SINGLE base noise seed.
          - Use num_samples different frame indices, uniformly spaced in [0, max_frame_idx).
          - For each index, append a frame token with that normalized id and run sampling.
        Otherwise:
          - Vectorized sampling over B=num_samples with shared pe/pam.
        """
        self.example_dir.mkdir(parents=True, exist_ok=True)
        step_dir = self.example_dir / f"step_{self.global_step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        device = self.device
        dtype = self.transformer.dtype

        if self._ex_latent_shape is None:
            raise RuntimeError("ex_latent_shape is None; run a training step first.")

        _, c, h, w = self._ex_latent_shape

        from PIL import Image
        wandb_images = []

        # ----------------------------------------------------
        # Case 1: using frame token -> same seed, varied frame_id
        # ----------------------------------------------------
        if self.use_frame_token and self.frame_proj is not None:
            # Base pe/pam (no extra frame token)
            base_pe = self.base_pe.to(device=device, dtype=dtype)   # [1,T,D]
            base_pam = self.base_pam.to(device=device)              # [1,T]

            # Shared base noise
            base_noise = torch.randn((1, c, h, w), device=device, dtype=dtype)

            # Choose frame indices uniformly in [0, max_frame_idx)
            frame_ids = torch.linspace(
                0,
                max(self.max_frame_idx - 1, 1),
                steps=num_samples,
                device=device,
                dtype=torch.float32,
            )
            frame_ids_rounded = frame_ids.round()  # for display

            infer_scheduler = deepcopy(self.infer_scheduler)

            for i in range(num_samples):
                fid = frame_ids[i]
                fid_disp = int(frame_ids_rounded[i].item())

                pe = base_pe.clone()   # [1,T,D]
                pam = base_pam.clone() # [1,T]

                # normalized idx: / max_frame_idx
                frame_norm = (fid / float(self.max_frame_idx)).view(1, 1)  # [1,1]

                pe, pam = self._append_frame_token(pe, pam, frame_norm)    # append frame token

                # fresh scheduler per frame value
                infer_scheduler = deepcopy(self.infer_scheduler)
                infer_scheduler.set_timesteps(self.num_sample_steps, device=device)
                timesteps = infer_scheduler.timesteps

                # same starting noise for all frames
                latents = base_noise.clone()

                for t in timesteps:
                    t_batch = t.repeat(1).to(device)  # [1]
                    t_batch = t_batch.to(dtype)

                    model_pred = self.transformer(
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

                img_np = self._decode_latents(latents)[0].numpy()
                out_path = step_dir / f"sample_frame{fid_disp:03d}.png"
                Image.fromarray(img_np).save(out_path)

                if isinstance(self.logger, WandbLogger):
                    try:
                        import wandb
                        wandb_images.append(
                            wandb.Image(
                                img_np,
                                caption=f"{self._ex_text or 'one-class'} | frame_idx={fid_disp}",
                            )
                        )
                    except Exception:
                        pass

        # ----------------------------------------------------
        # Case 2: no frame token -> vectorized batch sampling
        # ----------------------------------------------------
        else:
            B = num_samples
            pe = self.base_pe.to(device=device, dtype=dtype).expand(B, -1, -1)  # [B,T,D]
            pam = self.base_pam.to(device=device).expand(B, -1)                 # [B,T]

            infer_scheduler = deepcopy(self.infer_scheduler)
            infer_scheduler.set_timesteps(self.num_sample_steps, device=device)
            timesteps = infer_scheduler.timesteps

            latents = torch.randn((B, c, h, w), device=device, dtype=dtype)

            for t in timesteps:
                t_batch = t.repeat(B).to(device)
                t_batch = t_batch.to(dtype)

                model_pred = self.transformer(
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

            imgs = self._decode_latents(latents)  # [B,H,W,3]

            for i in range(B):
                img_np = imgs[i].numpy()
                out_path = step_dir / f"sample_{i:02d}.png"
                Image.fromarray(img_np).save(out_path)

                if isinstance(self.logger, WandbLogger):
                    try:
                        import wandb
                        wandb_images.append(
                            wandb.Image(
                                img_np,
                                caption=f"{self._ex_text or 'one-class'} | sample {i}",
                            )
                        )
                    except Exception:
                        pass

        # Log to W&B
        if wandb_images and isinstance(self.logger, WandbLogger):
            try:
                import wandb
                self.logger.experiment.log(
                    {
                        "samples/generated": wandb_images,
                        "samples/prompt": self._ex_text,
                        "global_step": int(self.global_step),
                    },
                    step=int(self.global_step),
                )
            except Exception:
                pass

    # --------------------------
    # cache example + ref image
    # --------------------------

    def _cache_one_sample(self, batch, x0):
        """
        Cache latent shape & reference image for logging.
        """
        if self._ex_latent_shape is None:
            self._ex_latent_shape = x0[0:1].shape

        if self._ex_text is None:
            txt = batch.get("processed_text_condition", None)
            if isinstance(txt, (list, tuple)) and len(txt) > 0:
                self._ex_text = str(txt[0])
            elif isinstance(txt, torch.Tensor) and txt.numel() > 0:
                self._ex_text = str(txt[0])
            elif isinstance(txt, str):
                self._ex_text = txt
            else:
                self._ex_text = self.instance_prompt or "one-class"

        if (not self._logged_ref_image) and isinstance(self.logger, WandbLogger):
            try:
                import wandb
                ref_latent = x0[0:1].detach()
                img_np = self._decode_latents(ref_latent)[0].numpy()

                self.logger.experiment.log(
                    {
                        "reference/original_target": wandb.Image(
                            img_np,
                            caption=f"Reference x0 | {self._ex_text}",
                        )
                    },
                    step=int(self.global_step),
                )
                self._logged_ref_image = True
            except Exception:
                pass

    # --------------------------
    # Lightning: training / val
    # --------------------------

    def training_step(self, batch, batch_idx):
        x0 = batch["latents"].to(self.device)  # [B,C,H,W]

        B = x0.shape[0]
        pe = self.base_pe.to(self.device, dtype=self.transformer.dtype).expand(B, -1, -1)
        pam = self.base_pam.to(self.device).expand(B, -1)  # [B,T]

        # Optional frame token conditioning
        if self.use_frame_token and self.frame_proj is not None:
            if "frame_idx" not in batch:
                raise KeyError("Batch missing 'frame_idx' required for frame token")
            frame_idx = batch["frame_idx"].to(self.device, dtype=torch.float32)  # [B]
            if frame_idx.dim() == 0:
                frame_idx = frame_idx.unsqueeze(0)
            frame_norm = (frame_idx / float(self.max_frame_idx)).view(B, 1)      # [B,1]
            pe, pam = self._append_frame_token(pe, pam, frame_norm)

        # cache example & reference image
        self._cache_one_sample(batch, x0)

        loss = self._forward_flow_matching(x0, pe, pam)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x0 = batch["latents"].to(self.device)
        B = x0.shape[0]
        pe = self.base_pe.to(self.device, dtype=self.transformer.dtype).expand(B, -1, -1)
        pam = self.base_pam.to(self.device).expand(B, -1)

        if self.use_frame_token and self.frame_proj is not None:
            if "frame_idx" not in batch:
                raise KeyError("Batch missing 'frame_idx' required for frame token")
            frame_idx = batch["frame_idx"].to(self.device, dtype=torch.float32)
            if frame_idx.dim() == 0:
                frame_idx = frame_idx.unsqueeze(0)
            frame_norm = (frame_idx / float(self.max_frame_idx)).view(B, 1)
            pe, pam = self._append_frame_token(pe, pam, frame_norm)

        val_loss = self._forward_flow_matching(x0, pe, pam)
        self.log("val/loss", val_loss, prog_bar=True, on_step=True)
        return val_loss

    # --------------------------
    # LoRA warmup / checkpointing
    # --------------------------

    def _maybe_unfreeze_lora(self):
        if not self.do_lora or self._lora_unfrozen:
            return
        if self.global_step >= self.lora_freeze_steps:
            for p in self.lora_params:
                p.requires_grad_(True)
            self._lora_unfrozen = True
            if isinstance(self.logger, WandbLogger):
                try:
                    self.logger.experiment.log(
                        {"control/lora_unfrozen_at": int(self.global_step)},
                        step=int(self.global_step),
                    )
                except Exception:
                    pass

    def _maybe_save_ckpt(self):
        if self.save_pt_every_steps <= 0:
            return
        if self.global_step == 0 or (self.global_step % self.save_pt_every_steps) != 0:
            return

        save_dir = self.ckpt_root / f"step_{self.global_step:07d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.do_lora:
            try:
                SanaPipeline.save_lora_weights(
                    save_directory=str(save_dir / "lora"),
                    transformer_lora_layers=get_peft_model_state_dict(self.transformer),
                )
            except Exception:
                pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._maybe_unfreeze_lora()

        if self.save_every_steps > 0 and self.global_step > 0:
            if (self.global_step % self.save_every_steps) == 0:
                if getattr(self.trainer, "global_rank", 0) == 0:
                    self._sample_images(num_samples=4)

        if getattr(self.trainer, "global_rank", 0) == 0:
            self._maybe_save_ckpt()

    # --------------------------
    # grad norm logging
    # --------------------------

    def on_after_backward(self):
        total_norm_sq = 0.0
        param_count = 0

        train_params = []
        if self.use_frame_token and self.frame_proj is not None:
            train_params += list(self.frame_proj.parameters())
        if self.do_lora:
            train_params += self.lora_params

        for p in train_params:
            if p.grad is None:
                continue
            param_count += 1
            total_norm_sq += p.grad.detach().data.norm(2).item() ** 2

        if param_count > 0:
            total_norm = math.sqrt(total_norm_sq)
            self.log("grad/total_norm", total_norm, prog_bar=False, on_step=True, on_epoch=False)

    # --------------------------
    # optim
    # --------------------------

    def configure_optimizers(self):
        params = []
        if self.use_frame_token and self.frame_proj is not None:
            params += list(self.frame_proj.parameters())
        if self.do_lora:
            params += self.lora_params
        return torch.optim.AdamW(params, lr=self.lr)

    # --------------------------
    # final save
    # --------------------------

    def on_fit_end(self):
        out_dir = self.ckpt_root / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.do_lora:
            SanaPipeline.save_lora_weights(
                save_directory=str(out_dir / "lora"),
                transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            )