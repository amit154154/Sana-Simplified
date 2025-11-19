import copy
from pathlib import Path
import math
import gc
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
from tqdm import tqdm
from copy import deepcopy
import lovely_tensors as lt

lt.monkey_patch()


def _slug(s: str, max_len: int = 64) -> str:
    s = "".join(c if c.isalnum() else "_" for c in str(s))
    while "__" in s:
        s = s.replace("__", "_")
    return s[:max_len].strip("_") or "sample"


class SanaZoomLoRA(pl.LightningModule):
    """
    Sana 1.5 / Sprint-compatible zoom fine-tuning with the *official* flow-matching +
    FlowMatchEulerDiscreteScheduler, and:

      - Zoom conditioning as an APPENDED TOKEN (org/target/delta → zoom token)
      - Optional SigLIP token (IP-adapter-like), also appended
      - No text encoder: uses precomputed text embeddings from dataset
      - One-class setup (single instance prompt)

    Dataset batch keys expected:
      latent_target_zoom: [B, C, H, W]  (VAE latent x0 = data)
      zoom_org:           [B, 1]        (scaled, e.g. /5)
      zoom_target:        [B, 1]
      text_encoding_pe:   [B, 1, T, D] or [B, T, D]
      text_encoding_pam:  [B, 1, T] or [B, T]
      (optional) siglip_org_embed: [B, D_siglip]
      processed_text_condition: list[str] (for logging only)
    """

    def __init__(
            self,
            transformer: SanaTransformer2DModel,
            vae: AutoencoderDC,
            scheduler,  # FlowMatchEulerDiscreteScheduler
            lr: float = 1e-4,
            do_lora: bool = True,
            example_dir: Path = Path("samples"),
            save_examples_every_steps: int = 1000,
            save_pt_every_steps: int = 1000,
            lora_freeze_steps: int = 0,
            siglip_dim: Optional[int] = None,
            instance_prompt: Optional[str] = None,
            num_sample_steps: int = 30,
            # flow-matching weighting config (matches Sana/SD3 trainer style)
            weighting_scheme: str = "none",  # "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"
            logit_mean: float = 0.0,
            logit_std: float = 1.0,
            mode_scale: float = 1.29,
            ckpt_root: Path = Path("zoom_lora"),

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

        # -----------------------
        # flow-matching schedulers
        # -----------------------
        # One copy for training (never call set_timesteps)
        # One copy for inference (we call set_timesteps here)
        self.train_scheduler = copy.deepcopy(scheduler)
        self.infer_scheduler = copy.deepcopy(scheduler)

        self.weighting_scheme = weighting_scheme
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.mode_scale = float(mode_scale)

        # zoom conditioning: [zoom_org, zoom_target, delta] → zoom TOKEN in text space
        text_hidden_dim = 2304  # Sana 1.5 text dimension
        self.zoom_proj = nn.Linear(3, text_hidden_dim)
        nn.init.zeros_(self.zoom_proj.weight)
        nn.init.zeros_(self.zoom_proj.bias)

        # optional SigLIP projection → text dimension (also appended as a token)
        self.siglip_proj = None
        if siglip_dim is not None:
            self.siglip_proj = nn.Linear(siglip_dim, text_hidden_dim)
            nn.init.zeros_(self.siglip_proj.weight)
            nn.init.zeros_(self.siglip_proj.bias)

        # collect LoRA params if needed (we assume adapters already added)
        self.lora_params = []
        if self.do_lora:
            # Freeze everything by default, then re-enable only LoRA params
            for name, p in self.transformer.named_parameters():
                if "lora" in name.lower():  # LoRA weights
                    p.requires_grad_(True)
                    self.lora_params.append(p)
                else:
                    p.requires_grad_(False)
        else:
            # No LoRA: fully freeze transformer
            for _, p in self.transformer.named_parameters():
                p.requires_grad_(False)

        # Optional: LoRA warmup (keep them frozen for first N steps)
        if self.do_lora and self.lora_freeze_steps > 0:
            for p in self.lora_params:
                p.requires_grad_(False)

        # VAE frozen
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        # example cache for periodic sampling
        self._ex_text = None
        self._ex_pe0 = None  # base text embeddings [1,T,D]
        self._ex_pam0 = None  # base attention mask  [1,T]
        self._ex_zo = None
        self._ex_zt = None
        self._ex_latent_shape = None
        self._ex_siglip_org = None

        # flag: log decoded reference x0 image only once
        self._logged_ref_image = False

        self.save_hyperparameters(
            ignore=["transformer", "vae", "train_scheduler", "infer_scheduler"]
        )

    # ---------------------------
    # helpers: zoom token + SigLIP token
    # ---------------------------

    def _append_zoom_token(
            self,
            pe: torch.Tensor,
            pam: torch.Tensor,
            zo: torch.Tensor,
            zt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append a *zoom token* at the end of the text sequence.

          pe:  [B, T, D]
          pam: [B, T]
          zo, zt: [B, 1] (already normalized in dataset, e.g. /5)
        """
        dz = zt - zo  # [B,1]
        zcat = torch.cat([zo, zt, dz], dim=1)  # [B,3]
        zoom_tok = self.zoom_proj(zcat)  # [B,D_text]
        zoom_tok = zoom_tok.unsqueeze(1)  # [B,1,D_text]

        pe = torch.cat([pe, zoom_tok], dim=1)  # [B,T+1,D]
        pam = F.pad(pam, (0, 1), value=True)  # [B,T+1]
        return pe, pam

    def _apply_siglip_token(
            self,
            pe: torch.Tensor,
            pam: torch.Tensor,
            batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append SigLIP token to text sequence if siglip_proj and siglip_org_embed are provided.
          pe:  [B, T, D]
          pam: [B, T]
        """
        if self.siglip_proj is None:
            return pe, pam

        siglip = batch.get("siglip_org_embed", None)
        if siglip is None:
            return pe, pam

        if not isinstance(siglip, torch.Tensor):
            siglip = torch.as_tensor(siglip)

        # 🔑 Make sure siglip matches the projector's weight dtype & device
        proj_weight = self.siglip_proj.weight
        siglip = siglip.to(device=proj_weight.device, dtype=proj_weight.dtype)  # [B, D_siglip]

        tok = self.siglip_proj(siglip)  # [B, D_text]
        tok = tok.unsqueeze(1)  # [B, 1, D_text]

        pe = torch.cat([pe, tok], dim=1)  # [B, T+1, D]
        pam = F.pad(pam, (0, 1), value=True)  # [B, T+1]
        return pe, pam

    # --------------------------
    # scheduler helpers (official Sana style)
    # --------------------------

    def _get_sigmas(self, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Same sigma lookup trick as the official Sana/SD3 trainer, but using the *training* scheduler.
        """
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        # find indices in scheduler.timesteps that match each sampled t
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # --------------------------
    # core flow-matching (official version)
    # --------------------------

    def _forward_flow_matching(
            self,
            x0: torch.Tensor,  # data latents (target zoom) [B,C,H,W]
            pe: torch.Tensor,  # text embeddings (with zoom/siglip tokens) [B,T,D]
            pam: torch.Tensor,  # attention mask [B,T]
    ) -> torch.Tensor:
        """
        One flow-matching step with:

          - x0: data latents (from VAE, already scaled)
          - noise ~ N(0, I)
          - timesteps from density scheduler
          - z_t = (1 - sigma) * x0 + sigma * noise
          - model_pred = transformer(z_t, cond, t)
          - target = noise - x0
          - weighting = compute_loss_weighting_for_sd3(...)
        """
        B = x0.shape[0]
        device = self.device

        x0 = x0.to(device=device, dtype=self.transformer.dtype)  # [B,C,H,W]
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

        # sigmas for these timesteps
        sigmas = self._get_sigmas(timesteps, n_dim=x0.ndim, dtype=x0.dtype)  # broadcastable

        # z_t = (1 - sigma)*x0 + sigma*noise
        z_t = (1.0 - sigmas) * x0 + sigmas * noise

        # forward transformer
        pe_in = pe.to(self.transformer.dtype)
        model_pred = self.transformer(
            hidden_states=z_t,
            encoder_hidden_states=pe_in,
            encoder_attention_mask=pam,
            timestep=timesteps.to(self.transformer.dtype),
            return_dict=False,
        )[0]  # [B,C,H,W]-ish for Sana

        model_pred = model_pred.float()
        x0_f = x0.float()
        noise_f = noise.float()

        # loss weighting
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas,
        ).float()

        target = (noise_f - x0_f)  # [B,C,H,W]

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
        latents: [B,C,H,W] in VAE latent scale (i.e., multiplied by scaling_factor)
        returns [B,H,W,3] uint8 on CPU
        """
        vae = self.vae
        sf = float(vae.config.scaling_factor)

        lat = latents.to(self.device, dtype=vae.dtype) / sf
        out = vae.decode(lat, return_dict=False)[0]  # [-1,1]
        img = (out.clamp(-1, 1) + 1.0) * 0.5  # [0,1]
        img = (img.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1))  # [B,H,W,3]
        return img.cpu()

    # --------------------------
    # sampling for logging (scheduler-based)
    # --------------------------

    @torch.no_grad()
    def _sample_zoom_panel(self):
        """
        Sampling with the *official* FlowMatchEulerDiscreteScheduler:
          - For each zoom_target:
            * clone the inference scheduler (fresh _step_index)
            * set timesteps
            * run a standard FM sampling loop
          - Save one image per zoom_target and log to W&B.
        """
        if any(x is None for x in [self._ex_pe0, self._ex_pam0, self._ex_zo, self._ex_latent_shape]):
            return

        self.example_dir.mkdir(parents=True, exist_ok=True)
        step_dir = self.example_dir / f"step_{self.global_step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        device = self.device
        dtype = self.transformer.dtype

        base_pe = self._ex_pe0.to(device)  # [1,T,D]
        base_pam = self._ex_pam0.to(device)  # [1,T]
        zo = torch.tensor([[self._ex_zo]], device=device, dtype=torch.float32)  # [1,1]

        # zooms we want to visualize (still in normalized space 0..1)
        zoom_targets = torch.linspace(0.0, 1.0, steps=5, device=device, dtype=torch.float32)
        prompt_slug = _slug(self._ex_text or "prompt")

        from PIL import Image
        wandb_images = []
        B_panel = 1  # one sample per zoom

        for zt_val in zoom_targets.tolist():
            zt = torch.tensor([[zt_val]], device=device, dtype=torch.float32)  # [1,1]

            pe = base_pe.clone()  # [1,T,D]
            pam = base_pam.clone()  # [1,T]

            # append zoom token
            pe, pam = self._append_zoom_token(pe, pam, zo, zt)

            # append SigLIP token if cached and projector exists
            if self.siglip_proj is not None and self._ex_siglip_org is not None:
                pe, pam = self._apply_siglip_token(
                    pe,
                    pam,
                    {"siglip_org_embed": self._ex_siglip_org},
                )

            # 🔁 fresh scheduler per zoom value (avoids _step_index issues)
            infer_scheduler = deepcopy(self.infer_scheduler)
            infer_scheduler.set_timesteps(self.num_sample_steps, device=device)
            timesteps = infer_scheduler.timesteps  # [num_steps]

            # latents ~ N(0, I) in Sana latent space
            _, c, h, w = self._ex_latent_shape
            latents = torch.randn((B_panel, c, h, w), device=device, dtype=dtype)

            # main flow-matching sampling loop
            for t in timesteps:
                t_batch = t.repeat(B_panel).to(device)  # [B]
                t_batch = t_batch.to(dtype)

                model_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=pe.to(dtype),
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
            out_path = step_dir / f"{prompt_slug}_zt{zt_val:.2f}.png"
            Image.fromarray(img_np).save(out_path)

            if isinstance(self.logger, WandbLogger):
                try:
                    import wandb
                    wandb_images.append(
                        wandb.Image(
                            img_np,
                            caption=f"{self._ex_text} | zo={self._ex_zo:.2f} → zt={zt_val:.2f}",
                        )
                    )
                except Exception:
                    pass

        # log panel to W&B
        if wandb_images and isinstance(self.logger, WandbLogger):
            try:
                self.logger.experiment.log(
                    {
                        "samples/zoom_panel": wandb_images,
                        "samples/prompt": self._ex_text,
                        "global_step": int(self.global_step),
                    },
                    step=int(self.global_step),
                )
            except Exception:
                pass

    # --------------------------
    # cache a reference example + log ref image once
    # --------------------------

    def _cache_one_sample(self, batch, pe, pam, zo, zt, x0):
        """
        Cache *base* pe/pam (no zoom/SigLIP tokens) + shape + example zooms,
        to reuse later in _sample_zoom_panel.

        Also: decode & log the reference x0 image ONCE to W&B.
        """
        if self._ex_text is None:
            txt = batch.get("processed_text_condition", None)
            if isinstance(txt, (list, tuple)) and len(txt) > 0:
                self._ex_text = str(txt[0])
            elif isinstance(txt, torch.Tensor) and txt.numel() > 0:
                self._ex_text = str(txt[0])
            elif isinstance(txt, str):
                self._ex_text = txt
            else:
                self._ex_text = self.instance_prompt or "prompt"

        if self._ex_pe0 is None:
            self._ex_pe0 = pe[0:1].detach().cpu()
        if self._ex_pam0 is None:
            self._ex_pam0 = pam[0:1].detach().cpu()
        if self._ex_latent_shape is None:
            self._ex_latent_shape = x0[0:1].shape
        if self._ex_zo is None:
            self._ex_zo = zo[0, 0].item()
        if self._ex_zt is None:
            self._ex_zt = zt[0, 0].item()

        # cache SigLIP (optional)
        if (
                self._ex_siglip_org is None
                and "siglip_org_embed" in batch
                and batch["siglip_org_embed"] is not None
        ):
            siglip = batch["siglip_org_embed"]
            if not isinstance(siglip, torch.Tensor):
                siglip = torch.as_tensor(siglip)
            self._ex_siglip_org = siglip[0:1].detach().cpu()

        # ---- log decoded reference image ONCE ----
        if (not self._logged_ref_image) and isinstance(self.logger, WandbLogger):
            try:
                import wandb
                ref_latent = x0[0:1].detach()
                img_np = self._decode_latents(ref_latent)[0].numpy()

                self.logger.experiment.log(
                    {
                        "reference/original_target": wandb.Image(
                            img_np,
                            caption=f"Reference x0 | zoom={self._ex_zo:.2f}",
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
        # data latents for the *target* zoom (Sana-style x0)
        x0 = batch["latent_target_zoom"].to(self.device)  # [B,C,H,W]

        # text embeddings from dataset
        pe = batch["text_encoding_pe"].to(self.device)
        if pe.dim() == 4:
            pe = pe.squeeze(1)  # [B,T,D]
        pam = batch["text_encoding_pam"]
        if pam is None:
            pam = torch.ones(pe.shape[:2], dtype=torch.bool, device=self.device)
        else:
            pam = pam.to(self.device)
            if pam.dim() == 3:
                pam = pam.squeeze(1)  # [B,T]
            pam = pam.to(torch.bool)

        zo = batch["zoom_org"].to(self.device, dtype=torch.float32)  # [B,1]
        zt = batch["zoom_target"].to(self.device, dtype=torch.float32)  # [B,1]

        # cache base (no extra tokens) + maybe log ref image
        self._cache_one_sample(batch, pe, pam, zo, zt, batch["latent_org_zoom"].to(self.device))

        # append zoom token and optional SigLIP token

        pe, pam = self._append_zoom_token(pe, pam, zo, zt)
        pe, pam = self._apply_siglip_token(pe, pam, batch)

        loss = self._forward_flow_matching(x0, pe, pam)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x0 = batch["latent_target_zoom"].to(self.device)

        pe = batch["text_encoding_pe"].to(self.device)
        if pe.dim() == 4:
            pe = pe.squeeze(1)
        pam = batch["text_encoding_pam"]
        if pam is None:
            pam = torch.ones(pe.shape[:2], dtype=torch.bool, device=self.device)
        else:
            pam = pam.to(self.device)
            if pam.dim() == 3:
                pam = pam.squeeze(1)
            pam = pam.to(torch.bool)

        zo = batch["zoom_org"].to(self.device, dtype=torch.float32)
        zt = batch["zoom_target"].to(self.device, dtype=torch.float32)

        pe, pam = self._append_zoom_token(pe, pam, zo, zt)
        pe, pam = self._apply_siglip_token(pe, pam, batch)

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

        torch.save(self.zoom_proj.state_dict(), save_dir / "zoom_proj.pt")
        if self.siglip_proj is not None:
            torch.save(self.siglip_proj.state_dict(), save_dir / "siglip_proj.pt")

        if self.do_lora:
            try:
                SanaPipeline.save_lora_weights(
                    save_directory=str(save_dir / "lora"),
                    transformer_lora_layers=get_peft_model_state_dict(self.transformer),
                )
            except Exception:
                pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # LoRA warmup
        self._maybe_unfreeze_lora()

        # periodic sample logging
        if self.save_every_steps > 0 and self.global_step > 0:
            if (self.global_step % self.save_every_steps) == 0:
                if getattr(self.trainer, "global_rank", 0) == 0:
                    self._sample_zoom_panel()

        # periodic ckpt
        if getattr(self.trainer, "global_rank", 0) == 0:
            self._maybe_save_ckpt()

    # --------------------------
    # grad norm logging
    # --------------------------

    def on_after_backward(self):
        """
        Log gradient norms after each backward pass.
        """
        total_norm_sq = 0.0
        param_count = 0

        for p in list(self.zoom_proj.parameters()) + (
                list(self.siglip_proj.parameters()) if self.siglip_proj is not None else []
        ) + (self.lora_params if self.do_lora else []):
            if p.grad is None:
                continue
            param_count += 1
            total_norm_sq += p.grad.detach().data.norm(2).item() ** 2

        if param_count > 0:
            total_norm = math.sqrt(total_norm_sq)
            self.log("grad/total_norm", total_norm, prog_bar=False, on_step=True, on_epoch=False)

            # optional histogram of grad magnitudes
            if isinstance(self.logger, WandbLogger):
                try:
                    import wandb
                    grads = []
                    for p in list(self.zoom_proj.parameters()) + (
                            list(self.siglip_proj.parameters()) if self.siglip_proj is not None else []
                    ) + (self.lora_params if self.do_lora else []):
                        if p.grad is not None:
                            grads.append(p.grad.detach().view(-1).cpu())
                    if grads:
                        all_grads = torch.cat(grads, dim=0)
                        self.logger.experiment.log(
                            {
                                "grad/hist": wandb.Histogram(all_grads.numpy()),
                            },
                            step=self.global_step,
                        )
                except Exception:
                    pass

    # --------------------------
    # optim
    # --------------------------

    def configure_optimizers(self):
        params = list(self.zoom_proj.parameters())
        if self.siglip_proj is not None:
            params += list(self.siglip_proj.parameters())
        if self.do_lora:
            params += self.lora_params
        return torch.optim.AdamW(params, lr=self.lr)

    # --------------------------
    # final save
    # --------------------------

    def on_fit_end(self):
        out_dir = self.ckpt_root / "final"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.zoom_proj.state_dict(), out_dir / "zoom_proj.pt")
        if self.siglip_proj is not None:
            torch.save(self.siglip_proj.state_dict(), out_dir / "siglip_proj.pt")
        if self.do_lora:
            SanaPipeline.save_lora_weights(
                save_directory=str(out_dir / "lora"),
                transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            )