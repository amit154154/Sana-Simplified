import copy
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

from diffusers import AutoencoderDC, SanaTransformer2DModel
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from pytorch_lightning.loggers import WandbLogger


class SanaDreamBoothToken(pl.LightningModule):
    """
    Token-only DreamBooth/Textual-Inversion for Sana 1.5 using
    Flow-Matching training logic.

    - Transformer is fully frozen (no LoRA).
    - We precompute a base text embedding (pe_base, pam_base) once in __init__
      for a prompt like: "a <sks> dog".
    - We learn a single "concept token" vector that is added at token_index.

    Training (flow-matching, same math as official trainer):
        x0 = VAE latents (already scaled) from dataset
        noise ~ N(0, I)
        z_t = (1 - sigma) * x0 + sigma * noise
        transformer(z_t, pe_with_delta, pam, t) -> model_pred
        target = noise - x0
        loss = E[ w(sigma) * ||model_pred - target||^2 ]

    Sampling:
        - use infer_scheduler in a standard loop
        - decode through VAE
        - save PNGs every `sample_every_n_steps`

    Word-generation logging:
        - uses a second text embedding (pe_word, pam_word) for a prompt like
          "a white poster with a single unknown word written..."
        - runs a full sampling loop with those embeddings every
          `log_word_every_n_steps` steps
        - logs to disk + W&B
    """

    def __init__(
        self,
        transformer: SanaTransformer2DModel,
        vae: AutoencoderDC,
        scheduler,                      # FlowMatchEulerDiscreteScheduler
        pe_base: torch.Tensor,
        pam_base: torch.Tensor,
        token_index: int = 0,           # which token position to "bind" the concept to
        lr: float = 2e-4,
        guidance_scale: float = 1.0,    # unused for Sana 1.5 forward, kept for future use
        weighting_scheme: str = "none", # "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        # --- optimizer config ---
        use_8bit_adam: bool = True,
        # --- sampling config (generic samples) ---
        sample_every_n_steps: int = 0,       # 0 => disable
        num_sample_images: int = 4,
        sample_num_inference_steps: int = 30,
        sample_dir: str | Path = "samples",
        sample_seed: int | None = 42,
        # --- word-generation logging config ---
        pe_word: torch.Tensor | None = None,
        pam_word: torch.Tensor | None = None,
        log_word_every_n_steps: int = 0,         # 0 => disable
        log_word_dir: str | Path = "word_samples",
        log_word_prompt: str | None = None,
        # --- NEW: periodic token embedding saving ---
        save_token_every_n_steps: int = 0,       # 0 => disable
        token_save_dir: str | Path = "token_snapshots",
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae

        # ------------- freeze transformer & vae -------------
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.lr = float(lr)
        self.guidance_scale = float(guidance_scale)

        # ------------- schedulers & training config -------------
        self.train_scheduler = copy.deepcopy(scheduler)
        self.infer_scheduler = copy.deepcopy(scheduler)

        self.weighting_scheme = weighting_scheme
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.mode_scale = float(mode_scale)

        # optimizer config
        self.use_8bit_adam = bool(use_8bit_adam)

        # --- sampling config (generic) ---
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.num_sample_images = int(num_sample_images)
        self.sample_num_inference_steps = int(sample_num_inference_steps)
        self.sample_dir = Path(sample_dir)
        self.sample_seed = sample_seed

        # --- word-generation logging config ---
        self.log_word_every_n_steps = int(log_word_every_n_steps)
        self.log_word_dir = Path(log_word_dir)
        self.log_word_prompt = log_word_prompt or ""

        # --- NEW: token embedding save config ---
        self.save_token_every_n_steps = int(save_token_every_n_steps)
        self.token_save_dir = Path(token_save_dir)

        # latent shape will be inferred from first batch
        self.latent_shape = None  # (C, H, W)

        # ------------- register base pe/pam (precomputed) -------------
        if pe_base.dim() == 2:   # [T, D]
            pe_base = pe_base.unsqueeze(0)  # [1, T, D]
        if pam_base.dim() == 1:  # [T]
            pam_base = pam_base.unsqueeze(0)  # [1, T]

        self.register_buffer("pe_base", pe_base)                   # [1, T, D]
        self.register_buffer("pam_base", pam_base.to(torch.bool))  # [1, T]

        self.token_index = int(token_index)
        T = self.pe_base.shape[1]
        assert 0 <= self.token_index < T, f"token_index={self.token_index} out of range T={T}"

        # ------------- optional word-pe/pam (for word logging) -------------
        if pe_word is not None:
            if pe_word.dim() == 2:
                pe_word = pe_word.unsqueeze(0)
            self.register_buffer("pe_word", pe_word)  # [1, T, D]
        else:
            self.pe_word = None

        if pam_word is not None:
            if pam_word.dim() == 1:
                pam_word = pam_word.unsqueeze(0)
            self.register_buffer("pam_word", pam_word.to(torch.bool))  # [1, T]
        else:
            self.pam_word = None

        # ------------- learnable "concept token" vector -------------
        D = self.pe_base.shape[-1]
        # still zero-init: residual on top of base token
        self.concept_delta = nn.Parameter(torch.zeros(1, 1, D))

    # ------------- helpers -------------

    def _make_pe_pam(self, batch_size: int):
        pe = self.pe_base.to(self.device, dtype=self.transformer.dtype)   # [1, T, D]
        pam = self.pam_base.to(self.device)                               # [1, T]

        pe = pe.expand(batch_size, -1, -1).clone()  # [B, T, D]
        pam = pam.expand(batch_size, -1)            # [B, T]

        idx = self.token_index
        pe[:, idx:idx+1, :] = pe[:, idx:idx+1, :] + self.concept_delta.to(pe.dtype)

        return pe, pam

    def _make_pe_pam_word(self, batch_size: int):
        if (self.pe_word is None) or (self.pam_word is None):
            return self._make_pe_pam(batch_size)

        pe = self.pe_word.to(self.device, dtype=self.transformer.dtype)   # [1, T, D]
        pam = self.pam_word.to(self.device)                               # [1, T]

        pe = pe.expand(batch_size, -1, -1).clone()  # [B, T, D]
        pam = pam.expand(batch_size, -1)            # [B, T]

        idx = self.token_index
        pe[:, idx:idx+1, :] = pe[:, idx:idx+1, :] + self.concept_delta.to(pe.dtype)

        return pe, pam

    def _get_sigmas(self, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ------------- core training logic (flow-matching) -------------

    def _forward_flow_matching(self, x0: torch.Tensor):
        B = x0.shape[0]
        device = self.device

        noise = torch.randn_like(x0)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=B,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        ).to(device)

        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        indices_cpu = indices.to(self.train_scheduler.timesteps.device)
        timesteps = self.train_scheduler.timesteps[indices_cpu].to(device=device)

        sigmas = self._get_sigmas(timesteps, n_dim=x0.ndim, dtype=x0.dtype)

        z_t = (1.0 - sigmas) * x0 + sigmas * noise

        pe, pam = self._make_pe_pam(B)

        model_pred = self.transformer(
            hidden_states=z_t.to(self.transformer.dtype),
            encoder_hidden_states=pe.to(self.transformer.dtype),
            encoder_attention_mask=pam,
            timestep=timestamps.to(self.transformer.dtype) if (timestamps := timesteps) is not None else None,
            return_dict=False,
        )[0]

        model_pred = model_pred.float()

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas,
        ).float()

        target = (noise - x0).float()

        loss = torch.mean(
            (weighting * (model_pred - target) ** 2).reshape(B, -1),
            dim=1,
        ).mean()

        return loss

    # ------------- logging helpers -------------

    def _log_concept_delta_stats(self):
        delta = self.concept_delta.detach()
        flat = delta.view(-1)

        self.log("concept_delta/norm", flat.norm().item(), prog_bar=False, on_step=True, on_epoch=False)
        self.log("concept_delta/mean_abs", flat.abs().mean().item(), prog_bar=False, on_step=True, on_epoch=False)
        self.log("concept_delta/max_abs", flat.abs().max().item(), prog_bar=False, on_step=True, on_epoch=False)

        if isinstance(self.logger, WandbLogger):
            try:
                import wandb
                self.logger.experiment.log(
                    {
                        "concept_delta/hist": wandb.Histogram(flat.detach().cpu().numpy()),
                    },
                    step=self.global_step,
                )
            except Exception as e:
                print(f"[wandb] concept_delta histogram logging failed: {e}")

    # --- NEW: save token embedding to disk ---

    @torch.no_grad()
    def _save_token_embedding(self):
        """
        Save:
          - base token embedding at token_index
          - learned delta
          - total = base + delta

        to a .pt file:
            token_step_{global_step:06d}.pt
        """
        if self.save_token_every_n_steps <= 0:
            return

        self.token_save_dir.mkdir(parents=True, exist_ok=True)

        step = int(self.global_step)

        base_vec = self.pe_base[0, self.token_index].detach().cpu()         # [D]
        delta_vec = self.concept_delta[0, 0].detach().cpu()                 # [D]
        total_vec = (base_vec + delta_vec).detach().cpu()                   # [D]

        payload = {
            "step": step,
            "token_index": int(self.token_index),
            "base": base_vec,      # [D]
            "delta": delta_vec,    # [D]
            "total": total_vec,    # [D] = base + delta
        }

        out_path = self.token_save_dir / f"token_step_{step:06d}.pt"
        torch.save(payload, out_path)

        # optional scalar log
        self.log("token_save/last_step", float(step), prog_bar=False)

        # optional W&B text log of path
        if isinstance(self.logger, WandbLogger):
            try:
                self.logger.experiment.log(
                    {"token_save/path": str(out_path)},
                    step=step,
                )
            except Exception as e:
                print(f"[wandb] token embedding logging failed: {e}")

    # ------------- training / validation -------------

    def training_step(self, batch, batch_idx):
        x0 = batch["latents"].to(self.device, dtype=self.transformer.dtype)

        if self.latent_shape is None:
            self.latent_shape = x0.shape[1:]

        loss = self._forward_flow_matching(x0)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self._log_concept_delta_stats()

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x0 = batch["latents"].to(self.device, dtype=self.transformer.dtype)
        loss = self._forward_flow_matching(x0)
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ------------- generic sampling & saving -------------

    @torch.no_grad()
    def _sample_and_save(self):
        if self.sample_every_n_steps <= 0:
            return
        if self.latent_shape is None:
            return

        self.sample_dir.mkdir(parents=True, exist_ok=True)

        B = self.num_sample_images
        C, H, W = self.latent_shape

        if self.sample_seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(self.sample_seed)
            latents = torch.randn(
                (B, C, H, W),
                generator=gen,
                device=self.device,
                dtype=self.transformer.dtype,
            )
        else:
            latents = torch.randn(
                (B, C, H, W),
                device=self.device,
                dtype=self.transformer.dtype,
            )

        infer_scheduler = self.infer_scheduler
        infer_scheduler.set_timesteps(self.sample_num_inference_steps, device=self.device)
        timesteps = infer_scheduler.timesteps

        for t in tqdm(timesteps, desc="Sampling (generic)", leave=False):
            pe, pam = self._make_pe_pam(B)

            t_batch = t.repeat(B).to(self.device)
            t_batch = t_batch.to(self.transformer.dtype)

            model_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=pe.to(self.transformer.dtype),
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

        sf = float(getattr(self.vae.config, "scaling_factor", 1.0))
        latents_vae = (latents / sf).to(self.vae.dtype)
        decoded = self.vae.decode(latents_vae).sample  # [-1, 1]
        images = (decoded.clamp(-1, 1) + 1.0) / 2.0

        step = self.global_step
        wandb_images = []

        for i in range(B):
            img = images[i].detach().cpu()
            pil = to_pil_image(img)
            out_path = self.sample_dir / f"step_{step:06d}_sample_{i}.png"
            pil.save(out_path)

            wandb_images.append((pil, out_path))

        self.log("samples/last_step", float(step), prog_bar=False)

        if isinstance(self.logger, WandbLogger):
            try:
                import wandb
                self.logger.experiment.log(
                    {
                        "samples/images": [
                            wandb.Image(pil, caption=f"train_sample_{i}_step_{step}")
                            for i, (pil, _) in enumerate(wandb_images)
                        ]
                    },
                    step=step,
                )
            except Exception as e:
                print(f"[wandb] sample image logging failed: {e}")

    # ------------- word-generation sampling & saving -------------

    @torch.no_grad()
    def _log_word_generation(self):
        if self.log_word_every_n_steps <= 0:
            return
        if self.latent_shape is None:
            return

        self.log_word_dir.mkdir(parents=True, exist_ok=True)

        B = max(1, self.num_sample_images)
        C, H, W = self.latent_shape

        base_seed = self.sample_seed if self.sample_seed is not None else 42
        seed = base_seed + 999
        gen = torch.Generator(device=self.device).manual_seed(seed)

        latents = torch.randn(
            (B, C, H, W),
            generator=gen,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        infer_scheduler = self.infer_scheduler
        infer_scheduler.set_timesteps(self.sample_num_inference_steps, device=self.device)
        timesteps = infer_scheduler.timesteps

        for t in tqdm(timesteps, desc="Sampling (word)", leave=False):
            pe, pam = self._make_pe_pam_word(B)

            t_batch = t.repeat(B).to(self.device)
            t_batch = t_batch.to(self.transformer.dtype)

            model_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=pe.to(self.transformer.dtype),
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

        sf = float(getattr(self.vae.config, "scaling_factor", 1.0))
        latents_vae = (latents / sf).to(self.vae.dtype)
        decoded = self.vae.decode(latents_vae).sample
        images = (decoded.clamp(-1, 1) + 1.0) / 2.0

        step = self.global_step
        wandb_images = []

        for i in range(B):
            img = images[i].detach().cpu()
            pil = to_pil_image(img)
            out_path = self.log_word_dir / f"step_{step:06d}_word_{i}.png"
            pil.save(out_path)

            wandb_images.append((pil, out_path))

        self.log("word_samples/last_step", float(step), prog_bar=False)

        if isinstance(self.logger, WandbLogger):
            try:
                import wandb
                caption_base = self.log_word_prompt or "<word prompt>"
                self.logger.experiment.log(
                    {
                        "word_samples/images": [
                            wandb.Image(pil, caption=f"{caption_base} | word_{i}_step_{step}")
                            for i, (pil, _) in enumerate(wandb_images)
                        ]
                    },
                    step=step,
                )
            except Exception as e:
                print(f"[wandb] word image logging failed: {e}")

    # ------------- batch-end hook -------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # generic sampling
        if self.sample_every_n_steps > 0:
            if self.global_step > 0 and (self.global_step % self.sample_every_n_steps == 0):
                try:
                    self._sample_and_save()
                except Exception as e:
                    print(f"[sample] error during sampling at step {self.global_step}: {e}")
                    self.sample_every_n_steps = 0

        # word-generation sampling
        if self.log_word_every_n_steps > 0:
            if self.global_step > 0 and (self.global_step % self.log_word_every_n_steps == 0):
                try:
                    self._log_word_generation()
                except Exception as e:
                    print(f"[word_sample] error during word sampling at step {self.global_step}: {e}")
                    self.log_word_every_n_steps = 0

        # NEW: token embedding snapshot
        if self.save_token_every_n_steps > 0:
            if self.global_step > 0 and (self.global_step % self.save_token_every_n_steps == 0):
                try:
                    self._save_token_embedding()
                except Exception as e:
                    print(f"[token_save] error during token embedding save at step {self.global_step}: {e}")
                    self.save_token_every_n_steps = 0

    # ------------- optimizer -------------

    def configure_optimizers(self):
        params = [self.concept_delta]

        if self.use_8bit_adam:
            if not torch.cuda.is_available():
                raise RuntimeError("8-bit AdamW (bitsandbytes) requires CUDA.")
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "Requested use_8bit_adam=True but bitsandbytes is not installed. "
                    "Install with `pip install bitsandbytes`."
                ) from e

            optimizer = bnb.optim.AdamW8bit(params, lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(params, lr=self.lr)

        return optimizer