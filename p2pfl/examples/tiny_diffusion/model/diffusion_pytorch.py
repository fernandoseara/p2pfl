"""Tiny DDPM denoiser for 2D point distributions."""

import math

import lightning as L
import torch
import torch.nn.functional as F

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class SinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TinyDiffusion(L.LightningModule):
    """
    Tiny DDPM model for learning 2D point distributions.

    Uses a small MLP conditioned on timestep to denoise 2D points.
    Training follows the standard DDPM objective: predict the noise
    added at a random timestep.
    """

    def __init__(
        self,
        point_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_emb_dim: int = 64,
        timesteps: int = 100,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.timesteps = timesteps
        self.lr = lr

        # --- Beta schedule (linear) ---
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        # --- Time embedding ---
        self.time_emb = torch.nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            torch.nn.Linear(time_emb_dim, hidden_dim),
            torch.nn.SiLU(),
        )

        # --- Denoiser MLP ---
        layers = [torch.nn.Linear(point_dim + hidden_dim, hidden_dim), torch.nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU()]
        layers.append(torch.nn.Linear(hidden_dim, point_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy input x_t and timestep t."""
        t_emb = self.time_emb(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process: add noise at timestep t."""
        sqrt_ab = self.sqrt_alpha_bar[t, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t, None]
        return sqrt_ab * x_0 + sqrt_omab * noise

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        x_0 = batch["x"].float()
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        x_0 = batch["x"].float()
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, n_samples: int = 1000) -> torch.Tensor:
        """Generate samples via DDPM reverse process."""
        device = self.betas.device
        x = torch.randn(n_samples, 2, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            beta = self.betas[i]
            alpha = 1.0 - beta
            alpha_bar = self.alpha_bar[i]
            noise_pred = self(x, t)
            # DDPM update
            x = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_bar).sqrt()) * noise_pred)
            if i > 0:
                x += beta.sqrt() * torch.randn_like(x)
        return x


def model_build_fn(*args, **kwargs) -> LightningModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModel(TinyDiffusion(*args, **kwargs), compression=compression)
