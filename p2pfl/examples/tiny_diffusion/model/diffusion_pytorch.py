"""Tiny conditional DDPM denoiser for 2D point distributions.

Architecture inspired by tanelp/tiny-diffusion: applies sinusoidal Fourier
features to each input coordinate (with high scale) to overcome the spectral
bias of MLPs and let the model learn high-frequency 2D distributions.

Conditional on a class label (num_classes), so a single model can generate
samples from each of the underlying distributions on demand.
"""

import math

import lightning as L
import torch
import torch.nn.functional as F

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class SinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal embedding. With scale > 1 acts as Fourier features for input coords."""

    def __init__(self, dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device).float() / half)
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(torch.nn.Module):
    """Residual block: x + GELU(Linear(x))."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.ff = torch.nn.Linear(size, size)
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.ff(x))


class TinyDiffusion(L.LightningModule):
    """
    Conditional DDPM model for learning 2D point distributions per class.

    Applies sinusoidal Fourier features (scale=25) to each input coordinate
    so the MLP can learn high-frequency target densities (moons, spiral, etc.).
    A learned class embedding lets a single model represent multiple
    distributions and sample from each one on demand.
    """

    def __init__(
        self,
        point_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        time_emb_dim: int = 128,
        input_emb_dim: int = 128,
        input_scale: float = 25.0,
        num_classes: int = 4,
        class_emb_dim: int = 128,
        timesteps: int = 1000,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.save_hyperparameters()
        self.timesteps = timesteps
        self.lr = lr
        self.grad_clip = grad_clip
        self.num_classes = num_classes

        # --- Beta schedule (linear) ---
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        # --- Embeddings: Fourier per coord, sinusoidal time, learned class ---
        self.input_emb_x = SinusoidalEmbedding(input_emb_dim, scale=input_scale)
        self.input_emb_y = SinusoidalEmbedding(input_emb_dim, scale=input_scale)
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        self.class_emb = torch.nn.Embedding(num_classes, class_emb_dim)

        # --- Denoiser MLP with residual blocks ---
        concat_size = 2 * input_emb_dim + time_emb_dim + class_emb_dim
        layers: list[torch.nn.Module] = [torch.nn.Linear(concat_size, hidden_dim), torch.nn.GELU()]
        for _ in range(n_layers):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(torch.nn.Linear(hidden_dim, point_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy input x_t, timestep t, and class label."""
        x_emb = self.input_emb_x(x_t[:, 0])
        y_emb = self.input_emb_y(x_t[:, 1])
        t_emb = self.time_emb(t)
        c_emb = self.class_emb(label)
        h = torch.cat([x_emb, y_emb, t_emb, c_emb], dim=-1)
        return self.net(h)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process: add noise at timestep t."""
        sqrt_ab = self.sqrt_alpha_bar[t, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t, None]
        return sqrt_ab * x_0 + sqrt_omab * noise

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None) -> None:
        self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

    def _step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_0 = batch["x"].float()
        label = batch["label"].long()
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self(x_t, t, label)
        return F.mse_loss(noise_pred, noise)

    def training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, n_samples: int = 1000, label: int = 0) -> torch.Tensor:
        """Generate samples for a given class label via DDPM reverse process."""
        device = self.betas.device
        x = torch.randn(n_samples, 2, device=device)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            beta = self.betas[i]
            alpha = 1.0 - beta
            alpha_bar = self.alpha_bar[i]
            noise_pred = self(x, t, labels)
            # DDPM update
            x = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_bar).sqrt()) * noise_pred)
            if i > 0:
                x += beta.sqrt() * torch.randn_like(x)
        return x


def model_build_fn(*args, **kwargs) -> LightningModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModel(TinyDiffusion(*args, **kwargs), compression=compression)
