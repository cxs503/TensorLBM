"""Denoising Diffusion Probabilistic Model (DDPM) as an :class:`AI4SApplication`.

This module turns the Denoising Diffusion Probabilistic Model (DDPM) idea of
Ho et al. 2020 (NeurIPS, arXiv:2006.11239) into a concrete platform
application: a generative model that *samples* 2-D velocity-field snapshots
from pure Gaussian noise.  The forward process (a fixed, hand-crafted Markov
noising chain) and the reverse process (a learned denoiser that strips noise
one step at a time) are both implemented in pure PyTorch — no external
diffusion library is required.

The learned object is a convolutional denoiser ``(x_t, t) -> eps_hat`` that
predicts the noise added at timestep ``t``; training minimises the simple
``MSE(eps_hat, eps)`` objective of Ho et al. (eq. 14), and generation runs the
reverse (sampling) loop from ``x_T ~ N(0, I)`` down to ``x_0``.

Data production reuses the LBM smoke solver of :mod:`tensorlbm.ai.pipeline`
(``_run_les_smoke``) to harvest real velocity snapshots, with a synthetic
random-vortex-field generator as an injectable / config-selectable
alternative.  Both the solver (``produce_fn``) and the training loop
(``train_fn``) are injectable so the full :meth:`run` pipeline stays testable
without any HPC run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorlbm.apps.base import (
    AI4SApplication,
    DataProduct,
    Prediction,
    TrainingResult,
)

__all__ = [
    "DDPM",
    "DenoiseCNN",
    "DiffusionArch",
    "GenerativeFlow",
    "build_diffusion_model",
    "load_diffusion_model",
    "save_diffusion_model",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(x: Any) -> torch.Tensor:
    """Coerce ``x`` into a float32 ``torch.Tensor``."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _activation_module(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"unknown activation {name!r}")


def _num_groups(channels: int) -> int:
    """Pick a GroupNorm group count that divides ``channels`` (≤ 8)."""
    channels = int(channels)
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position embedding of timestep ``t`` (Ho et al. 2020)."""
    device = t.device
    half = int(dim) // 2
    exponent = (
        -math.log(10000.0)
        * torch.arange(0, half, device=device, dtype=torch.float32)
        / max(half, 1)
    )
    freqs = torch.exp(exponent)
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if int(dim) % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionArch:
    """Hyper-parameters describing a :class:`DenoiseCNN` + its DDPM schedule."""

    in_channels: int = 2
    out_channels: int = 2
    hidden_dim: int = 32
    n_layers: int = 3
    time_emb_dim: int = 32
    activation: str = "silu"
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"  # "linear" | "cosine"


class _ResBlock(nn.Module):
    """Residual conv block with scale-and-shift time conditioning."""

    def __init__(self, channels: int, time_emb_dim: int, activation: str) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = _activation_module(activation)
        self.time_proj = nn.Linear(int(time_emb_dim), 2 * channels)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_proj(temb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h_in = h
        h = self.act(self.norm1(h) * (1.0 + scale) + shift)
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h_in + h


class DenoiseCNN(nn.Module):
    """Convolutional denoiser ``(x_t, t) -> eps_hat`` for 2-D fields.

    A lightweight, residual CNN that keeps the spatial resolution fixed (so it
    is well-defined for any grid size): an input projection, a stack of
    time-conditioned :class:`_ResBlock` layers, and an output projection back
    to ``out_channels``.  The timestep ``t`` is embedded with a sinusoidal
    encoding and mapped through a small MLP before conditioning each block.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 32,
        n_layers: int = 3,
        time_emb_dim: int = 32,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.time_emb_dim = int(time_emb_dim)
        self.activation = str(activation)

        act = _activation_module(self.activation)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.hidden_dim),
            act,
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.in_conv = nn.Conv2d(self.in_channels, self.hidden_dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                _ResBlock(self.hidden_dim, self.hidden_dim, self.activation)
                for _ in range(self.n_layers)
            ]
        )
        self.out_conv = nn.Conv2d(self.hidden_dim, self.out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(_sinusoidal_embedding(t, self.time_emb_dim))
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, temb)
        return self.out_conv(h)

    def arch_dict(self) -> dict[str, Any]:
        """Return the denoiser hyper-parameters as a plain dict."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "time_emb_dim": self.time_emb_dim,
            "activation": self.activation,
        }


# ---------------------------------------------------------------------------
# The DDPM wrapper (forward noising chain + learned reverse process)
# ---------------------------------------------------------------------------


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapping a denoiser network.

    Holds the pre-computed noising schedule (betas / alphas / cumulative
    products as buffers) and exposes:

    * :meth:`forward` / :meth:`train_step` — the training objective
      ``MSE(eps_hat(x_t, t), eps)`` with a uniformly-sampled timestep.
    * :meth:`sample` / :meth:`sample_from` — the reverse (generative) loop.

    The schedule follows Ho et al. 2020 (linear or cosine beta schedule);
    ``timesteps`` is configurable so tests can use a tiny chain.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        *,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.schedule = str(schedule).lower()

        betas = self._make_betas()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def _make_betas(self) -> torch.Tensor:
        if self.schedule == "cosine":
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            s = 0.008
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1.0 + s) * math.pi / 2.0) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.02)
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    # -- forward / training -------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise ``eps_hat`` given a noisy field ``x`` and time ``t``."""
        return self.denoiser(x, t)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward noising step: ``x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps``."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def train_step(self, x0: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """One DDPM training step: random ``t`` + noise + ``MSE(eps_hat, eps)``."""
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.denoiser(x_t, t)
        return F.mse_loss(eps_hat, noise)

    # -- reverse / sampling -------------------------------------------------

    @torch.no_grad()
    def sample_from(self, x_t: torch.Tensor) -> torch.Tensor:
        """Reverse-diffuse a provided noisy tensor down to ``x_0`` (Ho et al. alg. 2)."""
        b = x_t.shape[0]
        device = x_t.device
        x = x_t
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            eps_hat = self.denoiser(x, t)
            beta = self.betas[i]
            if i > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            sigma = torch.sqrt(self.posterior_variance[i]).clamp_min(1e-8)
            x = (
                self.sqrt_recip_alphas[i]
                * (x - beta / self.sqrt_one_minus_alphas_cumprod[i] * eps_hat)
                + sigma * z
            )
        return x

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample a new field from pure Gaussian noise (full reverse loop)."""
        if seed is not None:
            torch.manual_seed(int(seed))
        x_t = torch.randn(shape, device=device)
        return self.sample_from(x_t)

    def arch_dict(self) -> dict[str, Any]:
        """Return the full architecture (denoiser + schedule) as a dict."""
        d: dict[str, Any] = {}
        denoiser_arch = getattr(self.denoiser, "arch_dict", None)
        if callable(denoiser_arch):
            d.update(denoiser_arch())
        d.update(
            {
                "timesteps": self.timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "schedule": self.schedule,
            }
        )
        return d


def build_diffusion_model(arch: Mapping[str, Any] | DiffusionArch | None = None) -> DDPM:
    """Construct a :class:`DDPM` (denoiser + schedule) from an arch mapping."""
    if arch is None:
        a = DiffusionArch()
    elif isinstance(arch, DiffusionArch):
        a = arch
    else:
        a = DiffusionArch(
            in_channels=int(arch.get("in_channels", 2)),
            out_channels=int(arch.get("out_channels", 2)),
            hidden_dim=int(arch.get("hidden_dim", 32)),
            n_layers=int(arch.get("n_layers", 3)),
            time_emb_dim=int(arch.get("time_emb_dim", 32)),
            activation=str(arch.get("activation", "silu")),
            timesteps=int(arch.get("timesteps", 1000)),
            beta_start=float(arch.get("beta_start", 1e-4)),
            beta_end=float(arch.get("beta_end", 0.02)),
            schedule=str(arch.get("schedule", "linear")),
        )
    denoiser = DenoiseCNN(
        in_channels=a.in_channels,
        out_channels=a.out_channels,
        hidden_dim=a.hidden_dim,
        n_layers=a.n_layers,
        time_emb_dim=a.time_emb_dim,
        activation=a.activation,
    )
    return DDPM(
        denoiser,
        timesteps=a.timesteps,
        beta_start=a.beta_start,
        beta_end=a.beta_end,
        schedule=a.schedule,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_diffusion_model(model: DDPM, path: str | Path) -> Path:
    """Serialize a :class:`DDPM` (denoiser weights + schedule buffers) to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {"arch": model.arch_dict(), "model_class": "DDPM", "format_version": 1}
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return p


def load_diffusion_model(path: str | Path) -> DDPM:
    """Load a :class:`DDPM` saved by :func:`save_diffusion_model`."""
    p = Path(path)
    state = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    arch_dict: dict[str, Any] = {}
    if meta_path.exists():
        arch_dict = json.loads(meta_path.read_text()).get("arch") or {}
    model = build_diffusion_model(arch_dict) if arch_dict else build_diffusion_model()
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Synthetic flow-field generators
# ---------------------------------------------------------------------------


def _random_vortex_field(
    nx: int,
    ny: int,
    *,
    n_vortices: int = 6,
    seed: int = 0,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a smooth 2-D velocity field from random Gaussian vortices.

    Deterministic given ``seed``; a cheap, physically-plausible stand-in for an
    LBM snapshot used by the test suite and by ``data_source="vortex"``.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ys = torch.arange(int(ny), device=device, dtype=torch.float32)
    xs = torch.arange(int(nx), device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ux = torch.zeros_like(xx)
    uy = torch.zeros_like(xx)
    for _ in range(max(1, int(n_vortices))):
        cx = float(nx) * torch.rand(1, generator=g).item()
        cy = float(ny) * torch.rand(1, generator=g).item()
        strength = (torch.rand(1, generator=g).item() - 0.5) * 4.0
        radius = 0.5 + 2.0 * torch.rand(1, generator=g).item()
        dx = xx - cx
        dy = yy - cy
        d2 = dx * dx + dy * dy
        omega = torch.exp(-d2 / (2.0 * radius * radius)) * strength
        ux = ux - dy * omega
        uy = uy + dx * omega
    return ux, uy


def _run_les_snapshots(
    nx: int,
    ny: int,
    *,
    tau: float,
    c_s: float,
    n_steps: int,
    sample_every: int,
    seed: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Harvest velocity snapshots from the LBM smoke solver (ai.pipeline)."""
    from tensorlbm.ai.pipeline import _run_les_smoke

    return _run_les_smoke(
        nx=int(nx),
        ny=int(ny),
        tau=float(tau),
        c_s=float(c_s),
        n_steps=int(n_steps),
        sample_every=int(sample_every),
        seed=int(seed),
        device=device,
    )


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------


class GenerativeFlow(AI4SApplication):
    """DDPM generative model of 2-D velocity fields as an AI4S application.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"diffusion"``, understood by
            :class:`tensorlbm.ml.serving.InferenceService`).
    """

    name: str = "generative_flow"
    family: str = "diffusion"
    version: str = "1.0"

    def __init__(
        self,
        *,
        train_fn: Callable[..., Any] | None = None,
        produce_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Create the app, optionally injecting the data / training steps.

        Args:
            train_fn: Optional override for the training step.  Called as
                ``train_fn(dataset, model, cfg)`` and must return a
                :class:`TrainingResult` (or a mapping with ``model_path`` /
                ``metrics`` / ``arch`` keys).  Defaults to the local DDPM
                noise-prediction loop.
            produce_fn: Optional override for data production.  Called as
                ``produce_fn(nx=..., ny=..., n_snapshots=..., seed=...)`` and
                must return a list of ``(ux, uy)`` velocity-field tuples.
                Defaults to the LBM smoke solver (or a vortex generator, per
                ``data_source``).
        """
        super().__init__()
        self._train_fn = train_fn
        self._produce_fn = produce_fn

    # ---- developer-implemented interface ---------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Generate 2-D velocity snapshots (LBM or random vortex) and return metadata.

        Snapshots are stacked into ``(C, ny, nx)`` tensors (``C=2`` for
        ``(ux, uy)``) and carried in :attr:`DataProduct.metadata["snapshots"]`
        so ``make_dataset`` needs no disk access.
        """
        nx = int(cfg.get("nx", 16))
        ny = int(cfg.get("ny", 16))
        n_snapshots = int(cfg.get("n_snapshots", 4))
        seed = int(cfg.get("seed", 0))
        device = torch.device(str(cfg.get("device", "cpu")))
        data_source = str(cfg.get("data_source", "les")).strip().lower()

        if self._produce_fn is not None:
            snapshots = self._produce_fn(
                nx=nx,
                ny=ny,
                n_snapshots=n_snapshots,
                seed=seed,
            )
        elif data_source == "vortex":
            snapshots = [
                _random_vortex_field(nx, ny, n_vortices=6, seed=seed + k, device=device)
                for k in range(n_snapshots)
            ]
        else:
            snapshots = _run_les_snapshots(
                nx=nx,
                ny=ny,
                tau=float(cfg.get("tau", 0.8)),
                c_s=float(cfg.get("c_s", 0.1)),
                n_steps=int(cfg.get("n_steps", 16)),
                sample_every=int(cfg.get("sample_every", 4)),
                seed=seed,
                device=device,
            )[:n_snapshots]

        if not snapshots:
            raise ValueError("velocity-snapshot production returned no snapshots")

        fields = [torch.stack([_to_tensor(ux), _to_tensor(uy)], dim=0) for ux, uy in snapshots]
        first = fields[0]
        return DataProduct(
            name="2D velocity-field snapshots (generative flow)",
            field_name="u",
            shape=tuple(first.shape),
            dtype=str(first.dtype),
            units="lu",
            metadata={
                "snapshots": fields,
                "n_snapshots": len(fields),
                "grid": (int(first.shape[1]), int(first.shape[2])),
                "channels": ["u", "v"],
                "data_source": data_source,
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the DDPM (denoiser + noise schedule) from an arch mapping."""
        return build_diffusion_model(arch)

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Normalise the snapshots to ``[-1, 1]`` and stack them into samples.

        Returns ``{"samples": (N, C, ny, nx), "n_samples", "grid", "channels",
        "data_min", "data_max"}``.  The min/max scale is retained so callers can
        denormalise generated fields back to physical units.
        """
        snapshots = product.metadata.get("snapshots")
        if not snapshots:
            raise ValueError("DataProduct metadata must carry 'snapshots'")

        samples = torch.stack([_to_tensor(s) for s in snapshots], dim=0)
        data_min = float(samples.min())
        data_max = float(samples.max())
        rng = data_max - data_min
        if rng < 1e-8:
            rng = 1.0
        normalized = (2.0 * (samples - data_min) / rng - 1.0).clamp(-1.0, 1.0)

        first = samples[0]
        return {
            "samples": normalized,
            "n_samples": int(samples.shape[0]),
            "grid": (int(first.shape[1]), int(first.shape[2])),
            "channels": ["u", "v"],
            "data_min": data_min,
            "data_max": data_max,
        }

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Train the DDPM and return weights path + metrics.

        The injectable ``train_fn`` (if provided) is called as
        ``train_fn(dataset, model, cfg)``; otherwise the local DDPM loop
        (random timestep + noise prediction MSE) runs and the checkpoint is
        written with :func:`save_diffusion_model`.
        """
        out_path = Path(
            str(cfg.get("out_path") or cfg.get("model_path") or "diffusion_model.pt"),
        )

        if self._train_fn is not None:
            return _coerce_training_result(
                self._train_fn(dataset, model, cfg),
                out_path=out_path,
            )

        result = _train_ddpm(
            dataset,
            cast(DDPM, model),
            out_path,
            epochs=int(cfg.get("epochs", 20)),
            batch_size=int(cfg.get("batch_size", 4)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
        )
        arch = dict(result.get("arch") or {})
        if not arch:
            arch = model.arch_dict()
        return TrainingResult(
            model_path=str(result.get("path", out_path)),
            metrics={"train_loss": float(result.get("train_loss", float("nan")))},
            arch=arch,
        )

    def infer(self, model: torch.nn.Module, sample: Any = None) -> Prediction:
        """Sample a generated velocity field via the DDPM reverse loop.

        ``sample`` controls the generation request and may be:

        * ``None`` — sample one ``(C, ny, nx)`` field at the default 32×32 grid.
        * a mapping ``{"shape": (ny, nx) | (C, ny, nx), "n_samples", "seed",
          "device"}``.
        * a ``(ny, nx)`` or ``(C, ny, nx)`` tuple/list.
        * a :class:`torch.Tensor` — treated as the initial noise ``x_T`` that
          is reverse-diffused (its shape defines the output).

        Returns a :class:`Prediction` whose ``output`` is the generated field
        (``(C, ny, nx)``, or ``(n_samples, C, ny, nx)`` when batched), still in
        the normalised ``[-1, 1]`` space.
        """
        model = cast(DDPM, model)
        if isinstance(sample, torch.Tensor):
            x_t = _to_tensor(sample)
            single = x_t.dim() == 3
            if single:
                x_t = x_t.unsqueeze(0)
            with torch.no_grad():
                generated = model.sample_from(x_t.to(next(model.parameters()).device))
            out = generated[0] if single else generated
        else:
            n_samples, channels, ny, nx, device, seed = _resolve_generation_spec(sample, model)
            generated = model.sample(
                (n_samples, channels, ny, nx),
                device=device,
                seed=seed,
            )
            out = generated[0] if n_samples == 1 else generated

        return Prediction(
            output=out,
            metadata={
                "field_name": "u",
                "shape": tuple(out.shape),
                "units": "lu (normalized)",
                "channels": ["u", "v"],
                "family": self.family,
            },
        )


# ---------------------------------------------------------------------------
# Default DDPM training loop
# ---------------------------------------------------------------------------


def _train_ddpm(
    dataset: Mapping[str, Any],
    model: DDPM,
    out_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """DDPM noise-prediction training loop (Ho et al. eq. 14) + checkpoint save."""
    X = dataset["samples"]
    torch_device = torch.device(device)
    torch.manual_seed(int(seed))
    model.to(torch_device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    n = int(X.shape[0])
    batch_size = max(1, int(batch_size))

    final_loss = float("nan")
    for _ in range(max(0, int(epochs))):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(torch_device)
            optimizer.zero_grad()
            loss = model.train_step(xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        final_loss = epoch_loss / n

    model.eval()
    path = save_diffusion_model(model, out_path)
    return {
        "path": str(path),
        "train_loss": final_loss,
        "arch": model.arch_dict(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_shape(shape: Any, default_channels: int) -> tuple[int, int, int]:
    """Parse a ``(ny, nx)`` / ``(C, ny, nx)`` / scalar shape into ``(C, ny, nx)``."""
    if isinstance(shape, int):
        return default_channels, int(shape), int(shape)
    dims = list(shape)
    if len(dims) == 2:
        return default_channels, int(dims[0]), int(dims[1])
    if len(dims) == 3:
        return int(dims[0]), int(dims[1]), int(dims[2])
    raise ValueError(f"shape must be (ny, nx) or (C, ny, nx), got {shape!r}")


def _resolve_generation_spec(
    sample: Any,
    model: DDPM,
) -> tuple[int, int, int, int, torch.device, int]:
    """Resolve a generation request into ``(n, C, ny, nx, device, seed)``."""
    denoiser = model.denoiser
    channels = int(getattr(denoiser, "in_channels", 2))
    n_samples = 1
    seed = 0
    device = torch.device("cpu")
    ny = nx = 32

    if sample is None:
        pass
    elif isinstance(sample, Mapping):
        shape = sample.get("shape") or sample.get("size")
        if shape is not None:
            channels, ny, nx = _parse_shape(shape, channels)
        n_samples = int(sample.get("n_samples", 1))
        seed = int(sample.get("seed", 0))
        device = torch.device(str(sample.get("device", "cpu")))
    else:
        channels, ny, nx = _parse_shape(sample, channels)

    return n_samples, channels, ny, nx, device, seed


def _coerce_training_result(
    result: Any,
    *,
    out_path: Path,
) -> TrainingResult:
    """Normalise a ``train_fn`` return value into a :class:`TrainingResult`."""
    if isinstance(result, TrainingResult):
        return result
    if isinstance(result, Mapping):
        metrics = dict(result.get("metrics") or {})
        if not metrics:
            metrics = {
                "train_loss": float(
                    result.get("train_loss", result.get("final_train_loss", float("nan")))
                ),
            }
        return TrainingResult(
            model_path=str(result.get("model_path") or result.get("path") or out_path),
            metrics=metrics,
            arch=dict(result.get("arch") or {}),
        )
    raise TypeError(
        "train_fn must return a TrainingResult or a mapping with "
        f"'model_path'/'metrics'/'arch' keys, got {type(result).__name__}",
    )
