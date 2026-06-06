"""End-to-end HPC + AI pipeline: solve → sample → store → train → infer.

The :func:`run_ai_les_pipeline` function chains everything in this
sub-package together and is the high-level entry point used by the
platform agent.  It runs a *small* 2-D LBM simulation (so the pipeline is
fast in CI), harvests velocity snapshots into a regression dataset,
records every step in a SQLite database, trains an MLP eddy-viscosity
model, and finally executes a short LBM run that uses the trained model
as its LES closure.  The function returns paths to every artefact plus
training metrics so an agent can summarise the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from ..d2q9 import equilibrium, macroscopic
from ..solver import collide_bgk, stream
from ..turbulence import collide_smagorinsky_bgk
from .database import LBMDatabase
from .dataset import EddyViscosityDataset, extract_les_samples_2d, save_dataset_pt
from .inference import collide_ai_les_bgk, predict_nu_t_2d
from .model import load_model
from .train import TrainConfig, train_eddy_viscosity_model

# ---------------------------------------------------------------------------
# Reference data-generation simulation
# ---------------------------------------------------------------------------

def _init_random_velocity_field(
    nx: int, ny: int, seed: int, device: torch.device, mean_u: float = 0.05,
) -> torch.Tensor:
    """Initialise an LBM distribution from a turbulent-looking random field.

    A simple superposition of low-wavenumber sinusoids is good enough to
    generate non-trivial strain-rate samples for training a regression
    model — this is a *training-data generator*, not a physically
    converged simulation, and the AI model is judged purely on its
    ability to fit the algebraic Smagorinsky label.
    """
    torch.manual_seed(int(seed))
    ys = torch.arange(ny, device=device).float()
    xs = torch.arange(nx, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    kx = 2.0 * torch.pi / max(nx, 1)
    ky = 2.0 * torch.pi / max(ny, 1)
    ux = mean_u + 0.02 * (
        torch.sin(2.0 * kx * xx) * torch.cos(ky * yy)
        + 0.5 * torch.sin(4.0 * kx * xx + 0.3) * torch.cos(2.0 * ky * yy)
    )
    uy = 0.02 * (
        torch.cos(kx * xx) * torch.sin(2.0 * ky * yy)
        + 0.5 * torch.cos(3.0 * kx * xx) * torch.sin(ky * yy + 0.7)
    )
    rho = torch.ones_like(ux)
    return equilibrium(rho, ux, uy)


def _run_les_smoke(
    nx: int,
    ny: int,
    tau: float,
    c_s: float,
    n_steps: int,
    sample_every: int,
    seed: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run a periodic LES smoke test, return ``(ux, uy)`` snapshots."""
    f = _init_random_velocity_field(nx, ny, seed=seed, device=device)
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(int(n_steps)):
        f = collide_smagorinsky_bgk(f, tau=float(tau), C_s=float(c_s))
        f = stream(f)
        if sample_every > 0 and (step + 1) % sample_every == 0:
            _rho, ux, uy = macroscopic(f)
            snapshots.append((ux.detach().clone(), uy.detach().clone()))
    if not snapshots:
        _rho, ux, uy = macroscopic(f)
        snapshots.append((ux.detach().clone(), uy.detach().clone()))
    return snapshots


def _coarsen_mean(field: torch.Tensor, factor: int) -> torch.Tensor:
    """Block-average a 2-D field by an integer factor."""
    if factor <= 1:
        return field
    if field.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {tuple(field.shape)}")
    ny, nx = int(field.shape[0]), int(field.shape[1])
    cy = ny // int(factor)
    cx = nx // int(factor)
    if cy < 1 or cx < 1:
        raise ValueError(
            f"coarsen factor {factor} is too large for field shape {tuple(field.shape)}",
        )
    trimmed = field[: cy * factor, : cx * factor]
    return trimmed.reshape(cy, factor, cx, factor).mean(dim=(1, 3))


def _run_dns_reference(
    nx: int,
    ny: int,
    tau: float,
    n_steps: int,
    sample_every: int,
    seed: int,
    device: torch.device,
    dns_scale: int = 2,
    warmup_steps: int = 20,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run a higher-resolution BGK reference and downsample to LES grid.

    This is a practical in-repo "DNS-like" generator: evolve a finer LBM
    grid with plain BGK (no SGS model), then block-average snapshots to
    the requested training resolution.
    """
    scale = max(1, int(dns_scale))
    nx_f = int(nx) * scale
    ny_f = int(ny) * scale
    f = _init_random_velocity_field(nx_f, ny_f, seed=seed, device=device)
    for _ in range(max(0, int(warmup_steps))):
        f = collide_bgk(f, tau=float(tau))
        f = stream(f)

    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(int(n_steps)):
        f = collide_bgk(f, tau=float(tau))
        f = stream(f)
        if sample_every > 0 and (step + 1) % sample_every == 0:
            _rho, ux_f, uy_f = macroscopic(f)
            ux = _coarsen_mean(ux_f, scale)
            uy = _coarsen_mean(uy_f, scale)
            snapshots.append((ux.detach().clone(), uy.detach().clone()))
    if not snapshots:
        _rho, ux_f, uy_f = macroscopic(f)
        ux = _coarsen_mean(ux_f, scale)
        uy = _coarsen_mean(uy_f, scale)
        snapshots.append((ux.detach().clone(), uy.detach().clone()))
    return snapshots


def _run_ai_validation(
    nx: int,
    ny: int,
    tau: float,
    n_steps: int,
    seed: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    """Short LBM run using the trained AI LES closure and a BGK baseline.

    Returns max velocity / kinetic-energy diagnostics for each case as a
    sanity check that the AI model keeps the solver stable.
    """
    model = load_model(model_path).to(device)
    f_ai = _init_random_velocity_field(nx, ny, seed=seed, device=device)
    f_bgk = f_ai.clone()
    for _ in range(int(n_steps)):
        f_ai = collide_ai_les_bgk(f_ai, tau=float(tau), model=model)
        f_ai = stream(f_ai)
        f_bgk = collide_bgk(f_bgk, tau=float(tau))
        f_bgk = stream(f_bgk)
    _rho_ai, ux_ai, uy_ai = macroscopic(f_ai)
    _rho_bgk, ux_bgk, uy_bgk = macroscopic(f_bgk)
    nu_t = predict_nu_t_2d(model, ux_ai, uy_ai)
    return {
        "ai_umax": float(torch.sqrt(ux_ai * ux_ai + uy_ai * uy_ai).max()),
        "bgk_umax": float(torch.sqrt(ux_bgk * ux_bgk + uy_bgk * uy_bgk).max()),
        "ai_ke": float(0.5 * (ux_ai * ux_ai + uy_ai * uy_ai).mean()),
        "bgk_ke": float(0.5 * (ux_bgk * ux_bgk + uy_bgk * uy_bgk).mean()),
        "ai_nu_t_mean": float(nu_t.mean()),
        "ai_nu_t_max": float(nu_t.max()),
        "stable": bool(
            torch.isfinite(ux_ai).all() and torch.isfinite(uy_ai).all(),
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

@dataclass
class AIPipelineResult:
    """Artefacts and diagnostics produced by :func:`run_ai_les_pipeline`."""

    work_dir: Path
    db_path: Path
    dataset_path: Path
    model_path: Path
    run_id: int
    dataset_id: int
    model_id: int
    n_samples: int
    data_source: str = "les"
    n_snapshots: int = 0
    training_time_s: float = 0.0
    training: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["work_dir"] = str(self.work_dir)
        d["db_path"] = str(self.db_path)
        d["dataset_path"] = str(self.dataset_path)
        d["model_path"] = str(self.model_path)
        return d


def run_ai_les_pipeline(
    work_dir: str | Path,
    nx: int = 64,
    ny: int = 64,
    tau: float = 0.8,
    c_s: float = 0.1,
    data_steps: int = 40,
    sample_every: int = 10,
    val_steps: int = 20,
    train_config: TrainConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    run_name: str = "ai_les_demo",
    data_source: Literal["les", "dns"] = "les",
    dns_scale: int = 2,
    dns_warmup_steps: int = 20,
) -> AIPipelineResult:
    """Run the full HPC + AI demonstration end-to-end.

    See module docstring for the pipeline overview.

    Args:
        work_dir: Output directory.  Will be created if needed.
        nx, ny: Grid size of both the data-generation and validation runs.
        tau: Baseline LBM relaxation time.
        c_s: Smagorinsky constant used to label the training data.
        data_steps: Number of LBM steps in the data-generation run.
        sample_every: Cadence at which velocity snapshots are sampled.
        val_steps: Number of LBM steps in the AI-LES validation run.
        train_config: Optional :class:`TrainConfig`.
        seed: Reproducibility seed.
        device: ``"cpu"``, ``"cuda"`` or any torch device string.
        run_name: Logical name recorded in the database.
        data_source: ``"les"`` (Smagorinsky data run) or ``"dns"``
            (higher-resolution BGK reference then downsampled).
        dns_scale: Fine/coarse scale for ``data_source="dns"``.
        dns_warmup_steps: Optional warmup steps for the DNS reference run.

    Returns:
        An :class:`AIPipelineResult` populated with paths and metrics.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "ai_pipeline.db"
    dataset_path = work / "dataset.pt"
    model_path = work / "model.pt"
    torch_device = torch.device(device)
    source = str(data_source).strip().lower()
    if source not in {"les", "dns"}:
        raise ValueError(
            f"data_source must be 'les' or 'dns', got {data_source!r}",
        )
    dns_scale = max(1, int(dns_scale))
    dns_warmup_steps = max(0, int(dns_warmup_steps))

    db = LBMDatabase.open(db_path)
    try:
        run_id = db.insert_run(
            name=run_name,
            run_type="dns_data_generation" if source == "dns" else "les_data_generation",
            config={
                "nx": int(nx), "ny": int(ny), "tau": float(tau),
                "c_s": float(c_s), "data_steps": int(data_steps),
                "sample_every": int(sample_every), "seed": int(seed),
                "device": str(device), "data_source": source,
                "dns_scale": int(dns_scale),
                "dns_warmup_steps": int(dns_warmup_steps),
            },
            output_dir=str(work),
        )

        if source == "dns":
            snapshots = _run_dns_reference(
                nx=int(nx), ny=int(ny), tau=float(tau),
                n_steps=int(data_steps), sample_every=int(sample_every),
                seed=int(seed), device=torch_device, dns_scale=int(dns_scale),
                warmup_steps=int(dns_warmup_steps),
            )
        else:
            snapshots = _run_les_smoke(
                nx=int(nx), ny=int(ny), tau=float(tau), c_s=float(c_s),
                n_steps=int(data_steps), sample_every=int(sample_every),
                seed=int(seed), device=torch_device,
            )

        feats_list: list[torch.Tensor] = []
        targs_list: list[torch.Tensor] = []
        for ux, uy in snapshots:
            f, t = extract_les_samples_2d(ux, uy, c_s=float(c_s))
            feats_list.append(f)
            targs_list.append(t)
        features = torch.cat(feats_list, dim=0)
        targets = torch.cat(targs_list, dim=0)
        dataset = EddyViscosityDataset(
            features=features, targets=targets, c_s=float(c_s),
            description=(
                f"{source.upper()} snapshots from run #{run_id} "
                f"({len(snapshots)} frames)"
            ),
        )
        save_dataset_pt(dataset, dataset_path)
        dataset_id = db.insert_dataset(
            name=f"{run_name}_dataset",
            path=str(dataset_path),
            n_samples=len(dataset),
            run_id=run_id,
            metadata={
                "c_s": float(c_s),
                "n_snapshots": len(snapshots),
                "grid": [int(ny), int(nx)],
                "data_source": source,
                "dns_scale": int(dns_scale),
            },
        )

        training = train_eddy_viscosity_model(
            dataset=dataset,
            out_path=model_path,
            config=train_config or TrainConfig(),
        )
        model_id = db.insert_model(
            name=f"{run_name}_eddy_viscosity_mlp",
            path=str(model_path),
            arch=training["arch"],
            dataset_id=dataset_id,
            metrics={
                "final_train_mse": training["final_train_mse"],
                "final_val_mse": training["final_val_mse"],
                "final_val_r2": training["final_val_r2"],
            },
        )

        validation = _run_ai_validation(
            nx=int(nx), ny=int(ny), tau=float(tau),
            n_steps=int(val_steps), seed=int(seed) + 1,
            device=torch_device, model_path=model_path,
        )
    finally:
        db.close()

    return AIPipelineResult(
        work_dir=work,
        db_path=db_path,
        dataset_path=dataset_path,
        model_path=model_path,
        run_id=run_id,
        dataset_id=dataset_id,
        model_id=model_id,
        n_samples=len(dataset),
        data_source=source,
        n_snapshots=len(snapshots),
        training_time_s=float(training.get("training_time_s", 0.0)),
        training=training,
        validation=validation,
    )


def run_ai_dns_pipeline(
    work_dir: str | Path,
    nx: int = 64,
    ny: int = 64,
    tau: float = 0.8,
    c_s: float = 0.1,
    data_steps: int = 40,
    sample_every: int = 10,
    val_steps: int = 20,
    train_config: TrainConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    run_name: str = "ai_dns_demo",
    dns_scale: int = 2,
    dns_warmup_steps: int = 20,
) -> AIPipelineResult:
    """Convenience wrapper: DNS data generation + training + AI-LES embed."""
    return run_ai_les_pipeline(
        work_dir=work_dir,
        nx=nx,
        ny=ny,
        tau=tau,
        c_s=c_s,
        data_steps=data_steps,
        sample_every=sample_every,
        val_steps=val_steps,
        train_config=train_config,
        seed=seed,
        device=device,
        run_name=run_name,
        data_source="dns",
        dns_scale=dns_scale,
        dns_warmup_steps=dns_warmup_steps,
    )
