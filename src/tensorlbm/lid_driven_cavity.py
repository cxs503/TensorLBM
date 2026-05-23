"""Lid-driven cavity: classic 2D CFD benchmark with Ghia et al. (1982) validation.

The lid-driven cavity problem is a square domain [0,L]×[0,L] in which:
- Top wall (y=L) moves at velocity *u_lid* in the x-direction.
- All other walls are no-slip (bounce-back).
- No inlet/outlet.

At steady state the velocity profiles along the horizontal and vertical
centrelines can be compared against the tabulated data of
Ghia, Ghia & Shin (1982) for Re = 100, 400, 1000.

Exported symbols
----------------
- :class:`LidDrivenCavityConfig`  – simulation configuration
- :func:`run_lid_driven_cavity`   – runner
- ``GHIA_RE100``, ``GHIA_RE400``, ``GHIA_RE1000``  – reference data dicts
- :func:`compare_ghia`            – compare a run's centreline profiles to Ghia
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data  (Re = 100 / 400 / 1000)
# y/L positions and u/U_lid for the vertical centreline (x=L/2)
# ---------------------------------------------------------------------------

GHIA_RE100: dict[str, list[float]] = {
    "y": [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
          0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
    "u": [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
          -0.21090, -0.20581, -0.13641,  0.00332,  0.23111,  0.68717,  0.73722,  0.78871,
           0.84123,  1.00000],
}

GHIA_RE400: dict[str, list[float]] = {
    "y": [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
          0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
    "u": [0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
          -0.17119, -0.11477,  0.02135,  0.16256,  0.29093,  0.55892,  0.61756,  0.68439,
           0.75837,  1.00000],
}

GHIA_RE1000: dict[str, list[float]] = {
    "y": [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
          0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
    "u": [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
          -0.10648, -0.06080,  0.05702,  0.18719,  0.33304,  0.46604,  0.51117,  0.57492,
           0.65928,  1.00000],
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LidDrivenCavityConfig:
    """Configuration for a 2D lid-driven cavity simulation.

    Parameters
    ----------
    n : int
        Grid size (n × n square domain; default 128).
    u_lid : float
        Lid velocity in lattice units (default 0.1).
    re : float
        Reynolds number Re = u_lid · n / ν.
    n_steps : int
        Total number of LBM steps.
    output_interval : int
        Steps between diagnostics and flow snapshots.
    """

    n: int = 128
    u_lid: float = 0.1
    re: float = 100.0
    n_steps: int = 5000
    output_interval: int = 500
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def nu(self) -> float:
        return self.u_lid * self.n / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.n < 8:
            raise ValueError("n must be >= 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_lid <= 0.0 or self.re <= 0.0:
            raise ValueError("u_lid and re must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"Invalid tau={self.tau:.4f}; decrease re or u_lid"
            )

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return f"n{self.n}_re{re_label}_ulid{self.u_lid:.3f}_steps{self.n_steps}"

    def save(self, path: "Path | str") -> None:
        from .config_io import save_config_json
        save_config_json(self, path)

    @classmethod
    def load(cls, path: "Path | str") -> "LidDrivenCavityConfig":
        from .config_io import load_config_json
        return load_config_json(cls, path)


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def _apply_lid_cavity_boundaries(
    f: torch.Tensor,
    n: int,
    u_lid: float,
    device: torch.device,
) -> torch.Tensor:
    """Apply lid-driven cavity boundary conditions.

    - Top wall (y=n-1): moving lid with velocity u_lid (Zou/He BC).
    - Bottom, left, right walls (y=0, x=0, x=n-1): no-slip bounce-back.
    """
    from .d2q9 import C, OPPOSITE, W

    c = C.to(device)
    w = W.to(device)
    opp = OPPOSITE.to(device)

    # --- Moving top lid (y = n-1): Zou/He velocity BC ---
    # Incoming directions at top (cy < 0): 4, 7, 8
    # Outgoing (cy > 0): 2, 5, 6  → to be recomputed
    f6, f2, f5 = f[6, -1, :], f[2, -1, :], f[5, -1, :]
    f3, f0, f1 = f[3, -1, :], f[0, -1, :], f[1, -1, :]
    f7, f4, f8 = f[7, -1, :], f[4, -1, :], f[8, -1, :]

    rho_top = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)
    f_new = f.clone()
    f_new[4, -1, :] = f2 - (2.0 / 3.0) * rho_top * 0.0  # uy=0 at lid (only ux)
    # uy_lid = 0, ux_lid = u_lid
    f_new[4, -1, :] = f2
    f_new[7, -1, :] = f5 - 0.5 * (f1 - f3) - (1.0 / 6.0) * rho_top * u_lid
    f_new[8, -1, :] = f6 + 0.5 * (f1 - f3) - (1.0 / 6.0) * rho_top * u_lid

    # Use proper Zou/He for the moving lid
    rho_top = (
        f_new[0, -1, :] + f_new[1, -1, :] + f_new[3, -1, :]
        + 2.0 * (f_new[2, -1, :] + f_new[5, -1, :] + f_new[6, -1, :])
    )
    f_new[4, -1, :] = f_new[2, -1, :]
    f_new[7, -1, :] = f_new[5, -1, :] - 0.5 * (f_new[1, -1, :] - f_new[3, -1, :]) \
                      - (1.0 / 6.0) * rho_top * u_lid
    f_new[8, -1, :] = f_new[6, -1, :] + 0.5 * (f_new[1, -1, :] - f_new[3, -1, :]) \
                      - (1.0 / 6.0) * rho_top * u_lid

    # --- Static no-slip walls: bounce-back ---
    wall = torch.zeros((n, n), dtype=torch.bool, device=device)
    wall[0, :] = True    # bottom
    wall[:, 0] = True    # left
    wall[:, -1] = True   # right
    bounced = f_new.clone()
    bounced[:, wall] = f_new[opp][:, wall]
    return bounced


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_lid_driven_cavity(config: LidDrivenCavityConfig) -> Path:
    """Run a 2D lid-driven cavity simulation and save results.

    Args:
        config: Validated :class:`LidDrivenCavityConfig`.

    Returns:
        Path to the run output directory.
    """
    config.validate()
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "lid_driven_cavity", config.resolved_run_name(), config.overwrite
    )

    n = config.n
    rho0 = torch.ones((n, n), device=device)
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    diagnostics: list[dict[str, object]] = []
    initial_mass = float(rho0.sum().item())

    print(
        f"Running 2D lid-driven cavity  n={n}  Re={config.re:.0f}  "
        f"tau={config.tau:.4f}  steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    for step in range(1, config.n_steps + 1):
        f = collide_bgk(f, tau=config.tau)
        f = stream(f)
        f = _apply_lid_cavity_boundaries(f, n, config.u_lid, device)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            mass = float(rho.sum().item())
            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(torch.sqrt(ux * ux + uy * uy).max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            print(
                f"step={point.step:5d}  mass={point.mass:.5f}  "
                f"drift={point.mass_drift:+.5f}  max|u|={point.max_speed:.5f}"
            )

    # Save centreline profiles
    rho_final, ux_final, uy_final = macroscopic(f)
    mid = n // 2

    # Vertical centreline: ux at x=n//2 as a function of y  (normalised)
    ux_vc = (ux_final[:, mid] / config.u_lid).tolist()
    y_norm = (torch.arange(n, device=device).float() / (n - 1)).tolist()

    # Horizontal centreline: uy at y=n//2 as a function of x (normalised)
    uy_hc = (uy_final[mid, :] / config.u_lid).tolist()
    x_norm = (torch.arange(n, device=device).float() / (n - 1)).tolist()

    metadata["diagnostics"] = diagnostics
    metadata["centreline"] = {
        "y_norm": y_norm,
        "ux_vc_norm": ux_vc,
        "x_norm": x_norm,
        "uy_hc_norm": uy_hc,
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}")
    return run_dir


# ---------------------------------------------------------------------------
# Ghia comparison helper
# ---------------------------------------------------------------------------

def compare_ghia(
    run_dir: Path | str,
    re: float = 100.0,
) -> dict[str, float]:
    """Compare centreline profiles in a run against Ghia et al. (1982) data.

    Reads ``run_metadata.json`` from *run_dir*, interpolates the simulated
    ``ux_vc_norm`` at the Ghia tabulated y/L positions, and returns the
    root-mean-square error.

    Args:
        run_dir: Output directory produced by :func:`run_lid_driven_cavity`.
        re: Reynolds number for Ghia reference selection (100, 400, or 1000).

    Returns:
        Dict with keys ``"rmse_ux"`` (and ``"re"``).
    """
    ref_map = {100.0: GHIA_RE100, 400.0: GHIA_RE400, 1000.0: GHIA_RE1000}
    if re not in ref_map:
        raise ValueError(f"Ghia data not available for Re={re}; choose from 100, 400, 1000")
    ref = ref_map[re]

    run_dir = Path(run_dir)
    data = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    y_sim = data["centreline"]["y_norm"]
    ux_sim = data["centreline"]["ux_vc_norm"]

    # Linear interpolation of simulated profile at Ghia y positions
    y_sim_t = torch.tensor(y_sim, dtype=torch.float64)
    ux_sim_t = torch.tensor(ux_sim, dtype=torch.float64)
    y_ref = torch.tensor(ref["y"], dtype=torch.float64)
    ux_ref = torch.tensor(ref["u"], dtype=torch.float64)

    ux_interp = torch.interp(y_ref, y_sim_t, ux_sim_t) if hasattr(torch, "interp") else \
        _torch_interp(y_ref, y_sim_t, ux_sim_t)

    rmse = float(torch.sqrt(((ux_interp - ux_ref) ** 2).mean()).item())
    return {"re": re, "rmse_ux": rmse}


def _torch_interp(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple linear interpolation fallback (torch.interp not available in all versions)."""
    import numpy as np
    return torch.tensor(
        np.interp(x_new.numpy(), x.numpy(), y.numpy()),
        dtype=x_new.dtype,
    )


__all__ = [
    "LidDrivenCavityConfig",
    "run_lid_driven_cavity",
    "GHIA_RE100",
    "GHIA_RE400",
    "GHIA_RE1000",
    "compare_ghia",
]
