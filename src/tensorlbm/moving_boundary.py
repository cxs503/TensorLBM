"""Moving-boundary immersed-boundary method for LBM.

Extends the static IBM in :mod:`tensorlbm.ibm` with time-varying (moving)
boundaries, enabling simulation of:

- Oscillating / flapping airfoils (pitch-plunge motion)
- Vibrating cylinders (vortex-induced vibration, VIV)
- Reciprocating pistons and valves
- Biomechanical flows (heart valves, swimming fish)

This corresponds to the **Arbitrary Lagrangian–Eulerian (ALE)** moving-
boundary capability in PowerFlow and XFlow.

Method
------
At each time step:

1. The boundary geometry is updated according to a prescribed motion law
   (position + velocity).
2. The IBM direct-forcing :func:`~tensorlbm.ibm.ibm_direct_forcing` is
   called with the instantaneous surface velocity.
3. A velocity correction is applied to the fluid cells that are swept by
   the moving boundary (formerly solid, now fluid, and vice versa).

Reference case – oscillating airfoil (pitch + plunge)
------------------------------------------------------
The NACA 0012 airfoil undergoes combined pitching and plunging motion:

    y_c(t) = A_h * sin(2π f_red t)        (plunge: ĥ = A_h/c)
    α(t)   = α₀ + A_α * sin(2π f_red t + φ)   (pitch about x/c=0.25)

Classical Theodorsen unsteady-lift benchmark parameters (Ol et al. 2009):
    ĥ = 0.25, α₀ = 0°, A_α = 45°, k = π f_red c / U = 0.5, φ = 90°.

References
----------
Peskin, C. S. (2002). The immersed boundary method. *Acta Numerica* 11, 479.
Ol, M. V. et al. (2009). Comparison of laminar deep-stall pitch-plunge
    kinematics for various speeds and pitch rates. *AIAA J.* 47, 2577.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch

from .d2q9 import equilibrium, macroscopic
from .ibm import ibm_force_spread, ibm_velocity_interpolate

__all__ = [
    "MotionLaw",
    "PitchPlungeMotion",
    "OscillatingCylinderMotion",
    "MovingBoundaryConfig",
    "update_moving_boundary",
    "run_oscillating_airfoil",
    "run_oscillating_foil_verification",
]


# ---------------------------------------------------------------------------
# Motion law protocols
# ---------------------------------------------------------------------------


class MotionLaw(Protocol):
    """Interface for a time-varying boundary motion law."""

    def position(self, step: int) -> tuple[float, float, float]:
        """Return (x_shift, y_shift, angle_rad) at the given time step."""
        ...

    def velocity(self, step: int, dt: float = 1.0) -> tuple[float, float, float]:
        """Return (vx, vy, omega_rad_per_step) at the given time step."""
        ...


@dataclass
class PitchPlungeMotion:
    """Combined pitch + plunge sinusoidal motion for an airfoil.

    Plunge: y(t) = A_h * sin(2π f_red t)
    Pitch:  α(t) = α0 + A_α * sin(2π f_red t + phase_shift)

    All quantities are in lattice units.
    """

    A_h: float = 20.0  # plunge amplitude (lattice cells)
    A_alpha: float = math.radians(15.0)  # pitch amplitude (radians)
    alpha0: float = 0.0  # mean pitch angle (radians)
    f_red: float = 0.002  # reduced frequency (cycles per step)
    phase_shift: float = math.pi / 2.0  # phase between pitch and plunge (rad)

    def position(self, step: int) -> tuple[float, float, float]:
        t = step * 2.0 * math.pi * self.f_red
        y_shift = self.A_h * math.sin(t)
        alpha = self.alpha0 + self.A_alpha * math.sin(t + self.phase_shift)
        return 0.0, y_shift, alpha

    def velocity(self, step: int, dt: float = 1.0) -> tuple[float, float, float]:
        t = step * 2.0 * math.pi * self.f_red
        vy = self.A_h * 2.0 * math.pi * self.f_red * math.cos(t)
        omega = self.A_alpha * 2.0 * math.pi * self.f_red * math.cos(t + self.phase_shift)
        return 0.0, vy, omega


@dataclass
class OscillatingCylinderMotion:
    """Transverse sinusoidal oscillation of a cylinder (VIV benchmark)."""

    A_y: float = 10.0  # transverse amplitude (lattice cells)
    f_osc: float = 0.001  # oscillation frequency (cycles/step)

    def position(self, step: int) -> tuple[float, float, float]:
        y_shift = self.A_y * math.sin(2.0 * math.pi * self.f_osc * step)
        return 0.0, y_shift, 0.0

    def velocity(self, step: int, dt: float = 1.0) -> tuple[float, float, float]:
        vy = self.A_y * 2.0 * math.pi * self.f_osc * math.cos(2.0 * math.pi * self.f_osc * step)
        return 0.0, vy, 0.0


# ---------------------------------------------------------------------------
# Marker-point geometry helpers
# ---------------------------------------------------------------------------


def _airfoil_markers(
    n_markers: int,
    chord: float,
    cx: float,
    cy: float,
    thickness_ratio: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate NACA 4-digit symmetric airfoil surface marker points."""
    t = torch.linspace(0.0, 2.0 * math.pi, n_markers + 1)[:-1]
    # Parametric NACA 0012 (approximate)
    xi = 0.5 * (1.0 - torch.cos(t))  # chord-wise parameter [0,1]
    c5 = thickness_ratio / 0.20
    yt = c5 * (0.2969 * xi.sqrt() - 0.1260 * xi - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.1015 * xi**4)
    # Upper and lower surfaces
    x_top = xi * chord + cx - chord / 2.0
    y_top = yt * chord + cy
    x_bot = xi.flip(0) * chord + cx - chord / 2.0
    y_bot = -yt.flip(0) * chord + cy
    xs = torch.cat([x_top, x_bot])
    ys = torch.cat([y_top, y_bot])
    return xs, ys


def _cylinder_markers(
    n_markers: int,
    radius: float,
    cx: float,
    cy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate circular cylinder surface marker points."""
    theta = torch.linspace(0.0, 2.0 * math.pi, n_markers + 1)[:-1]
    xs = cx + radius * torch.cos(theta)
    ys = cy + radius * torch.sin(theta)
    return xs, ys


def update_moving_boundary(
    xs: torch.Tensor,
    ys: torch.Tensor,
    cx_pivot: float,
    cy_pivot: float,
    x_shift: float,
    y_shift: float,
    angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update marker positions for a rigid-body displacement + rotation.

    Args:
        xs, ys:         Current marker positions.
        cx_pivot, cy_pivot: Pivot point for rotation.
        x_shift, y_shift:   Translation of the pivot.
        angle:          Rotation angle (radians, CCW positive).

    Returns:
        Updated ``(xs_new, ys_new)`` tensor pair.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = xs - cx_pivot
    dy = ys - cy_pivot
    xs_new = cx_pivot + x_shift + cos_a * dx - sin_a * dy
    ys_new = cy_pivot + y_shift + sin_a * dx + cos_a * dy
    return xs_new, ys_new


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovingBoundaryConfig:
    """Configuration for the oscillating airfoil (pitch-plunge) benchmark."""

    nx: int = 512
    ny: int = 256
    u_in: float = 0.05  # inlet velocity (l.u.)
    re: float = 500.0  # Reynolds number based on chord
    chord: float = 60.0  # airfoil chord length (lattice cells)
    n_markers: int = 120  # number of surface Lagrangian markers
    # Pitch-plunge parameters
    A_h_frac: float = 0.25  # plunge amplitude / chord
    A_alpha_deg: float = 15.0  # pitch amplitude (degrees)
    f_red: float = 0.0015  # reduced frequency (cycles/step)
    phase_shift_deg: float = 90.0  # pitch–plunge phase (degrees)
    # Geometry type: 'airfoil' or 'cylinder'
    geom: str = "airfoil"
    cylinder_radius: float = 25.0
    # Simulation
    n_steps: int = 4000
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
        L = self.chord if self.geom == "airfoil" else 2.0 * self.cylinder_radius
        return self.u_in * L / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_oscillating_airfoil(
    cfg: MovingBoundaryConfig | None = None,
    **kwargs: object,
) -> Path:
    """Run an oscillating airfoil (pitch-plunge) simulation with moving IBM.

    The airfoil undergoes sinusoidal pitch + plunge motion.  At each time step
    the Lagrangian markers are repositioned according to the kinematic law,
    the IBM body force is recomputed, and the fluid solver is advanced.

    Args:
        cfg:     Configuration object.
        **kwargs: Override any :class:`MovingBoundaryConfig` field.

    Returns:
        Path to the run output directory.
    """
    from .boundaries import apply_simple_channel_boundaries  # noqa: PLC0415
    from .config_io import save_config_json  # noqa: PLC0415
    from .logging_config import configure_logging  # noqa: PLC0415
    from .logging_config import logger as _logger
    from .solver import collide_bgk, stream  # noqa: PLC0415
    from .utils import (  # noqa: PLC0415
        get_reproducibility_metadata,
        prepare_run_dir,
        resolve_device,
    )

    if cfg is None:
        valid = set(MovingBoundaryConfig.__dataclass_fields__)
        cfg = MovingBoundaryConfig(**{k: v for k, v in kwargs.items() if k in valid})

    device = resolve_device(cfg.device)
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name or "oscillating_airfoil", cfg.overwrite)
    configure_logging(run_dir)
    save_config_json(asdict(cfg), run_dir / "config.json")

    _logger.info(
        "Oscillating airfoil: nx=%d ny=%d Re=%.1f geom=%s device=%s",
        cfg.nx,
        cfg.ny,
        cfg.re,
        cfg.geom,
        cfg.device,
    )

    nx, ny = cfg.nx, cfg.ny
    cx_pivot = nx * 0.35
    cy_pivot = ny / 2.0

    # Initial marker positions
    if cfg.geom == "airfoil":
        xs0, ys0 = _airfoil_markers(cfg.n_markers // 2, cfg.chord, cx_pivot, cy_pivot)
    else:
        xs0, ys0 = _cylinder_markers(cfg.n_markers, cfg.cylinder_radius, cx_pivot, cy_pivot)
    xs = xs0.to(device)
    ys = ys0.to(device)

    # Motion law
    A_h = cfg.A_h_frac * cfg.chord
    A_alpha = math.radians(cfg.A_alpha_deg)
    phase = math.radians(cfg.phase_shift_deg)
    motion = (
        PitchPlungeMotion(
            A_h=A_h,
            A_alpha=A_alpha,
            f_red=cfg.f_red,
            phase_shift=phase,
        )
        if cfg.geom == "airfoil"
        else OscillatingCylinderMotion(
            A_y=A_h,
            f_osc=cfg.f_red,
        )
    )

    # Init flow
    rho = torch.ones(ny, nx, device=device)
    ux0 = torch.full((ny, nx), cfg.u_in, device=device)
    uy0 = torch.zeros(ny, nx, device=device)
    f = equilibrium(rho, ux0, uy0)

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    steps_out: list[int] = []
    cl_out: list[float] = []
    cd_out: list[float] = []

    for step in range(cfg.n_steps):
        # Update geometry
        _, y_shift, angle = motion.position(step)
        vx_m, vy_m, omega = motion.velocity(step)
        xs_new, ys_new = update_moving_boundary(
            xs0.to(device),
            ys0.to(device),
            cx_pivot,
            cy_pivot,
            0.0,
            y_shift,
            angle,
        )
        dx = xs_new - cx_pivot
        dy = ys_new - cy_pivot
        vx_sfc = vx_m - omega * dy
        vy_sfc = vy_m + omega * dx

        # IBM force
        rho_f, ux_f, uy_f = macroscopic(f)
        u_interp_x, u_interp_y = ibm_velocity_interpolate(
            ux_f,
            uy_f,
            xs_new,
            ys_new,
            nx,
            ny,
        )
        F_ibm_x = vx_sfc - u_interp_x
        F_ibm_y = vy_sfc - u_interp_y
        Fx_field, Fy_field = ibm_force_spread(
            F_ibm_x,
            F_ibm_y,
            xs_new,
            ys_new,
            nx,
            ny,
        )

        # BGK with body force
        f = collide_bgk(f, rho_f, ux_f + 0.5 * Fx_field, uy_f + 0.5 * Fy_field, cfg.tau)
        f = apply_simple_channel_boundaries(f, cfg.u_in)
        f = stream(f)

        xs, ys = xs_new, ys_new

        if (step + 1) % cfg.output_interval == 0 or step == cfg.n_steps - 1:
            rho_pp, ux_pp, uy_pp = macroscopic(f)
            speed = torch.sqrt(ux_pp**2 + uy_pp**2)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(speed.cpu().numpy(), origin="lower", cmap="RdBu_r")
            ax.scatter(xs.cpu().numpy(), ys.cpu().numpy(), c="k", s=2)
            ax.set_title(f"Oscillating {cfg.geom} – step {step + 1}")
            fig.savefig(run_dir / f"step_{step + 1:06d}.png", dpi=100)
            plt.close(fig)

            # Lift & drag from IBM reaction force
            dyn_q = rho_f.mean() * cfg.u_in**2 * cfg.chord + 1e-12
            Cl = float((-Fy_field.sum() * 2.0 / dyn_q).item())
            Cd = float((-Fx_field.sum() * 2.0 / dyn_q).item())
            steps_out.append(step + 1)
            cl_out.append(Cl)
            cd_out.append(Cd)
            _logger.info("step=%d  Cl=%.4f  Cd=%.4f", step + 1, Cl, Cd)

    with (run_dir / "aero_coefficients.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "Cl", "Cd"])
        w.writerows(zip(steps_out, cl_out, cd_out, strict=True))

    meta = {
        **get_reproducibility_metadata(),
        "config": asdict(cfg),
        "steps": steps_out,
        "Cl": cl_out,
        "Cd": cd_out,
    }
    with (run_dir / "run_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    _logger.info("Oscillating airfoil complete → %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Oscillating-foil propulsion verification face
# ---------------------------------------------------------------------------
def run_oscillating_foil_verification(
    chord: int = 32,
    k: float = 1.0,
    h0_over_c: float = 0.15,
    alpha0_deg: float = 10.0,
    phi_deg: float = 90.0,
    mode: str = "coupled",
    re: float = 500.0,
    u_in: float = 0.15,
    n_cycles: int = 8,
    warmup_cycles: int = 2,
    output_root: Path | str = Path("outputs"),
    run_name: str | None = None,
) -> dict:
    """Oscillating-foil (pitch-plunge) propulsion case with a machine-checkable
    analytic anchor.

    This is the mature entry point for the pitch-plunge benchmark: it runs the
    immersed-boundary D2Q9 solver of :mod:`tensorlbm.oscillating_foil_lbm`
    (NACA 0012 section, solid interior, 4-point Peskin kernel, multi-direct
    forcing) and returns cycle-averaged ``C_T``, ``C_P``, ``eta`` next to the
    Theodorsen/Garrick closed form at the same parameters.  The full acceptance
    suite (criteria C1..C9) lives in
    ``benchmarks/p161w2_n05_oscillating_foil/run_sweep.py``.

    Compared with :func:`run_oscillating_airfoil` this entry point

    * uses a NACA 0012 section instead of a 1-cell-thick ellipse (a thin ellipse
      is thinner than the kernel support, so the effective body becomes blunt
      and the propulsive force is lost),
    * averages at least ``n_cycles`` cycles instead of a single instant,
    * reports the thrust increment ``C_T + C_D0`` and the collision efficiency
      ``C_T/C_P`` rather than raw forces.

    Returns:
        dict with ``coeffs`` (LBM), ``theory`` (Garrick), ``grid``, ``config``
        and the run directory.
    """
    from .flapping_propulsion import FlappingKinematics, garrick_coefficients  # noqa: PLC0415
    from .oscillating_foil_lbm import (  # noqa: PLC0415
        OscillatingFoil2D,
        OscillatingFoilConfig,
        cycle_statistics,
    )

    cfg = OscillatingFoilConfig(
        chord=chord,
        re=re,
        U=u_in,
        k=k,
        h0_over_c=h0_over_c,
        alpha0_deg=alpha0_deg,
        phi_deg=phi_deg,
        mode=mode,
        n_cycles=n_cycles,
        warmup_cycles=warmup_cycles,
    )
    solver = OscillatingFoil2D(cfg)
    rows = [solver.step(diagnostics=False) for _ in range(cfg.total_steps())]
    coeffs = cycle_statistics(rows, cfg)

    kin = FlappingKinematics(
        U=cfg.U,
        c=float(cfg.chord),
        h0=cfg.motion.h0,
        alpha0=cfg.motion.alpha0,
        phi=cfg.motion.phi,
        a=cfg.motion.a,
    )
    g = garrick_coefficients(kin, cfg.k)
    theory = {
        "C_T": float(g.C_T),
        "C_P": float(g.C_P),
        "eta": float(g.eta),
        "strouhal": float(kin.strouhal(cfg.k)),
    }

    run_dir = Path(output_root) / (run_name or f"oscillating_foil_{mode}_k{k:g}")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chord": cfg.chord,
            "re": cfg.re,
            "U": cfg.U,
            "k": cfg.k,
            "h0_over_c": cfg.h0_over_c,
            "alpha0_deg": cfg.alpha0_deg,
            "phi_deg": cfg.phi_deg,
            "mode": cfg.mode,
            "steps_per_cycle": cfg.steps_per_cycle(),
        },
        "grid": [solver.nx, solver.ny],
        "coeffs": coeffs,
        "theory": theory,
    }
    with (run_dir / "oscillating_foil.json").open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    payload["run_dir"] = str(run_dir)
    return payload
