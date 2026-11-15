"""Finite-density-ratio (two-fluid) Rayleigh-Taylor instability, D2Q9 Shan-Chen.

Why this module exists
----------------------
:mod:`tensorlbm.rayleigh_taylor` integrates a **Boussinesq** two-layer flow: one
density field, buoyancy entering as ``A g phi``.  Those equations are invariant
under ``(phi, y) -> (-phi, -y)``, so the solver cannot produce a spike/bubble
asymmetry -- the case report of ``CASE-RT-INSTABILITY-ATWOOD`` records exactly
that as an *ungated* diagnostic (``u_spike / u_bubble = 1.0004`` against the
two-fluid prediction ``sqrt((1+A)/(1-A)) = 1.732`` at ``A = 1/2``).  The anchor
of that case is the **spike** growth rate, which a model whose spike and bubble
are the same front cannot answer.

This module supplies the missing physics instead of re-weighting an estimator:
a single-component Shan-Chen (SCMP) lattice gas, i.e. two *phases* of one
component separated by a real interface with a measured density ratio, driven by
gravity acting on mass (``F_g = rho g``) rather than on a Boussinesq scalar.
Because the equilibrium carries the local density, the momentum equation retains
the variable inertia ``rho (du/dt + u.grad u)`` and the bubble/spike asymmetry
appears from the model.  A two-component (MCMP) cross-repulsion variant was
tried first and rejected -- it only reached ``A = 0.33`` at the largest coupling
that stayed stable for ``tau <= 1`` (see ``docs/CASE-RT-TWOFLUID.md`` sec. 5), so
its evidence is recorded but no case is built on it.

Conventions (locked here, asserted in ``tests/test_rayleigh_taylor_twofluid.py``)
-------------------------------------------------------------------------------
* ``psi(rho) = 1 - exp(-rho)`` (Shan & Chen 1994 exponential pseudopotential).
* Interaction force, using the lattice gradient identity
  ``sum_i w_i c_i psi(x + c_i) = cs^2 grad psi`` (to gradient order):

      F_int(x) = sc_g psi(x) sum_i w_i c_i psi(x + c_i)

  so ``sc_g > 0`` is **attractive** (the force points up the pseudopotential
  gradient), which is the phase-separating branch used here.  The relation to the
  ``G`` of the single-component force quoted elsewhere in the library
  (``F = G cs^2 psi sum_i w_i c_i psi``, ``cs^2 = 1/3``) is ``sc_g = G / 3``;
  the calibration below is stated in ``sc_g`` because that is what is measured.
* Gravity is a body force per unit mass ``F_g = rho g`` with ``g = (0, -g_mag)``
  (downward).  Initial condition: **heavy phase on top** (unstable).
* Forcing is the Exact Difference Method (Kupershtokh 2004):
  ``f_post = f + (f_eq(rho,u) - f)/tau + [f_eq(rho, u + F/rho) - f_eq(rho,u)]``
  with the momentum-consistent velocity ``u = (sum_i f_i c_i + 0.5 F)/rho``.
* Walls: no-slip bounce-back at the first and last row (the wall sits half-way
  between row 0 and row 1), x periodic with one wavelength; ``psi`` is mirrored
  into the wall rows so wall wetting is neutral.
* Units: lattice.  The dimensionless time is ``tau = t sqrt(A g k)`` with
  ``k = 2 pi / lam``; ``Re = u lam / nu`` with ``nu = (tau_r - 1/2)/3``.

Reference values (closed form, evaluated in code -- nothing transcribed)
-----------------------------------------------------------------------
``u_b = sqrt(2 A g / ((1+A) C k))``, ``u_s = sqrt(2 A g / ((1-A) C k))``
(Layzer 1955; Goncharov 2002, Phys. Rev. Lett. 88:134502; two-fluid form quoted
by arXiv:2009.14655 Eq. (1)), 2-D geometric constant ``C = 1``, with
``A = (rho_h - rho_l)/(rho_h + rho_l)`` **measured** from the coexistence run of
this model rather than assumed from the nominal 1/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .d2q9 import C, OPPOSITE, W, equilibrium

__all__ = [
    "CS2",
    "SCMPRTConfig",
    "CoexistenceResult",
    "ATWOOD_NOMINAL",
    "atwood_from_ratio",
    "ratio_from_atwood",
    "psi_exp",
    "psi_sqrt",
    "scmp_forces",
    "scmp_step",
    "scmp_coexistence",
    "calibrate_atwood",
    "init_scmp_rt",
    "interface_crossings",
    "front_state",
    "layzer_velocity",
    "run_scmp_rt",
]

CS2 = 1.0 / 3.0

#: nominal Atwood number of the anchor case ( -> density ratio 3 ).
ATWOOD_NOMINAL = 0.5


def atwood_from_ratio(ratio: float) -> float:
    """``A = (r - 1) / (r + 1)`` for a density ratio ``r = rho_h / rho_l``."""
    return (ratio - 1.0) / (ratio + 1.0)


def ratio_from_atwood(atwood: float) -> float:
    """Inverse of :func:`atwood_from_ratio`."""
    return (1.0 + atwood) / (1.0 - atwood)


def psi_exp(rho: torch.Tensor) -> torch.Tensor:
    """Shan-Chen exponential pseudopotential, bounded in ``[0, 1)`` for ``rho > 0``."""
    return 1.0 - torch.exp(-rho)


def psi_sqrt(rho: torch.Tensor) -> torch.Tensor:
    """Square-root pseudopotential ``psi = sqrt(rho)`` (stronger low-density coupling)."""
    return torch.sqrt(torch.clamp(rho, min=0.0))


PSI_KINDS = {"exp": psi_exp, "sqrt": psi_sqrt}


def _psi(rho: torch.Tensor, kind: str = "exp") -> torch.Tensor:
    if kind not in PSI_KINDS:
        raise ValueError(f"unknown psi kind {kind!r}; expected one of {tuple(PSI_KINDS)}")
    return PSI_KINDS[kind](rho)


# ---------------------------------------------------------------------------
# Shan-Chen forcing and stepping
# ---------------------------------------------------------------------------


def _shift(field: torch.Tensor, i: int) -> torch.Tensor:
    """``field(x + c_i)`` on an ``(ny, nx)`` grid (x periodic, y wrapped)."""
    cx, cy = int(C[i, 0]), int(C[i, 1])
    return torch.roll(field, shifts=(-cy, -cx), dims=(0, 1))


def _mirror_wall_rows(field: torch.Tensor) -> torch.Tensor:
    """Mirror row 1 -> row 0 and row -2 -> row -1 (neutral wall wetting)."""
    out = field.clone()
    out[0, :] = field[1, :]
    out[-1, :] = field[-2, :]
    return out


def scmp_forces(
    rho: torch.Tensor,
    sc_g: float,
    gravity_y: float = 0.0,
    psi_kind: str = "exp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Fx, Fy)`` for the SCMP model including gravity.

    ``sc_g > 0`` is attractive; ``gravity_y < 0`` points down.
    """
    # solid rows hold no mass (rho = 0 there), so psi = 0 on the wall: the wall is
    # *neutral*.  Mirroring the fluid psi into the wall rows (the usual shortcut)
    # makes the wall strongly attractive and grows wetting films that drain the
    # vapour phase -- measured as a runaway "vapour" density of 46..1e11 in the
    # calibration, see docs/CASE-RT-TWOFLUID.md sec. 5.
    psi = _psi(rho, psi_kind)
    w = W.to(dtype=rho.dtype, device=rho.device)
    fx = torch.zeros_like(rho)
    fy = torch.zeros_like(rho)
    for i in range(1, 9):
        shifted = _shift(psi, i)
        fx = fx + w[i] * float(C[i, 0]) * shifted
        fy = fy + w[i] * float(C[i, 1]) * shifted
    fx = sc_g * psi * fx
    fy = sc_g * psi * fy
    if gravity_y != 0.0:
        fy = fy + rho * gravity_y
    return fx, fy


def _momentum(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = C.to(dtype=f.dtype, device=f.device)
    rho = f.sum(dim=0)
    mx = (f * c[:, 0].view(9, 1, 1)).sum(dim=0)
    my = (f * c[:, 1].view(9, 1, 1)).sum(dim=0)
    return rho, mx, my


def _stream_bounce_back(f_post: torch.Tensor) -> torch.Tensor:
    """Stream with periodic x and **mass-conserving** half-way bounce-back in y.

    Rows 0 and ``ny-1`` are solid (no mass is stored there).  A population whose
    destination row would be outside the fluid band is reflected at the *source*
    node with the opposite direction, which makes the transport an exact
    permutation of the fluid populations -- total mass is conserved to machine
    precision (a wrap-around-and-overwrite closure leaks mass; that defect was
    measured at -0.05 %/step before this rewrite, see the case report sec. 5).
    """
    ny = f_post.shape[1]
    out = torch.zeros_like(f_post)
    out[0] = f_post[0]  # rest direction does not move
    for i in range(1, 9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        shifted = torch.roll(f_post[i], shifts=(-cx,), dims=(1,))
        lo = max(1, 1 + cy)
        hi = min(ny - 2, ny - 2 + cy)
        if lo <= hi:
            out[i, lo:hi + 1, :] = shifted[lo - cy:hi + 1 - cy, :]
        if cy == -1:  # source row 1 would leave through the bottom wall
            out[int(OPPOSITE[i]), 1, :] = out[int(OPPOSITE[i]), 1, :] + f_post[i, 1, :]
        elif cy == 1:  # source row ny-2 would leave through the top wall
            out[int(OPPOSITE[i]), ny - 2, :] = (
                out[int(OPPOSITE[i]), ny - 2, :] + f_post[i, ny - 2, :]
            )
    return out


def scmp_step(
    f: torch.Tensor,
    *,
    tau: float,
    sc_g: float,
    gravity_y: float,
    psi_kind: str = "exp",
) -> torch.Tensor:
    """One lattice step of the SCMP model (BGK + EDM forcing + walls)."""
    rho, mx, my = _momentum(f)
    fx, fy = scmp_forces(rho, sc_g, gravity_y, psi_kind)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (mx + 0.5 * fx) / rho_safe
    uy = (my + 0.5 * fy) / rho_safe
    feq = equilibrium(rho, ux, uy)
    feq_f = equilibrium(rho, ux + fx / rho_safe, uy + fy / rho_safe)
    f_post = f + (feq - f) / tau + (feq_f - feq)
    return _stream_bounce_back(f_post)


def velocity_field(f: torch.Tensor, *, sc_g: float, gravity_y: float,
                   psi_kind: str = "exp") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rho, ux, uy)`` with the half-force velocity correction (for diagnostics)."""
    rho, mx, my = _momentum(f)
    fx, fy = scmp_forces(rho, sc_g, gravity_y, psi_kind)
    rho_safe = torch.clamp(rho, min=1e-12)
    return rho, (mx + 0.5 * fx) / rho_safe, (my + 0.5 * fy) / rho_safe


# ---------------------------------------------------------------------------
# Coexistence calibration: which sc_g gives the anchor Atwood number
# ---------------------------------------------------------------------------


@dataclass
class CoexistenceResult:
    sc_g: float
    tau: float
    psi_kind: str
    rho_high: float
    rho_low: float
    ratio: float
    atwood: float
    steps: int
    mass_drift_pct: float
    stable: bool

    def as_dict(self) -> dict:
        return {
            "sc_g": self.sc_g,
            "tau": self.tau,
            "psi": self.psi_kind,
            "rho_high": self.rho_high,
            "rho_low": self.rho_low,
            "density_ratio": self.ratio,
            "atwood": self.atwood,
            "steps": self.steps,
            "mass_drift_pct": self.mass_drift_pct,
            "stable": self.stable,
        }


def scmp_coexistence(
    sc_g: float,
    *,
    rho_high_init: float = 1.6,
    rho_low_init: float = 0.35,
    ny: int = 64,
    nx: int = 8,
    width: float = 2.0,
    steps: int = 6000,
    tau: float = 1.0,
    psi_kind: str = "exp",
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> CoexistenceResult:
    """Measure SCMP coexistence densities for ``sc_g``.

    A liquid slab sandwiched between vapour (flat interfaces, neutral walls); the
    bulk densities are read from the central 10 % band and the outer 10 % bands
    after relaxation.  No gravity: this is the equation-of-state measurement whose
    result the RT run uses for its initial profile and for its Atwood number.
    """
    device = device or torch.device("cpu")
    y = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
    mid = 0.5 * (ny - 1)
    prof = torch.tanh((y - mid) / width).expand(ny, nx).contiguous()
    rho = 0.5 * (rho_high_init + rho_low_init) + 0.5 * (rho_high_init - rho_low_init) * prof
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    mass0 = float(rho.sum())

    band = max(1, int(round(0.10 * ny)))
    mid_i = int(round(mid))
    for _ in range(steps):
        f = scmp_step(f, tau=tau, sc_g=sc_g, gravity_y=0.0, psi_kind=psi_kind)
        if not torch.isfinite(f).all():
            return CoexistenceResult(sc_g, tau, psi_kind, float("nan"), float("nan"),
                                     float("nan"), float("nan"), steps, float("nan"), False)
    rho = f.sum(dim=0)
    rho_high = float(rho[mid_i - band: mid_i + band, :].mean())
    rho_low = float(rho[:band, :].mean())
    ratio = rho_high / max(rho_low, 1e-12)
    return CoexistenceResult(
        sc_g=sc_g,
        tau=tau,
        psi_kind=psi_kind,
        rho_high=rho_high,
        rho_low=rho_low,
        ratio=ratio,
        atwood=atwood_from_ratio(ratio) if ratio == ratio else float("nan"),
        steps=steps,
        mass_drift_pct=100.0 * (float(rho.sum()) - mass0) / mass0,
        stable=True,
    )


def calibrate_atwood(
    *,
    target_ratio: float = 3.0,
    grid: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70),
    tau: float = 1.0,
    psi_kind: str = "exp",
    steps: int = 6000,
    dtype: torch.dtype = torch.float64,
) -> tuple[float, list[CoexistenceResult]]:
    """Scan ``sc_g`` and interpolate to the coupling that yields ``target_ratio``.

    No point is discarded: every scan result is returned for audit, and the
    refined point is appended.  If the scan does not bracket the target the best
    measured point is returned with ``ratio`` reported as measured (the case then
    runs at *its* Atwood number, which is what the reference formula uses).
    """
    results = [scmp_coexistence(g, tau=tau, psi_kind=psi_kind, steps=steps, dtype=dtype)
               for g in grid]
    good = [r for r in results if r.stable and math.isfinite(r.ratio)]
    if not good:
        return grid[len(grid) // 2], results
    order = sorted(good, key=lambda r: r.ratio)
    bracket = None
    for lo, hi in zip(order[:-1], order[1:]):
        if lo.ratio <= target_ratio <= hi.ratio:
            bracket = (lo, hi)
            break
    best = min(good, key=lambda r: abs(r.ratio - target_ratio))
    if bracket is not None:
        lo, hi = bracket
        w = (target_ratio - lo.ratio) / (hi.ratio - lo.ratio)
        g_ref = lo.sc_g + w * (hi.sc_g - lo.sc_g)
        refined = scmp_coexistence(g_ref, tau=tau, psi_kind=psi_kind, steps=steps, dtype=dtype)
        results.append(refined)
        if refined.stable and math.isfinite(refined.ratio):
            best = min([best, refined], key=lambda r: abs(r.ratio - target_ratio))
    return best.sc_g, results


# ---------------------------------------------------------------------------
# The RT run
# ---------------------------------------------------------------------------


@dataclass
class SCMPRTConfig:
    """Single-mode RT box: one wavelength wide, ``ny = layers * lam`` tall."""

    lam: int = 128
    layers: float = 6.0
    atwood: float = ATWOOD_NOMINAL
    sc_g: float = 0.55
    gravity: float = 1.5e-4
    tau_relax: float = 0.8
    a0_over_lam: float = 0.02
    interface_width: float = 2.0
    steps: int = 6000
    sample_every: int = 5
    ma_limit: float = 0.20
    psi_kind: str = "exp"
    dtype: torch.dtype = torch.float64
    name: str = "scmp"

    @property
    def nx(self) -> int:
        return int(self.lam)

    @property
    def ny(self) -> int:
        return int(round(self.layers * self.lam))

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.lam

    @property
    def nu(self) -> float:
        return (self.tau_relax - 0.5) / 3.0

    @property
    def y_interface(self) -> float:
        return 0.5 * self.ny

    def validate(self) -> None:
        if self.lam < 16:
            raise ValueError("lam must be >= 16")
        if not (-1.0 < self.atwood < 1.0):
            raise ValueError("atwood must be in (-1, 1)")
        if self.sc_g <= 0.0:
            raise ValueError("sc_g must be > 0 (attractive branch)")
        if self.gravity <= 0.0:
            raise ValueError("gravity is the downward magnitude and must be > 0")
        if self.tau_relax <= 0.5:
            raise ValueError("tau_relax must be > 0.5")
        if not (0.0 < self.a0_over_lam < 0.25):
            raise ValueError("a0_over_lam must be in (0, 0.25)")
        if self.y_interface < 2.0 * self.lam:
            raise ValueError("interface too close to the wall: increase layers")


def init_scmp_rt(
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise ``(f, phi)`` with the heavy phase on top and one cosine mode."""
    dtype, device = cfg.dtype, torch.device("cpu")
    y = torch.arange(cfg.ny, dtype=dtype, device=device).view(cfg.ny, 1)
    x = torch.arange(cfg.nx, dtype=dtype, device=device).view(1, cfg.nx)
    y_int = cfg.y_interface + cfg.a0_over_lam * cfg.lam * torch.cos(
        2.0 * math.pi * x / cfg.lam
    )
    phi = torch.clamp(torch.tanh((y - y_int) / cfg.interface_width), -1.0, 1.0)
    half = 0.5 * (rho_high + rho_low)
    amp = 0.5 * (rho_high - rho_low)
    rho = half + amp * phi
    rho[0, :] = 0.0  # rows 0 and ny-1 are solid: no mass stored there
    rho[-1, :] = 0.0
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero), phi


def interface_crossings(phi: torch.Tensor) -> torch.Tensor:
    """Column-wise ``phi = 0`` crossings, in rows (NaN where none).

    ``phi`` is positive in the heavy phase, which sits **on top**, so the
    crossing of interest is the ascending one (``phi < 0`` below, ``phi >= 0``
    above).  Detecting the opposite sign change (the descending one) silently
    returns no crossings at all for this orientation -- a defect caught by
    ``tests/test_rayleigh_taylor_twofluid.py::test_interface_crossing_interpolates_the_zero_level``.
    """
    ny = phi.shape[0]
    above = phi[:-1, :]
    below = phi[1:, :]
    mask = (above < 0.0) & (below >= 0.0)
    rows = torch.arange(ny - 1, dtype=phi.dtype, device=phi.device).view(ny - 1, 1)
    # signed denominator: for the ascending crossing above-below is negative, so
    # clamping it at a small positive value (the obvious guard) turns the
    # interpolation into a 1e-12 division and returns ~-1e12 row positions
    denom = torch.where(mask, above - below, torch.ones_like(above))
    height = rows + above / denom  # crossing between row k and row k+1
    return torch.where(mask, height, torch.full_like(height, float("nan")))


def front_state(phi: torch.Tensor, cfg: SCMPRTConfig) -> dict:
    """Spike (heavy, descending) and bubble (light, rising) front heights.

    The spike tip is the **lowest** crossing, the bubble apex the **highest**.
    The pre-declared primary estimators are the quantiles ``q10`` (spike) and
    ``q90`` (bubble), which a single noisy column cannot move; the raw tips are
    reported as secondary.
    """
    h = interface_crossings(phi)
    flat = h[~torch.isnan(h)].flatten()
    if flat.numel() < 4:
        return {"valid": False}
    y0 = cfg.y_interface
    return {
        "valid": True,
        "spike_tip_lam": (float(flat.min()) - y0) / cfg.lam,
        "bubble_tip_lam": (float(flat.max()) - y0) / cfg.lam,
        "spike_q10_lam": (float(torch.quantile(flat, 0.10)) - y0) / cfg.lam,
        "bubble_q90_lam": (float(torch.quantile(flat, 0.90)) - y0) / cfg.lam,
        "n_columns": int(flat.numel()),
    }


def layzer_velocity(atwood: float, gravity: float, lam: float, *, c_2d: float = 1.0) -> dict:
    """Two-fluid Layzer-Goncharov terminal velocities (closed form)."""
    k = 2.0 * math.pi / lam
    u_b = math.sqrt(2.0 * atwood * gravity / ((1.0 + atwood) * c_2d * k))
    u_s = math.sqrt(2.0 * atwood * gravity / ((1.0 - atwood) * c_2d * k))
    return {"u_bubble": u_b, "u_spike": u_s, "asymmetry": u_s / u_b}


def _linear_fit(t: list[float], h: list[float]) -> tuple[float, float]:
    n = len(t)
    if n < 3:
        raise ValueError("need at least 3 samples")
    mt = sum(t) / n
    mh = sum(h) / n
    sxx = sum((ti - mt) ** 2 for ti in t)
    sxy = sum((ti - mt) * (hi - mh) for ti, hi in zip(t, h))
    slope = sxy / sxx
    intercept = mh - slope * mt
    ss_tot = sum((hi - mh) ** 2 for hi in h)
    ss_res = sum((hi - (intercept + slope * ti)) ** 2 for ti, hi in zip(t, h))
    return slope, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def run_scmp_rt(
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
    pos_window: tuple[float, float] = (0.15, 0.45),
    sub_windows: tuple[tuple[float, float], ...] = ((0.15, 0.30), (0.30, 0.45)),
) -> dict:
    """Run the finite-density-ratio RT case and measure the terminal front velocities.

    The terminal-velocity window is declared in **front position** (spike
    displacement in wavelengths), not in time: it spans the interval where the
    constant-velocity regime holds -- after the linear stage (``a >~ 0.15 lam``)
    and before the fronts compete with the far wall / the neighbouring mode
    (``a <~ 0.45 lam``, i.e. under a third of the free height).  The window is
    therefore fixed by the initial condition and the box, not by the result, and
    it is passed in by the caller.  ``tau = t sqrt(A g k)`` is reported for each
    sample so the correspondence to the dimensionless time is auditable.
    """
    cfg.validate()
    f, phi = init_scmp_rt(cfg, rho_high=rho_high, rho_low=rho_low)
    atwood = atwood_from_ratio(rho_high / rho_low)
    scale = math.sqrt(atwood * cfg.gravity * cfg.k)
    midpoint = 0.5 * (rho_high + rho_low)
    amplitude = 0.5 * (rho_high - rho_low)

    history: list[dict] = []
    mass0 = float(f.sum())
    stop_reason = "completed"
    step = 0
    for step in range(1, cfg.steps + 1):
        f = scmp_step(f, tau=cfg.tau_relax, sc_g=cfg.sc_g,
                      gravity_y=-cfg.gravity, psi_kind=cfg.psi_kind)
        if step % cfg.sample_every and step != cfg.steps:
            continue
        rho, ux, uy = velocity_field(f, sc_g=cfg.sc_g, gravity_y=-cfg.gravity,
                                     psi_kind=cfg.psi_kind)
        if not torch.isfinite(rho).all():
            stop_reason = "blowup"
            break
        phi = _mirror_wall_rows((rho - midpoint) / amplitude)
        speed = float(torch.sqrt(ux[1:-1, :] ** 2 + uy[1:-1, :] ** 2).max())
        row = {
            "step": step,
            "t": float(step),
            "tau": step * scale,
            "max_speed": speed,
            "ma": speed / math.sqrt(CS2),
            "mass_drift_pct": 100.0 * (float(f.sum()) - mass0) / mass0,
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
        }
        row.update(front_state(phi, cfg))
        history.append(row)
        if not row.get("valid", False):
            stop_reason = "interface_lost"
            break
        if row["ma"] > cfg.ma_limit:
            stop_reason = "mach"
            break
        if max(abs(row["bubble_tip_lam"]), abs(row["spike_tip_lam"])) > 0.5 * cfg.layers - 1.0:
            stop_reason = "wall"
            break

    ref = layzer_velocity(atwood, cfg.gravity, cfg.lam)

    def _window_fit(win: tuple[float, float], key: str) -> dict:
        """Linear fit of the front position over a position window (not time)."""
        sel = [
            r for r in history
            if r.get("valid") and win[0] <= abs(r["spike_tip_lam"]) <= win[1]
        ]
        if len(sel) < 5:
            return {"n": len(sel), "ok": False}
        slope, r2 = _linear_fit([r["t"] for r in sel], [r[key] * cfg.lam for r in sel])
        return {"n": len(sel), "ok": True, "u": slope, "r2": r2,
                "tau_span": [sel[0]["tau"], sel[-1]["tau"]]}

    primary = {
        "spike": _window_fit(pos_window, "spike_q10_lam"),
        "bubble": _window_fit(pos_window, "bubble_q90_lam"),
        "spike_tip": _window_fit(pos_window, "spike_tip_lam"),
        "bubble_tip": _window_fit(pos_window, "bubble_tip_lam"),
    }
    subs = {f"{w[0]:g}-{w[1]:g}": _window_fit(w, "spike_q10_lam") for w in sub_windows}

    u_spike = primary["spike"].get("u", float("nan"))
    u_bubble = primary["bubble"].get("u", float("nan"))
    asym = u_spike / u_bubble if u_bubble else float("nan")
    us = [v["u"] for v in subs.values() if v.get("ok")]
    drift = 100.0 * (max(us) - min(us)) / abs(u_spike) if (us and u_spike == u_spike) else float("nan")
    re_spike = u_spike * cfg.lam / cfg.nu if cfg.nu > 0 else float("nan")

    return {
        "name": cfg.name,
        "lam": cfg.lam,
        "layers": cfg.layers,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "sc_g": cfg.sc_g,
        "psi": cfg.psi_kind,
        "gravity": cfg.gravity,
        "tau_relax": cfg.tau_relax,
        "nu": cfg.nu,
        "a0_over_lam": cfg.a0_over_lam,
        "interface_width": cfg.interface_width,
        "rho_high": rho_high,
        "rho_low": rho_low,
        "density_ratio": rho_high / rho_low,
        "atwood_nominal": cfg.atwood,
        "atwood_measured": atwood,
        "steps_run": step,
        "stop_reason": stop_reason,
        "window_position_lam": list(pos_window),
        "window_tau_span": primary["spike"].get("tau_span", []),
        "u_spike_sim": u_spike,
        "u_bubble_sim": u_bubble,
        "u_spike_tip_sim": primary["spike_tip"].get("u", float("nan")),
        "u_bubble_tip_sim": primary["bubble_tip"].get("u", float("nan")),
        "u_spike_ref": ref["u_spike"],
        "u_bubble_ref": ref["u_bubble"],
        "asym_ref": ref["asymmetry"],
        "asym_sim": asym,
        "err_spike_pct": 100.0 * (u_spike / ref["u_spike"] - 1.0) if ref["u_spike"] else float("nan"),
        "err_bubble_pct": 100.0 * (u_bubble / ref["u_bubble"] - 1.0) if ref["u_bubble"] else float("nan"),
        "err_asym_pct": 100.0 * (asym / ref["asymmetry"] - 1.0) if asym == asym else float("nan"),
        "re_spike": re_spike,
        "fit_spike_r2": primary["spike"].get("r2", float("nan")),
        "fit_bubble_r2": primary["bubble"].get("r2", float("nan")),
        "terminal_drift_pct": drift,
        "ma_peak": max((r["ma"] for r in history), default=float("nan")),
        "mass_drift_pct": history[-1]["mass_drift_pct"] if history else float("nan"),
        "rho_min_end": history[-1]["rho_min"] if history else float("nan"),
        "rho_max_end": history[-1]["rho_max"] if history else float("nan"),
        "history": history,
    }