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
    "column_height",
    "column_front_state",
    "interface_profile_gradient",
    "gradient_front_state",
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
    # rows 0 and ny-1 are solid: they hold no mass, and the half-way bounce-back
    # discards whatever is put there.  Leaving them filled made the reported mass
    # drift read exactly -2/ny (-4.167 % at ny = 48) although the fluid mass is
    # conserved to machine precision -- a bookkeeping defect, not a leak.
    rho[0, :] = 0.0
    rho[-1, :] = 0.0
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


def column_height(
    rho: torch.Tensor,
    *,
    rho_high: float,
    rho_low: float,
) -> torch.Tensor:
    """Heavy-fluid column height per column, in rows (a conserved moment).

    ``h(x) = sum_y (rho - rho_low) / (rho_high - rho_low)`` over the free rows,
    which is the lattice counterpart of the heavy-fluid column height used by the
    Boussinesq arm of this case.  It has two properties the ``phi = 0`` crossing
    does not have (both measured, see the case report):

    * it is a **mass** moment, so it is immune to the hydrostatic density offset
      that gravity imposes on the whole box (``Delta rho / rho ~ g H / c_s^2``):
      a fixed threshold density drifts with that offset, which reads as a
      uniform descent of the interface;
    * it does not need a threshold, so a thin low-density layer against a wall
      cannot invent a spurious "interface" (the crossing estimator does exactly
      that once the vapour phase depletes at the bottom wall).
    """
    frac = (rho[1:-1, :] - rho_low) / (rho_high - rho_low)
    return frac.clamp(0.0, 1.0).sum(dim=0)


def column_front_state(
    rho: torch.Tensor,
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
    reference_row: float | None = None,
) -> dict:
    """Spike/bubble displacements from the column-height moment.

    The heavy phase sits **on top**, so a column that holds ``h`` rows of heavy
    fluid has its interface at row ``ny - 1 - h``.  Displacements are reported
    relative to ``reference_row`` (the t = 0 value, which absorbs both the diffuse
    interface half-width and the hydrostatic offset), in wavelengths: negative is
    the descending spike, positive the rising bubble.

    ``spike_q10`` / ``bubble_q90`` are the pre-declared robust estimators (a
    single bad column cannot move a quantile); the raw extrema are reported for
    audit.

    All reported displacements are measured **relative to the instantaneous mean
    interface**, not to a fixed row.  Under gravity the whole box compresses
    (``Delta rho / rho ~ g H / c_s^2``), which shifts the mean interface even when
    the perturbation does not move; referencing the mean cancels that common mode
    exactly, and it is also the frame in which the Layzer-Goncharov terminal
    velocity is defined.  The common mode itself is returned as
    ``common_mode_lam`` for audit.
    """
    is_mean_referenced = True
    h = column_height(rho, rho_high=rho_high, rho_low=rho_low)
    y_i = (cfg.ny - 1) - h
    ref = float(y_i.mean()) if reference_row is None else float(reference_row)
    d = (y_i - ref) / cfg.lam
    x = torch.arange(cfg.nx, dtype=d.dtype)
    phase = torch.cos(2.0 * math.pi * x / cfg.nx)
    amp = 2.0 * float(((y_i - y_i.mean()) * phase).sum()) / cfg.nx / cfg.lam
    mean = float(d.mean())
    d_rel = d - mean
    return {
        "valid": True,
        "reference_row": ref,
        "mean_interface_lam": mean,
        "common_mode_lam": mean,
        "mean_referenced": is_mean_referenced,
        "spike_tip_lam": float(d_rel.min()),
        "bubble_tip_lam": float(d_rel.max()),
        "spike_q10_lam": float(torch.quantile(d_rel, 0.10)),
        "bubble_q90_lam": float(torch.quantile(d_rel, 0.90)),
        "amplitude_lam": amp,
        "n_columns": int(d.numel()),
    }


def interface_profile_gradient(
    rho: torch.Tensor,
    cfg: SCMPRTConfig,
    *,
    wall_guard: int = 3,
    search_half_width_lam: float = 0.6,
    lock_half_width_cells: int | None = None,
) -> dict:
    """Per-column interface row from the **density-gradient centroid**.

    Why this estimator exists (measured defect, mission m22097)
    -----------------------------------------------------------
    :func:`column_height` is a *conserved mass moment*: it sums the clamped heavy
    fraction over the free rows.  For an interface displacement that is (to
    leading order) a vertical shift of the density profile, that sum is almost
    invariant -- a shifted copy of the same 1-D profile occupies the same number
    of equivalent rows, and only the wall-cut terms change.  Measured on a
    relaxed SCMP interface (``artifacts_estimator/calibration.json``, check K1):
    imposing a known ``a0 = 0.02 lam`` cosine (1.28 rows) changed the per-column
    moment by a spread of **0.063 rows**, i.e. 2.4 % of the imposed +/-1.28 row
    signal, and the fitted amplitude came back as ``5.4e-5 lam``
    (error ``-99.7 %``).  The same blindness was recorded independently in
    ``docs/CASE-RT-TWOFLUID-PRECOND.md`` (P4: ``-0.0021 lam`` against a planned
    ``+0.02 lam``) -- its guessed mechanism (``clamp`` saturation) is only a
    secondary aggravation: in a synthetic profile whose bulk density exceeds
    ``rho_high`` by 36 % the moment still tracked the shift to within ``-0.9 %``.
    A conserved moment is simply the wrong functional for a *position*.

    The replacement is the standard interface-tracking estimator: per column,
    lock onto the row of maximum ``|d rho / d y|`` inside a search window (walls
    excluded), then take the ``|d rho / d y|``-weighted centroid over a narrow
    band around that lock.  It is (i) position sensitive by construction,
    (ii) insensitive to a bulk density offset because only the *shape* of the
    gradient enters, and (iii) threshold free, so the low-density layer that
    collects against a wall cannot invent a second interface.

    Returns ``y_rows`` (per column, lattice rows) plus the lock rows, the search
    window and two quality indicators.  All returned values are floats/ints --
    the dict is safe to serialise into a run history.
    """
    ny, nx = rho.shape
    dtype = rho.dtype
    if ny < 2 * wall_guard + 8:
        raise ValueError("box too short for the wall guard")
    deriv = torch.zeros_like(rho)
    deriv[1:-1, :] = 0.5 * (rho[2:, :] - rho[:-2, :])
    weight = deriv.abs()

    half = search_half_width_lam * cfg.lam
    lo = max(wall_guard, int(math.floor(cfg.y_interface - half)))
    hi = min(ny - 1 - wall_guard, int(math.ceil(cfg.y_interface + half)) + 1)
    if hi - lo < 5:
        raise ValueError("search window collapsed: increase layers or reduce the guard")

    lock = torch.argmax(weight[lo:hi, :], dim=0) + lo          # (nx,)
    # sub-row refinement: parabolic vertex of |d rho / d y| around the lock.
    # Without it the centroid below is biased towards the integer lock row
    # (measured -11 % at a = 0.005 lam on an exact tanh interface).
    lc = lock.clamp(min=lo + 1, max=hi - 2)
    w_km1 = weight.gather(0, (lc - 1).view(1, nx)).view(nx)
    w_k = weight.gather(0, lc.view(1, nx)).view(nx)
    w_kp1 = weight.gather(0, (lc + 1).view(1, nx)).view(nx)
    curv = w_km1 - 2.0 * w_k + w_kp1
    delta = torch.where(
        curv.abs() > 1e-30,
        0.5 * (w_km1 - w_kp1) / torch.where(curv.abs() > 1e-30, curv, torch.ones_like(curv)),
        torch.zeros_like(curv),
    ).clamp(-0.5, 0.5)
    lock_f = lock.to(dtype) + delta                            # fractional lock row
    lw = lock_half_width_cells
    if lw is None:
        lw = max(2, int(round(2.0 * cfg.interface_width)))
    idx = torch.arange(ny, dtype=dtype).view(ny, 1)
    band = (idx >= (lock_f - lw).clamp(min=wall_guard)) & (
        idx <= (lock_f + lw).clamp(max=ny - 1 - wall_guard)
    )
    wl = weight * band
    denom = wl.sum(dim=0)
    ok = denom > 1e-30
    y_rows = (wl * idx).sum(dim=0) / denom.clamp_min(1e-30)
    peaks = weight.gather(0, lock.view(1, nx)).view(nx)
    edge = (lock <= lo) | (lock >= hi - 1)
    return {
        "valid": bool(ok.all()),
        "y_rows": y_rows,
        "lock_rows": lock,
        "search_rows": (int(lo), int(hi)),
        "lock_spread_rows": int(lock.max()) - int(lock.min()),
        "peak_min": float(peaks.min()),
        "peak_max": float(peaks.max()),
        "peak_ratio": float(peaks.min()) / float(peaks.max()) if float(peaks.max()) else 0.0,
        "n_wall_locked": int(edge.sum()),
    }


def gradient_front_state(
    rho: torch.Tensor,
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
    reference_row: float | None = None,
    wall_guard: int = 3,
    search_half_width_lam: float = 0.6,
    lock_half_width_cells: int | None = None,
) -> dict:
    """Spike/bubble displacements from :func:`interface_profile_gradient`.

    Same contract as :func:`column_front_state` (displacements in wavelengths,
    negative = descending spike, all reported relative to the **instantaneous**
    mean interface so the hydrostatic common mode cancels), same key names, so
    the two are drop-in interchangeable in :func:`run_scmp_rt`.  The estimated
    absolute level is returned as ``mean_interface_lam`` relative to
    ``reference_row`` (default: the geometric interface row).
    """
    del rho_high, rho_low  # the gradient estimator is density-scale free
    prof = interface_profile_gradient(
        rho,
        cfg,
        wall_guard=wall_guard,
        search_half_width_lam=search_half_width_lam,
        lock_half_width_cells=lock_half_width_cells,
    )
    if not prof["valid"]:
        return {"valid": False, "estimator": "gradient"}
    y = prof["y_rows"]
    nx = int(y.numel())
    mean = float(y.mean())
    ref = float(cfg.y_interface) if reference_row is None else float(reference_row)
    d = (y - mean) / cfg.lam
    x = torch.arange(nx, dtype=y.dtype)
    phase = torch.cos(2.0 * math.pi * x / nx)
    amp = 2.0 * float(((y - mean) * phase).sum()) / nx / cfg.lam
    return {
        "valid": True,
        "estimator": "gradient",
        "reference_row": ref,
        "mean_interface_lam": (mean - ref) / cfg.lam,
        "common_mode_lam": (mean - ref) / cfg.lam,
        "mean_referenced": True,
        "spike_tip_lam": float(d.min()),
        "bubble_tip_lam": float(d.max()),
        "spike_q10_lam": float(torch.quantile(d, 0.10)),
        "bubble_q90_lam": float(torch.quantile(d, 0.90)),
        "amplitude_lam": amp,
        "n_columns": nx,
        "grad_lock_spread_rows": prof["lock_spread_rows"],
        "grad_peak_ratio": prof["peak_ratio"],
        "grad_n_wall_locked": prof["n_wall_locked"],
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


def _interface_band(cfg: SCMPRTConfig, cells: int) -> torch.Tensor:
    """Boolean ``(ny, nx)`` mask of the row band straddling the flat interface."""
    band = torch.zeros((cfg.ny, cfg.nx), dtype=torch.bool)
    lo = max(1, int(cfg.y_interface) - cells)
    hi = min(cfg.ny - 1, int(cfg.y_interface) + cells + 1)
    band[lo:hi, :] = True
    return band


def _ma_split(
    ux: torch.Tensor,
    uy: torch.Tensor,
    band: torch.Tensor,
) -> tuple[float, float, float]:
    """``(ma_all, ma_iface, ma_bulk)``; walls and the interface band excluded
    from the bulk metric.  ``ma_* = |u|_max / c_s``."""
    speed = torch.sqrt(ux * ux + uy * uy)
    fluid = torch.ones_like(band)
    fluid[0, :] = False
    fluid[-1, :] = False
    inv_cs = 1.0 / math.sqrt(CS2)
    ma_all = float(speed[fluid].max()) * inv_cs
    iface = fluid & band
    bulk = fluid & ~band
    ma_iface = float(speed[iface].max()) * inv_cs if bool(iface.any()) else 0.0
    ma_bulk = float(speed[bulk].max()) * inv_cs if bool(bulk.any()) else 0.0
    return ma_all, ma_iface, ma_bulk


def _flat_state(
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
) -> torch.Tensor:
    """Equilibrium populations of a **flat** (unperturbed) two-phase slab."""
    dtype, device = cfg.dtype, torch.device("cpu")
    y = torch.arange(cfg.ny, dtype=dtype, device=device).view(cfg.ny, 1)
    half = 0.5 * (rho_high + rho_low)
    amp = 0.5 * (rho_high - rho_low)
    rho = half + amp * torch.tanh((y - cfg.y_interface) / cfg.interface_width)
    rho = rho.expand(cfg.ny, cfg.nx).contiguous()
    rho[0, :] = 0.0
    rho[-1, :] = 0.0
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


def _perturb_density(
    rho: torch.Tensor,
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
) -> torch.Tensor:
    """Displace the interface by one cosine mode of amplitude ``a0_over_lam*lam``.

    The displacement is applied by **sampling the relaxed 1-D profile at
    ``y - d(x)``** instead of writing an ideal ``tanh`` back into the box.  An
    ideal profile at a hand-picked width is *not* the model's equilibrium profile
    (the measured equilibrium interface is wider than ``interface_width``), so
    re-writing it would re-introduce exactly the start-up impulse that the
    preconditioning step removes.  ``rho_high``/``rho_low`` are kept in the
    signature for API symmetry with :func:`precondition_scmp_rt` and are not used.
    """
    del rho_high, rho_low
    ny, nx = rho.shape
    prof = rho.mean(dim=1)                      # relaxed profile, averaged over x
    y_idx = torch.arange(ny, dtype=rho.dtype).view(ny, 1)
    x_idx = torch.arange(nx, dtype=rho.dtype).view(1, nx)
    a0 = cfg.a0_over_lam * cfg.lam
    d = a0 * torch.cos(2.0 * math.pi * x_idx / cfg.lam)
    src = (y_idx - d).clamp(0.0, float(ny - 1))
    i0 = src.floor().clamp(0.0, float(ny - 2)).long()
    frac = src - i0.to(src.dtype)
    out = prof[i0] * (1.0 - frac) + prof[(i0 + 1).clamp(max=ny - 1)] * frac
    out[0, :] = 0.0
    out[-1, :] = 0.0
    return out


def precondition_scmp_rt(
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
    relax_steps: int = 800,
    sample_every: int = 100,
) -> tuple[torch.Tensor, dict]:
    """Relax a **flat** interface at ``g = 0``, then seed the single mode.

    Why this exists (measured, not assumed)
    ---------------------------------------
    ``init_scmp_rt`` puts a hand-picked ``tanh(width)`` interface into the box and
    switches gravity on at the same time.  The initial profile is then *not* the
    model's equilibrium profile, so the first steps carry a force impulse
    ``|F|/rho`` that the case report (mission m21799) recorded as a spurious
    ``Ma`` of 0.26 / 0.30 at step 5 -- above the 0.20 guard, which aborted both
    finite-density-ratio runs before the interface moved at all
    (``stop_reason = "mach"``, ``steps_run = 5``).

    The staged protocol removes the two causes separately:

    1. relax the flat interface with gravity **off** until the pseudopotential
       spurious current has settled (its floor is returned in ``info``);
    2. rebuild the populations from the relaxed density field (velocity reset to
       zero, so the residual pseudo-momentum of step 1 is dropped) and *then*
       displace the interface by the single mode;
    3. the caller ramps gravity in over ``ramp_steps`` (see ``run_scmp_rt``).

    Returns ``(f, info)`` with ``info`` holding the spurious-current decay curve.
    """
    f = _flat_state(cfg, rho_high=rho_high, rho_low=rho_low)
    trace: list[dict] = []
    band = _interface_band(cfg, int(round(3.0 * cfg.interface_width)))
    mass0 = float(f.sum())
    for step in range(1, relax_steps + 1):
        f = scmp_step(f, tau=cfg.tau_relax, sc_g=cfg.sc_g, gravity_y=0.0,
                      psi_kind=cfg.psi_kind)
        if step % sample_every and step != relax_steps:
            continue
        rho, ux, uy = velocity_field(f, sc_g=cfg.sc_g, gravity_y=0.0,
                                     psi_kind=cfg.psi_kind)
        ma_all, ma_iface, ma_bulk = _ma_split(ux, uy, band)
        trace.append({
            "step": step,
            "ma_all": ma_all,
            "ma_iface": ma_iface,
            "ma_bulk": ma_bulk,
            "mass_drift_pct": 100.0 * (float(f.sum()) - mass0) / mass0,
            "rho_min": float(rho[1:-1].min()),
            "rho_max": float(rho[1:-1].max()),
        })
    rho_relaxed = f.sum(dim=0)
    rho_pert = _perturb_density(rho_relaxed, cfg, rho_high=rho_high, rho_low=rho_low)
    zero = torch.zeros_like(rho_pert)
    f = equilibrium(rho_pert, zero, zero)
    info = {
        "relax_steps": relax_steps,
        "spurious_ma_floor": trace[-1]["ma_all"] if trace else float("nan"),
        "spurious_ma_iface_floor": trace[-1]["ma_iface"] if trace else float("nan"),
        "spurious_ma_peak": max((r["ma_all"] for r in trace), default=float("nan")),
        "relax_mass_drift_pct": trace[-1]["mass_drift_pct"] if trace else float("nan"),
        "rho_high_relaxed": trace[-1]["rho_max"] if trace else float("nan"),
        "rho_low_relaxed": trace[-1]["rho_min"] if trace else float("nan"),
        "relax_trace": trace,
    }
    return f, info


def run_scmp_rt(
    cfg: SCMPRTConfig,
    *,
    rho_high: float,
    rho_low: float,
    pos_window: tuple[float, float] = (0.15, 0.45),
    sub_windows: tuple[tuple[float, float], ...] = ((0.15, 0.30), (0.30, 0.45)),
    precondition_steps: int = 0,
    ramp_steps: int = 0,
    ma_guard: str = "all",
    iface_band_cells: int | None = None,
    gravity_scale: float = 1.0,
    front_estimator: str = "column",
    amp_window: tuple[float, float] = (0.03, 0.08),
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

    Staged start-up (``precondition_steps`` / ``ramp_steps`` / ``ma_guard``)
    ----------------------------------------------------------------------
    With the defaults (``0``, ``0``, ``"all"``) this is the original single-shot
    start-up: ``init_scmp_rt`` profile plus gravity from the first step, guarded
    on the *maximum* Mach number of the whole fluid.  That combination aborted
    the case runs at step 5 (see ``precondition_scmp_rt``).  The staged options
    are opt-in:

    ``precondition_steps > 0``  relax a flat interface at ``g = 0`` first, reset
        the velocity, then seed the single mode (``precondition_scmp_rt``);
    ``ramp_steps > 0``          ramp gravity linearly from 0 to its value over
        the first ``ramp_steps`` steps;
    ``ma_guard``               ``"all"`` (legacy), ``"iface"`` or ``"bulk"`` --
        which of the three reported Mach numbers is the stopping criterion.
        ``ma_iface`` is dominated by the pseudopotential spurious current at the
        interface and is therefore reported but not recommended as a guard;
        ``ma_bulk`` is the physical compressibility check.
    ``gravity_scale``          multiplies ``cfg.gravity``; ``0.0`` gives the
        *null-dynamics control* (perturbed interface, no gravity), whose measured
        front drift is the sensitivity floor of the estimator.
    """
    cfg.validate()
    if ma_guard not in ("all", "iface", "bulk"):
        raise ValueError(f"ma_guard must be 'all', 'iface' or 'bulk', got {ma_guard!r}")
    if front_estimator not in ("column", "crossings", "gradient"):
        raise ValueError(
            "front_estimator must be 'column', 'crossings' or 'gradient', got "
            f"{front_estimator!r}"
        )
    band = _interface_band(cfg, iface_band_cells if iface_band_cells is not None
                           else int(round(3.0 * cfg.interface_width)))
    precondition_info: dict = {}
    if precondition_steps > 0:
        f, precondition_info = precondition_scmp_rt(
            cfg, rho_high=rho_high, rho_low=rho_low, relax_steps=precondition_steps
        )
        phi = _mirror_wall_rows((f.sum(dim=0) - 0.5 * (rho_high + rho_low))
                                / (0.5 * (rho_high - rho_low)))
    else:
        f, phi = init_scmp_rt(cfg, rho_high=rho_high, rho_low=rho_low)
    atwood = atwood_from_ratio(rho_high / rho_low)
    if front_estimator == "gradient":
        # reference level of the gradient estimator: its own t = 0 mean row, so
        # the reported common mode is the drift of the interface *level* and not
        # an artefact of the estimator's offset convention
        ref_row = float(
            interface_profile_gradient(f.sum(dim=0), cfg)["y_rows"].mean()
        )
    else:
        ref_row = float(
            ((cfg.ny - 1) - column_height(f.sum(dim=0), rho_high=rho_high,
                                          rho_low=rho_low)).mean()
        )
    scale = math.sqrt(atwood * cfg.gravity * cfg.k)
    midpoint = 0.5 * (rho_high + rho_low)
    amplitude = 0.5 * (rho_high - rho_low)

    history: list[dict] = []
    mass0 = float(f.sum())
    stop_reason = "completed"
    step = 0
    for step in range(1, cfg.steps + 1):
        ramp = 1.0 if ramp_steps <= 0 else min(1.0, step / float(ramp_steps))
        g_now = -cfg.gravity * gravity_scale * ramp
        f = scmp_step(f, tau=cfg.tau_relax, sc_g=cfg.sc_g,
                      gravity_y=g_now, psi_kind=cfg.psi_kind)
        if step % cfg.sample_every and step != cfg.steps:
            continue
        rho, ux, uy = velocity_field(f, sc_g=cfg.sc_g, gravity_y=g_now,
                                     psi_kind=cfg.psi_kind)
        if not torch.isfinite(rho).all():
            stop_reason = "blowup"
            break
        phi = _mirror_wall_rows((rho - midpoint) / amplitude)
        state_col = column_front_state(rho, cfg, rho_high=rho_high,
                                       rho_low=rho_low, reference_row=ref_row)
        state_xc = front_state(phi, cfg)
        if front_estimator == "gradient":
            primary = gradient_front_state(rho, cfg, rho_high=rho_high,
                                           rho_low=rho_low, reference_row=ref_row)
            secondary = state_col
        elif front_estimator == "column":
            primary, secondary = state_col, state_xc
        else:
            primary, secondary = state_xc, state_col
        ma_all, ma_iface, ma_bulk = _ma_split(ux, uy, band)
        speed = max(ma_all, 0.0) * math.sqrt(CS2)
        row = {
            "step": step,
            "t": float(step),
            "tau": step * scale,
            "max_speed": speed,
            "ma": ma_all,
            "ma_all": ma_all,
            "ma_iface": ma_iface,
            "ma_bulk": ma_bulk,
            "gravity_y": float(g_now),
            "mass_drift_pct": 100.0 * (float(f.sum()) - mass0) / mass0,
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
        }
        row.update({k: v for k, v in primary.items() if k != "valid"})
        row.update({f"alt_{k}": v for k, v in secondary.items() if k != "valid"})
        row["valid"] = bool(primary.get("valid", False)) \
            if front_estimator in ("crossings", "gradient") else True
        history.append(row)
        if not row.get("valid", False):
            stop_reason = "interface_lost"
            break
        guard_ma = {"all": ma_all, "iface": ma_iface, "bulk": ma_bulk}[ma_guard]
        if guard_ma > cfg.ma_limit:
            stop_reason = "mach"
            break
        if max(abs(row["bubble_tip_lam"]), abs(row["spike_tip_lam"])) > 0.5 * cfg.layers - 1.0:
            stop_reason = "wall"
            break

    # linear growth rate of the column-height mode (the anchor quantity, linear
    # regime) against the equal-viscosity dispersion relation
    lin = [r for r in history
           if r.get("amplitude_lam") is not None
           and amp_window[0] <= abs(r["amplitude_lam"]) <= amp_window[1]]
    nu = cfg.nu
    k = cfg.k
    sigma_theory = math.sqrt(atwood * cfg.gravity * k + nu * nu * k ** 4) - nu * k * k
    sigma_amp = float("nan")
    sigma_r2 = float("nan")
    if len(lin) >= 3:
        try:
            sigma_amp, sigma_r2 = _linear_fit(
                [r["t"] for r in lin], [math.log(abs(r["amplitude_lam"])) for r in lin]
            )
        except ValueError:
            pass

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
        "precondition_steps": precondition_steps,
        "ramp_steps": ramp_steps,
        "ma_guard": ma_guard,
        "gravity_scale": gravity_scale,
        "precondition": precondition_info,
        "front_estimator": front_estimator,
        "reference_row": ref_row,
        "amp_window_lam": list(amp_window),
        "sigma_amp": sigma_amp,
        "sigma_amp_r2": sigma_r2,
        "sigma_theory_visc": sigma_theory,
        "sigma_theory_inv": math.sqrt(atwood * cfg.gravity * k),
        "err_sigma_pct": (100.0 * (sigma_amp / sigma_theory - 1.0)
                          if sigma_amp == sigma_amp and sigma_theory else float("nan")),
        "n_amp_fit": len(lin),
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
        "ma_peak_iface": max((r["ma_iface"] for r in history), default=float("nan")),
        "ma_peak_bulk": max((r["ma_bulk"] for r in history), default=float("nan")),
        "ma_peak_guard": ma_guard,
        "mass_drift_pct": history[-1]["mass_drift_pct"] if history else float("nan"),
        "rho_min_end": history[-1]["rho_min"] if history else float("nan"),
        "rho_max_end": history[-1]["rho_max"] if history else float("nan"),
        "history": history,
    }