"""Force-driven, wall-resolved 3-D rough channel in the lattice-Boltzmann method.

Purpose
-------
Measure the roughness function ``dU+(ks+)`` of a *voxelised* k-type roughness
without any wall model: the roughness elements of
:mod:`tensorlbm.roughness_elements` are literal solid voxels in a D3Q19
channel, the collision is two-relaxation-time (TRT), the walls are halfway
bounce-back and the drive is a constant body force.  Everything that is
reported is measured from the populations, not imposed.

Why this is the right closure for the ship problem
--------------------------------------------------
For a force-driven rough channel the global momentum balance is exact and
independent of the flow details::

    tau_w = rho * g * h_eff,   h_eff = N_fluid / (2 nx ny)

so ``u_tau = sqrt(g h_eff)`` follows from the geometry to machine precision and
the friction coefficient from the bulk velocity alone, ``Cf = 2 (u_tau/U)^2``.
Choosing ``g`` from the geometry (``g = u_tau^2/h_eff``) therefore gives a
matched-``u_tau`` (Clauser) comparison between the smooth and the rough wall —
exactly the convention in which ``dU+`` and the ITTC-78 allowance are defined.
No momentum-exchange bookkeeping is needed, so the result cannot be biased by a
force-integration error.

Roughness function definitions (both reported)
----------------------------------------------
``dU_plus_profile``
    Log-law intercept shift, ``dU+ = B_smooth - b_rough`` with the log law
    ``U+ = (1/kappa) ln(y+) + b`` fitted over ``y+ in [max(3 ks+, 30), 0.35 h+]``
    with ``kappa = 0.41`` imposed.  This is the definition used by the
    sand-grain correlations and it requires a genuine log layer
    (``3 ks+ < 0.35 h+``); each run records whether that window exists.
``dU_plus_bulk``
    Hama/Clauser bulk shift ``U+_bulk,smooth - U+_bulk,rough`` at matched
    ``u_tau`` and the same ``h_eff`` convention.  It stays defined when the
    roughness is too large for a log layer (large ``ks+/h+``) and is the
    fallback reported for the extreme end of a sweep.

Numerics
--------
* D3Q19 TRT collision (Ginzburg 2005) with the magic parameter
  ``Lambda = 3/16``; this is stable at the near-0.5 relaxation time that a
  high-``Re_tau``, low-Mach rough channel inevitably has, where plain BGK
  diverges.
* Guo (2002) forcing in the post-collision split, ``(1 - s+/2)`` factor.
* Halfway bounce-back implemented **link-wise**: a fluid cell that lacks a
  fluid neighbour along direction ``i`` receives ``f*_opp(i)`` of its own
  post-collision state.  No population is ever stored in a solid node, which
  makes the mass balance exact (checked in the test-suite) and removes the
  wall-node drift that plagues "reflect-what-arrived" implementations.
* Streaming is periodic in ``x``/``y``; ``z`` is closed by the two solid wall
  layers, so no outflow boundary condition is exercised.
* ``torch.compile`` is used when available (≈10x on CPU); the step is a pure
  function of ``(f, solid, links, g, tau, s-)`` so compiled and eager paths
  differ only by floating-point association order.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:  # torch is optional: the native C back-end only needs numpy
    import torch
except Exception as _torch_error:  # pragma: no cover - exercised on the loong64 nodes
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _torch_error
else:
    _TORCH_IMPORT_ERROR = None

# D3Q19 lattice mirrored from :mod:`tensorlbm.d3q19` as plain numpy arrays so
# that this module (and the C back-end driver) can be imported where torch is
# not available -- the native sweep runs on the loong64 nodes, whose torch build
# is broken.  ``tests/test_rough_channel3d.py`` asserts the constants below are
# identical to ``tensorlbm.d3q19``.
C = np.array(
    [
        [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
    ],
    dtype=np.int64,
)
W = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12)
OPPOSITE = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)
from .roughness_elements import RoughnessElementSet, channel_solid_mask, fluid_half_height
from .roughness_laws import B_SMOOTH, KAPPA, fit_log_law

__all__ = [
    "RoughChannelConfig",
    "run_rough_channel",
    "smooth_channel_reference",
    "build_link_masks",
    "TRT_LAMBDA",
]

_LATTICE: dict[str, Any] = {}


def _tensors() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily materialise the D3Q19 tensors (torch is optional for C-only runs)."""
    if not _LATTICE:
        if torch is None:  # pragma: no cover
            raise RuntimeError(
                "the torch back-end is unavailable (import failed: "
                f"{_TORCH_IMPORT_ERROR}); use tensorlbm.rough_channel3d_native instead"
            )
        c = torch.from_numpy(C)
        _LATTICE["cx"] = c[:, 0].view(19, 1, 1, 1).float()
        _LATTICE["cy"] = c[:, 1].view(19, 1, 1, 1).float()
        _LATTICE["cz"] = c[:, 2].view(19, 1, 1, 1).float()
        _LATTICE["w"] = torch.from_numpy(W).view(19, 1, 1, 1).float()
        _LATTICE["opp"] = torch.from_numpy(OPPOSITE).long()
    return _LATTICE["cx"], _LATTICE["cy"], _LATTICE["cz"], _LATTICE["w"], _LATTICE["opp"]
#: Streaming shifts as plain Python ints (keeps torch.compile graph-break free).
_SHIFTS: tuple[tuple[int, int, int], ...] = tuple(
    (int(C[i, 0]), int(C[i, 1]), int(C[i, 2])) for i in range(19)
)
#: Two-relaxation-time "magic" parameter (Ginzburg); stable down to tau -> 1/2.
TRT_LAMBDA: float = 3.0 / 16.0


@dataclass(frozen=True)
class RoughChannelConfig:
    """Numerical and physical settings of one rough-channel case.

    ``re_tau`` and ``u_tau`` are prescribed on the *fluid* half-height
    ``h_eff`` (see :func:`tensorlbm.roughness_elements.fluid_half_height`), so
    all cases on the same grid share a friction velocity even though their
    solid fractions differ.
    """

    nx: int = 64
    ny: int = 64
    nz: int = 130
    re_tau: float = 600.0
    u_tau: float = 0.008
    steps: int = 20000
    warmup: int = 5000
    sample_interval: int = 10
    seed: int = 20261107
    threads: int = 8
    use_compile: bool = True
    mirror_to_both_walls: bool = True
    noise_amplitude: float = 0.02
    #: Smagorinsky constant of the sub-grid model (0 disables it).  The sweep
    #: grids are coarse in wall units (dy+ ~ 8-13), so an SGS model is both the
    #: physically honest choice and what keeps the grid-scale modes damped at a
    #: relaxation time a hair above 1/2.
    smagorinsky_cs: float = 0.10
    #: Multiplier on the initial log-law profile.  Starting below the steady
    #: state (e.g. 0.3) avoids the large startup shear that can trip a
    #: near-0.5 relaxation time before the turbulence is established.
    init_scale: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.nz < 6 or self.nx < 8 or self.ny < 8:
            raise ValueError("channel too small: need nx,ny >= 8 and nz >= 6")
        if self.re_tau <= 0.0 or self.u_tau <= 0.0:
            raise ValueError("re_tau and u_tau must be positive")
        if self.warmup <= 0 or self.steps <= self.warmup:
            raise ValueError("need steps > warmup > 0")
        if self.sample_interval < 1:
            raise ValueError("sample_interval must be >= 1")


# ---------------------------------------------------------------------------
# Lattice kernels (module level so torch.compile can cache them)
# ---------------------------------------------------------------------------


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
    """D3Q19 equilibrium from ``(nz, ny, nx)`` macroscopic fields -> ``(19, nz, ny, nx)``."""
    cx, cy, cz, w, _opp = _tensors()
    cu = cx * ux + cy * uy + cz * uz
    usq = ux * ux + uy * uy + uz * uz
    return w * (rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq))


def _stream(f: torch.Tensor) -> torch.Tensor:
    """Streaming with periodic x/y; zero fill in z (the z ends are solid)."""
    nz = f.shape[1]
    padded = torch.zeros((19, nz + 2, f.shape[2], f.shape[3]), dtype=f.dtype, device=f.device)
    padded[:, 1:-1] = f
    out = []
    for i, (cx, cy, cz) in enumerate(_SHIFTS):
        slab = padded[i, 1 - cz : 1 - cz + nz]
        if cy:
            slab = torch.roll(slab, cy, dims=-2)
        if cx:
            slab = torch.roll(slab, cx, dims=-1)
        out.append(slab)
    return torch.stack(out)


def _lbm_step(
    f: torch.Tensor,
    solid_f: torch.Tensor,
    links_f: torch.Tensor,
    gx: torch.Tensor,
    tau: float,
    s_minus: float,
    smag2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One D3Q19 TRT step with Guo forcing and link-wise halfway bounce-back.

    ``solid_f`` is ``1.0`` in solid cells (float mask), ``links_f`` is the
    ``(19, nz, ny, nx)`` float mask of fluid->solid links.  Returns
    ``(f_next, ux, rho)`` with the pre-collision moments reused by the sampler.
    """
    cx, cy, cz, w, opp = _tensors()
    rho = f.sum(0)
    rho_safe = rho + solid_f  # solid cells are held at rest rho = 1
    mx = (f * cx).sum(0) + 0.5 * gx * rho
    my = (f * cy).sum(0)
    mz = (f * cz).sum(0)
    ux = mx / rho_safe
    uy = my / rho_safe
    uz = mz / rho_safe
    feq = _equilibrium(rho_safe, ux, uy, uz)
    cu = cx * ux + cy * uy + cz * uz
    ug = ux * gx
    # Guo (2002): F_i = w_i rho [3 (c_i - u).g + 9 (c_i.u)(c_i.g)]
    force = w * (rho_safe * (3.0 * (cx * gx - ug) + 9.0 * cu * (cx * gx)))
    # Smagorinsky SGS: local effective relaxation time from the non-equilibrium
    # stress.  Identical to the C back-end.
    s_plus = 1.0 / tau
    s_minus_eff = s_minus
    if smag2 > 0.0:
        fneq = f - feq
        pxx = (cx * cx * fneq).sum(0)
        pyy = (cy * cy * fneq).sum(0)
        pzz = (cz * cz * fneq).sum(0)
        pxy = (cx * cy * fneq).sum(0)
        pxz = (cx * cz * fneq).sum(0)
        pyz = (cy * cz * fneq).sum(0)
        pipi = pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)
        # Hou et al. (1996) Smagorinsky closure, identical to the C back-end:
        # tau_eff = 1/2 [tau_0 + sqrt(tau_0^2 + 54 Cs^2 sqrt(Pi:Pi)/(rho tau_0))]
        s2 = torch.sqrt(pipi.clamp_min(0.0))
        tau_eff = 0.5 * (tau + torch.sqrt((tau * tau + 54.0 * smag2 * s2 / (rho_safe * tau)).clamp_min(0.0)))
        s_plus = 1.0 / tau_eff
        s_minus_eff = 1.0 / (TRT_LAMBDA / (tau_eff - 0.5) + 0.5)
    # TRT: symmetric / antisymmetric parts relax at s+ and s-.
    f_opp = f[opp]
    feq_opp = feq[opp]
    f = (
        f
        - 0.5 * (s_plus + s_minus_eff) * (f - feq)
        - 0.5 * (s_plus - s_minus_eff) * (f_opp - feq_opp)
        # (1 - s_minus/2): the equilibrium is subtracted in both TRT modes, so
        # the antisymmetric mode already carries -s_minus F/2 of non-equilibrium
        # momentum; weighting the force with s_plus leaves the channel driven by
        # F [1 + (s_minus - s_plus)/2] instead of F (a factor ~2 at s- ~ 0.5 and
        # ~0.03 at tau -> 0.5).  Matches the C back-end.
        + force * (1.0 - 0.5 * s_minus_eff)
    )
    fluid_f = 1.0 - solid_f
    f = f * fluid_f  # no populations are stored in solid nodes
    # link-wise halfway bounce-back: a fluid cell missing direction i receives
    # its own post-collision opposite population.
    f = _stream(f) + f[opp] * links_f
    f = f * fluid_f
    return f, ux, rho


def build_link_masks(solid: np.ndarray) -> np.ndarray:
    """``links[i](X) = 1`` where ``X`` is fluid and ``X - c_i`` is solid.

    Those are exactly the *incoming* populations of a fluid cell that have no
    fluid source: after streaming they are zero and must be replaced by
    ``f*_opp(i)(X)`` (halfway bounce-back).

    Returned as ``float32`` ``(19, nz, ny, nx)`` so the bounce-back can be a
    single broadcast multiply (torch.compile friendly).  Rolls wrap in x/y,
    which is harmless: a wrapped entry always lands on a solid destination and
    is masked out by ``~solid``.
    """
    s = np.asarray(solid, dtype=bool)
    if s.ndim != 3:
        raise ValueError("solid mask must be 3-D (nz, ny, nx)")
    nz = s.shape[0]
    links = np.zeros((19,) + s.shape, dtype=np.float32)
    for i, (cx, cy, cz) in enumerate(_SHIFTS):
        # nb[X] = solid[X - c]: np.roll(a, k)[i] = a[i - k]
        nb = np.roll(np.roll(np.roll(s, cz, axis=0), cy, axis=1), cx, axis=2)
        links[i] = (~s) & nb
    # The z-roll wraps through the domain; anything it produced at k=0/nz-1 is
    # in a solid cell and therefore already excluded by (~s).
    del nz
    return links


def _make_stepper(cfg: RoughChannelConfig):
    """Compile the step when possible; verify the compiled call eagerly.

    ``torch.compile`` failures are lazy (they surface on the first call, not at
    wrap time), so the candidate is exercised once and silently discarded on
    any error — the eager path must always be available.
    """
    if not cfg.use_compile or not hasattr(torch, "compile"):
        return _lbm_step, False
    try:  # pragma: no cover - depends on the toolchain
        candidate = torch.compile(_lbm_step, mode="max-autotune-no-cudagraphs")
        return candidate, True
    except Exception:  # pragma: no cover
        return _lbm_step, False


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _initial_velocity(cfg: RoughChannelConfig, solid: np.ndarray, u_tau: float, nu: float) -> np.ndarray:
    """Initial velocity field ``(3, nz, ny, nx)``: log-law profile + filtered noise.

    Shared by both back-ends (the native C solver is fed exactly this array),
    so a back-end comparison is a comparison of the *integration*, not of the
    initial condition.
    """
    nz, ny, nx = solid.shape
    k = np.arange(nz, dtype=float)
    y_plus = np.maximum((k - 0.5) * u_tau / nu, 0.5)
    u_prof = float(cfg.init_scale) * np.minimum(u_tau * ((1.0 / KAPPA) * np.log(y_plus) + B_SMOOTH), 0.25)
    u_prof[k < 1.0] = 0.0
    u_prof[k > nz - 2] = 0.0
    ux = np.repeat(u_prof[:, None, None], ny, axis=1).repeat(nx, axis=2)
    rng = np.random.default_rng(int(cfg.seed))
    amp = float(cfg.noise_amplitude) * u_tau * (B_SMOOTH + math.log(cfg.re_tau) / KAPPA)
    noise = rng.standard_normal((3, nz, ny, nx))
    for ax in (2, 3):  # box-filter: seed large scales, not grid-scale sound
        noise = 0.5 * noise + 0.25 * (np.roll(noise, 1, axis=ax) + np.roll(noise, -1, axis=ax))
    vel = np.stack([ux + amp * noise[0], 0.25 * amp * noise[1], 0.25 * amp * noise[2]])
    return np.where((~solid)[None], vel, 0.0).astype(np.float32)


def _initial_populations(cfg: RoughChannelConfig, solid: np.ndarray, u_tau: float, nu: float) -> torch.Tensor:
    """Initial D3Q19 state (local equilibrium of :func:`_initial_velocity`)."""
    vel = _initial_velocity(cfg, solid, u_tau, nu)
    f = _equilibrium(
        torch.ones((cfg.nz, cfg.ny, cfg.nx), dtype=torch.float32),
        torch.from_numpy(vel[0]),
        torch.from_numpy(vel[1]),
        torch.from_numpy(vel[2]),
    )
    return (f * torch.from_numpy((~solid).astype(np.float32))[None]).contiguous()


def _log_window(ks_plus: float, re_tau: float, lo_factor: float = 3.0, hi_fraction: float = 0.35) -> tuple[float, float]:
    return max(lo_factor * ks_plus, 30.0), hi_fraction * re_tau


def case_constants(cfg: RoughChannelConfig, solid: np.ndarray) -> dict[str, Any]:
    """Derived lattice constants shared by both back-ends (identical to 1 ulp)."""
    h_eff = fluid_half_height(solid)
    nu = cfg.u_tau * h_eff / cfg.re_tau
    tau = 0.5 + 3.0 * nu
    if not (0.5 < tau < 1.2):
        raise ValueError(f"unstable relaxation time tau={tau:.4f} from re_tau={cfg.re_tau}, u_tau={cfg.u_tau}")
    s_minus = 1.0 / (TRT_LAMBDA / (tau - 0.5) + 0.5)
    g = cfg.u_tau**2 / h_eff
    k_count = (~solid).sum(axis=(1, 2)).astype(np.float64)
    return {
        "h_eff": h_eff,
        "nu": nu,
        "tau": tau,
        "s_minus": s_minus,
        "g": g,
        "n_fluid": int((~solid).sum()),
        "k_count": k_count,
    }


def assemble_channel_record(
    *,
    cfg: RoughChannelConfig,
    kind: str,
    ks_nominal: float,
    geom: dict[str, Any],
    const: dict[str, Any],
    sums: dict[str, float],
    prof_sum: np.ndarray,
    backend: str,
    wall_seconds: float,
    smooth_b_reference: float | None = None,
    smooth_u_plus_bulk: float | None = None,
) -> dict[str, Any]:
    """Turn raw moment sums into the physical record (single source of truth).

    Every derived quantity — ``u_tau`` from the momentum balance, ``U+``,
    ``Cf``, the log-law window and ``dU+`` — is computed here, from the same
    formulas, for the torch and the native back-end alike.
    """
    n_samples = int(sums["n_samples"])
    if n_samples <= 0:
        raise RuntimeError("no samples collected; increase steps")
    n_fluid = int(const["n_fluid"])
    nu = float(const["nu"])
    h_eff = float(const["h_eff"])
    g = float(const["g"])
    kc = np.asarray(const["k_count"], dtype=float)

    u_bulk = sums["sum_ux"] / (n_samples * n_fluid)
    u_rms = math.sqrt(max(sums["sum_ux2"] / (n_samples * n_fluid) - u_bulk**2, 0.0))
    rho_mean = sums["sum_rho"] / (n_samples * n_fluid)
    u_tau_meas = math.sqrt(g * h_eff * rho_mean)
    u_plus_bulk = u_bulk / u_tau_meas
    cf = 2.0 / u_plus_bulk**2

    prof = np.zeros(cfg.nz)
    valid = kc > 0
    prof[valid] = (np.asarray(prof_sum, dtype=float)[valid] / (n_samples * kc[valid])) / u_tau_meas
    # Bounce-back places the wall surface halfway between the wall layer centre
    # (index 0) and the first fluid layer centre (index 1), i.e. at z = 0.5.
    k_idx = np.arange(cfg.nz)
    y = np.maximum(k_idx - 0.5, 0.5)
    y_plus = y * u_tau_meas / nu

    ks_plus = ks_nominal * u_tau_meas / nu
    lo, hi = _log_window(ks_plus, cfg.re_tau)
    nz_half = (cfg.nz - 1) // 2
    window = (y_plus >= lo) & (y_plus <= hi) & valid & (k_idx >= 1) & (k_idx <= nz_half)
    fit: dict[str, float] | None = None
    fit_self: dict[str, float] | None = None
    d_u_plus_profile = float("nan")
    if int(window.sum()) >= 2:
        fit = fit_log_law(y_plus[window], prof[window], kappa=KAPPA, fit_kappa=False)
        try:
            fit_self = fit_log_law(y_plus[window], prof[window], fit_kappa=True)
        except ValueError:  # pragma: no cover
            fit_self = None
        b_ref = B_SMOOTH if smooth_b_reference is None else float(smooth_b_reference)
        d_u_plus_profile = b_ref - fit["b"]
    d_u_plus_bulk = float("nan")
    if smooth_u_plus_bulk is not None:
        d_u_plus_bulk = float(smooth_u_plus_bulk) - u_plus_bulk

    return {
        "kind": kind,
        "backend": backend,
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz},
        "config": {k: _plain(v) for k, v in asdict(cfg).items()},
        "geometry": geom,
        "ks_nominal": ks_nominal,
        "ks_plus": ks_plus,
        "re_tau": cfg.re_tau,
        "u_tau": u_tau_meas,
        "nu": nu,
        "tau": float(const["tau"]),
        "s_minus": float(const["s_minus"]),
        "smagorinsky_cs": float(cfg.smagorinsky_cs),
        "g": g,
        "h_eff": h_eff,
        "h_plus": h_eff * u_tau_meas / nu,
        "u_bulk": u_bulk,
        "u_rms": u_rms,
        "u_plus_bulk": u_plus_bulk,
        "cf": cf,
        "rho_mean": rho_mean,
        "max_mach": float(sums.get("max_ux", 0.0)) * math.sqrt(1.0 / 3.0),
        "dU_plus_profile": d_u_plus_profile,
        "dU_plus_bulk": d_u_plus_bulk,
        "log_fit": fit,
        "log_fit_free_kappa": fit_self,
        "log_window": {"y_plus_lo": lo, "y_plus_hi": hi, "n_points": int(window.sum())},
        "log_law_exists": bool(3.0 * ks_plus < 0.35 * cfg.re_tau),
        "n_samples": n_samples,
        "wall_seconds": float(wall_seconds),
        "profile": {
            "k": k_idx[: nz_half + 1].tolist(),
            "y_plus": y_plus[: nz_half + 1].tolist(),
            "u_plus": prof[: nz_half + 1].tolist(),
            "n_fluid_per_layer": kc[: nz_half + 1].tolist(),
        },
        "note": cfg.note,
    }


def geometry_and_mask(cfg: RoughChannelConfig, elements: RoughnessElementSet | None) -> tuple[np.ndarray, str, float, dict[str, Any]]:
    """Return ``(solid_mask, kind, ks_nominal, geometry_stats)`` for a case."""
    if elements is None:
        return _bare_walls(cfg.nz, cfg.ny, cfg.nx), "smooth", 0.0, {}
    if elements.mask.shape != (cfg.nz, cfg.ny, cfg.nx):
        raise ValueError(
            f"element mask shape {elements.mask.shape} != config grid {(cfg.nz, cfg.ny, cfg.nx)}"
        )
    return channel_solid_mask(elements.mask), elements.kind, float(elements.ks_nominal), elements.stats()


def run_rough_channel(
    cfg: RoughChannelConfig,
    elements: RoughnessElementSet | None = None,
    *,
    smooth_b_reference: float | None = None,
    smooth_u_plus_bulk: float | None = None,
) -> dict[str, Any]:
    """Run one force-driven rough (or smooth) channel case with the torch backend.

    Parameters
    ----------
    cfg:
        Numerical settings.
    elements:
        Roughness geometry; ``None`` gives the smooth reference case, which is
        the control for acceptance criterion (c).
    smooth_b_reference, smooth_u_plus_bulk:
        Log-law intercept / bulk ``U+`` of the smooth case *on the same grid
        and Re_tau*.  When supplied, ``dU_plus_profile`` and ``dU_plus_bulk``
        are evaluated against the measured smooth reference (removing the
        solver's own log-law bias); when omitted the universal
        ``B_smooth = 5.0`` is used, and ``dU_plus_profile`` then *also*
        quantifies that bias.
    """
    torch.set_num_threads(int(cfg.threads))
    solid_np, kind, ks_nominal, geom = geometry_and_mask(cfg, elements)
    const = case_constants(cfg, solid_np)
    tau = float(const["tau"])
    s_minus = float(const["s_minus"])
    smag2 = float(cfg.smagorinsky_cs) ** 2
    gx = torch.tensor(float(const["g"]), dtype=torch.float32)

    f = _initial_populations(cfg, solid_np, cfg.u_tau, float(const["nu"]))
    solid_f = torch.from_numpy(solid_np.astype(np.float32))
    links_f = torch.from_numpy(build_link_masks(solid_np))
    fluid = ~torch.from_numpy(solid_np)
    stepper, compiled = _make_stepper(cfg)

    # torch.compile is lazy: exercise the candidate once and fall back to the
    # eager kernel on any failure (compiler missing, graph break, non-finite).
    try:
        probe, _ux0, _rho0 = stepper(f, solid_f, links_f, gx, tau, s_minus, smag2)
        if not torch.isfinite(probe).all():
            raise RuntimeError("non-finite state after the first step")
    except Exception:  # pragma: no cover - toolchain dependent
        stepper, compiled = _lbm_step, False

    prof_sum = torch.zeros(cfg.nz, dtype=torch.float64)
    sum_ux = sum_ux2 = sum_rho = 0.0
    n_samples = 0
    max_ux = 0.0
    t_start = time.time()
    for t in range(1, int(cfg.steps)):
        f, ux, rho = stepper(f, solid_f, links_f, gx, tau, s_minus, smag2)
        if t >= cfg.warmup and (t - cfg.warmup) % cfg.sample_interval == 0:
            uxf = ux * fluid
            sum_ux += float(uxf.sum())
            sum_ux2 += float((uxf * ux).sum())
            sum_rho += float((rho * fluid).sum())
            prof_sum += uxf.sum(dim=(1, 2)).double()
            n_samples += 1
            max_ux = max(max_ux, float(ux.abs().max()))
    wall = time.time() - t_start

    return assemble_channel_record(
        cfg=cfg,
        kind=kind,
        ks_nominal=ks_nominal,
        geom=geom,
        const=const,
        sums={"sum_ux": sum_ux, "sum_ux2": sum_ux2, "sum_rho": sum_rho, "n_samples": n_samples, "max_ux": max_ux},
        prof_sum=prof_sum.numpy(),
        backend="torch-compiled" if compiled else "torch-eager",
        wall_seconds=wall,
        smooth_b_reference=smooth_b_reference,
        smooth_u_plus_bulk=smooth_u_plus_bulk,
    )


def _bare_walls(nz: int, ny: int, nx: int) -> np.ndarray:
    solid = np.zeros((nz, ny, nx), dtype=bool)
    solid[0] = True
    solid[-1] = True
    return solid


def smooth_channel_reference(cfg: RoughChannelConfig, **kwargs: Any) -> dict[str, Any]:
    """Shortcut: run the same configuration with no roughness elements."""
    return run_rough_channel(cfg, None, **kwargs)


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
