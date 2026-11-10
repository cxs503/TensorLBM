"""Development-section and isotropic-decay experiments for inlet-quality assessment.

Two LBM experiments that turn inlet quality into numbers:

* :func:`run_development_section` — a uniform stream ``U`` in a D3Q19 duct
  (periodic spanwise).  A synthetic-inflow plane at ``x < injection_cells`` sets
  the instantaneous velocity to ``U + u'(y, z, t)``; the flow then evolves
  downstream.  Reynolds stresses, mean velocity and pressure fluctuations are
  accumulated per streamwise slab, giving

  - the *inlet recovery* (realised vs prescribed ``R_ij`` just downstream of the
    forcing plane), and
  - the *self-preserving development length* ``L_dev``
    (:func:`development_length`).

* :func:`run_isotropic_decay` — a periodic D3Q19 box initialised with a
  divergence-free synthetic field of prescribed energy spectrum.  The kinetic
  energy decay ``K(t)`` is fitted to ``K ∝ (t + t0)^{-n}`` and compared with the
  Comte-Bellot & Corrsin (1971) exponent ``n ≈ 1.30``.

Both are deliberately *inlet-quality* experiments: no wall model, no body, no
drag claim.  The numerical settings are recorded in the returned dictionaries so
that every reported number is traceable to the run that produced it.

The BGK kernels here are written in a vectorised form (no per-direction Python
loop) because the development-section runs need thousands of steps;
:func:`check_kernels_against_library` verifies them against
:mod:`tensorlbm.d3q19` + :mod:`tensorlbm.solver3d` to round-off.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from ..d3q19 import C as _D3Q19_C
from ..d3q19 import W as _D3Q19_W
from ..d3q19 import equilibrium3d, macroscopic3d
from ..solver3d import collide_bgk3d, stream3d_cat

__all__ = [
    "DevelopmentSectionConfig",
    "check_kernels_against_library",
    "decay_exponent",
    "development_length",
    "isotropic_initial_field",
    "run_development_section",
    "run_isotropic_decay",
    "slab_stress",
]


# ---------------------------------------------------------------------------
# Vectorised D3Q19 BGK kernels (same arithmetic as tensorlbm.d3q19/solver3d)
# ---------------------------------------------------------------------------


def _equilibrium(rho, ux, uy, uz, c, w):
    u_sq = ux * ux + uy * uy + uz * uz
    cu = c[0] * ux + c[1] * uy + c[2] * uz
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)


def _macroscopic(f, c):
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    return (
        rho,
        (c[0] * f).sum(dim=0) / rho_safe,
        (c[1] * f).sum(dim=0) / rho_safe,
        (c[2] * f).sum(dim=0) / rho_safe,
    )


def _constants(device, dtype):
    """Return ``((cx, cy, cz), w)`` with shapes ``(19,1,1,1)`` / ``(19,1,1)``."""
    cv = _D3Q19_C.to(device=device, dtype=dtype)  # (19, 3)
    c = (cv[:, 0].view(19, 1, 1, 1), cv[:, 1].view(19, 1, 1, 1), cv[:, 2].view(19, 1, 1, 1))
    w = _D3Q19_W.to(device=device, dtype=dtype).view(19, 1, 1, 1)
    return c, w


def check_kernels_against_library(n: int = 8, seed: int = 0) -> dict:
    """Compare the vectorised kernels with the library ones (round-off agreement).

    Returns:
        Dict with the max absolute differences of ``equilibrium``, ``macroscopic``
        and one BGK collision step, plus their float32 round-off level.
    """
    gen = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.01 * torch.rand((n, n, n), generator=gen, dtype=torch.float32)
    ux = 0.05 * torch.rand((n, n, n), generator=gen, dtype=torch.float32)
    uy = 0.01 * torch.rand((n, n, n), generator=gen, dtype=torch.float32)
    uz = 0.01 * torch.rand((n, n, n), generator=gen, dtype=torch.float32)
    c, w = _constants(torch.device("cpu"), torch.float32)
    feq_fast = _equilibrium(rho, ux, uy, uz, c, w)
    feq_lib = equilibrium3d(rho, ux, uy, uz)
    f = feq_lib + 1e-3 * torch.randn(feq_lib.shape, generator=gen)
    r1 = _macroscopic(f, c)
    r2 = macroscopic3d(f)
    coll_fast = f - (f - _equilibrium(*r1, c, w)) / 0.55
    coll_lib = collide_bgk3d(f, 0.55)
    return {
        "equilibrium_max_abs_diff": float((feq_fast - feq_lib).abs().max()),
        "macroscopic_max_abs_diff": float(max((a - b).abs().max() for a, b in zip(r1, r2))),
        "collision_max_abs_diff": float((coll_fast - coll_lib).abs().max()),
        "field_magnitude": float(f.abs().max()),
    }


# ---------------------------------------------------------------------------
# Development section
# ---------------------------------------------------------------------------


@dataclass
class DevelopmentSectionConfig:
    """Numerical configuration of the uniform-stream development section.

    Args:
        nx, ny, nz: Domain size in cells (``x`` streamwise, ``y`` wall-normal,
            ``z`` spanwise).  Periodic in ``y`` and ``z``.
        tau: BGK relaxation time (``ν = (τ - 1/2)/3``).
        u_mean: Mean streamwise velocity in lattice units.
        injection_cells: Thickness of the fully forced inlet slab.
        ramp_cells: Length over which the forcing weight is cosine-ramped from
            1 to 0 downstream of the slab.  Without a ramp the abrupt velocity
            imposition emits a strong spurious pressure pulse that viscously
            annihilates the injected fluctuations within a few cells; the ramp
            spreads the momentum source and is the standard remedy.
        n_steps: Total number of LBM steps.
        n_warmup: Steps discarded before statistics are accumulated.
        sample_every: Sampling stride.
        outlet_zerograd: Zero-gradient (convective outflow) closure on the last
            ``x`` plane.
        device, dtype: Torch placement.
    """

    nx: int = 80
    ny: int = 24
    nz: int = 24
    tau: float = 0.515
    u_mean: float = 0.05
    injection_cells: int = 2
    ramp_cells: int = 4
    n_steps: int = 3000
    n_warmup: int = 800
    sample_every: int = 10
    outlet_zerograd: bool = True
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.injection_cells < 1 or self.injection_cells >= self.nx // 4:
            raise ValueError("injection_cells must be >=1 and < nx/4")
        if self.ramp_cells < 0 or self.injection_cells + self.ramp_cells >= self.nx // 2:
            raise ValueError("injection_cells + ramp_cells must stay below nx/2")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for a positive viscosity")
        if self.n_warmup >= self.n_steps:
            raise ValueError("n_warmup must be smaller than n_steps")

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0


def _as_sequence(velocity_sequence, n_steps: int) -> torch.Tensor:
    """Materialise a ``(n_steps, 3, ny, nz)`` fluctuation sequence if needed."""
    if callable(velocity_sequence):
        return velocity_sequence(n_steps)
    if not torch.is_tensor(velocity_sequence):
        velocity_sequence = torch.as_tensor(velocity_sequence)
    if velocity_sequence.ndim != 4 or velocity_sequence.shape[1] != 3:
        raise ValueError("velocity sequence must have shape (n_steps, 3, ny, nz)")
    return velocity_sequence


def run_development_section(
    velocity_sequence,
    cfg: DevelopmentSectionConfig,
    *,
    pressure_stats: bool = True,
    snapshot_planes: tuple[int, ...] = (),
) -> dict:
    """Convect synthetic inlet turbulence through a D3Q19 duct and measure ``R_ij(x)``.

    Args:
        velocity_sequence: Fluctuation sequence ``(n_steps, 3, ny, nz)`` or a
            callable ``f(n_steps) -> tensor``; components ``(u', v', w')`` with
            the plane indexed ``[y, z]``.
        cfg: :class:`DevelopmentSectionConfig`.
        pressure_stats: Accumulate density (pressure) fluctuation variance per
            slab — the negative-control metric for spurious acoustic waves.
        snapshot_planes: Step indices whose velocity field is stored.

    Returns:
        Dict with ``x``, ``R`` ``(nx, 3, 3)``, ``R_stderr``, ``mean_velocity``,
        ``tke``, ``pressure_rms``, ``pressure_rms_stderr``, ``samples``,
        ``snapshots`` and ``config``.
    """
    device, dtype = torch.device(cfg.device), cfg.dtype
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    seq = _as_sequence(velocity_sequence, cfg.n_steps).to(device=device, dtype=dtype)
    if tuple(seq.shape[2:]) != (ny, nz):
        raise ValueError(
            f"velocity sequence plane {tuple(seq.shape[2:])} does not match grid {(ny, nz)}"
        )
    c, w = _constants(device, dtype)

    rho = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    ux = torch.full((nz, ny, nx), float(cfg.u_mean), dtype=dtype, device=device)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    f = _equilibrium(rho, ux, uy, uz, c, w)

    n_slab = nx
    m1 = torch.zeros((3, n_slab), dtype=torch.float64, device=device)
    m1_sq = torch.zeros((3, n_slab), dtype=torch.float64, device=device)
    m2 = torch.zeros((6, n_slab), dtype=torch.float64, device=device)
    m2_sq = torch.zeros((6, n_slab), dtype=torch.float64, device=device)
    pp = torch.zeros((n_slab,), dtype=torch.float64, device=device)
    pp_sq = torch.zeros((n_slab,), dtype=torch.float64, device=device)
    p_mean = torch.zeros((n_slab,), dtype=torch.float64, device=device)
    samples = 0
    kept: dict[int, torch.Tensor] = {}
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    inj = int(cfg.injection_cells)
    ramp = int(cfg.ramp_cells)
    n_force = inj + ramp
    beta = torch.zeros(n_slab, dtype=dtype, device=device)
    beta[:inj] = 1.0
    if ramp > 0:
        k = torch.arange(ramp, dtype=dtype, device=device)
        beta[inj:n_force] = 0.5 * (1.0 + torch.cos(math.pi * (k + 0.5) / ramp))
    beta_v = beta[:n_force].view(1, 1, -1)

    for step in range(cfg.n_steps):
        rho, ux, uy, uz = _macroscopic(f, c)
        f = f - (f - _equilibrium(rho, ux, uy, uz, c, w)) / float(cfg.tau)
        f = stream3d_cat(f)

        # --- inlet: drive the velocity to U + u'(y, z, t) over slab + ramp ---
        sl = slice(0, n_force)
        f_sl = f[..., sl]
        rho_sl = torch.clamp(f_sl.sum(dim=0), min=1e-12)
        ux_sl = (c[0] * f_sl).sum(dim=0) / rho_sl
        uy_sl = (c[1] * f_sl).sum(dim=0) / rho_sl
        uz_sl = (c[2] * f_sl).sum(dim=0) / rho_sl
        fl = seq[step]
        ux_t = (cfg.u_mean + fl[0].T).unsqueeze(-1).expand(nz, ny, n_force)
        uy_t = fl[1].T.unsqueeze(-1).expand(nz, ny, n_force)
        uz_t = fl[2].T.unsqueeze(-1).expand(nz, ny, n_force)
        f[..., sl] += beta_v * (
            _equilibrium(rho_sl, ux_t, uy_t, uz_t, c, w)
            - _equilibrium(rho_sl, ux_sl, uy_sl, uz_sl, c, w)
        )

        # --- outlet: zero-gradient (convective) closure ---
        if cfg.outlet_zerograd:
            f[:, :, :, -1] = f[:, :, :, -2]

        if step >= cfg.n_warmup and (step - cfg.n_warmup) % cfg.sample_every == 0:
            rho, ux, uy, uz = _macroscopic(f, c)
            u = torch.stack([ux, uy, uz]).to(torch.float64)
            s1 = u.mean(dim=(1, 2))
            m1 += s1
            m1_sq += s1 * s1
            for k, (i, j) in enumerate(pairs):
                prof = (u[i] * u[j]).mean(dim=(0, 1))
                m2[k] += prof
                m2_sq[k] += prof * prof
            if pressure_stats:
                p_fl = rho.to(torch.float64) - 1.0
                p_mean += p_fl.mean(dim=(0, 1))
                pp_prof_inst = (p_fl * p_fl).mean(dim=(0, 1))
                pp += pp_prof_inst
                pp_sq += pp_prof_inst * pp_prof_inst
            samples += 1
            if step in snapshot_planes:
                kept[step] = u.to(torch.float32).cpu()

    if samples == 0:  # pragma: no cover - config guard
        raise RuntimeError("no statistics samples were accumulated")

    inv = 1.0 / samples
    mean_u = m1 * inv
    R = torch.zeros((n_slab, 3, 3), dtype=torch.float64, device=device)
    R_err = torch.zeros((n_slab, 3, 3), dtype=torch.float64, device=device)
    for k, (i, j) in enumerate(pairs):
        second = m2[k] * inv
        cov = second - mean_u[i] * mean_u[j]
        var = torch.clamp(m2_sq[k] * inv - second * second, min=0.0)
        se = torch.sqrt(var * inv)  # lower bound: consecutive samples correlate
        R[:, i, j] = cov
        R[:, j, i] = cov
        R_err[:, i, j] = se
        R_err[:, j, i] = se
    tke = 0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    if pressure_stats:
        # fluctuating pressure: remove the steady inlet pressure gradient, which
        # otherwise dominates the acoustic signal of interest
        p_bar = p_mean * inv
        pressure_rms = torch.sqrt(torch.clamp(pp * inv - p_bar * p_bar, min=0.0))
        p_var = torch.clamp(pp_sq * inv - (pp * inv) ** 2, min=0.0)
        pressure_rms_err = torch.sqrt(p_var * inv)
    else:
        pressure_rms = torch.full((n_slab,), float("nan"), device=device)
        pressure_rms_err = torch.full((n_slab,), float("nan"), device=device)

    return {
        "x": torch.arange(n_slab, dtype=torch.float64).add(0.5),
        "R": R.cpu(),
        "R_stderr": R_err.cpu(),
        "mean_velocity": mean_u.cpu(),
        "tke": tke.cpu(),
        "pressure_rms": pressure_rms.cpu(),
        "pressure_rms_stderr": pressure_rms_err.cpu(),
        "mean_pressure": (p_mean * inv).cpu(),
        "samples": samples,
        "config": {
            **{
                k: (str(v) if isinstance(v, torch.dtype | torch.device) else v)
                for k, v in asdict(cfg).items()
            },
            "viscosity": cfg.viscosity,
            "u_rms_inlet": float(seq[:, 0].std()),
            "samples_used": samples,
        },
        "snapshots": kept,
    }


def slab_stress(R: torch.Tensor, x0: int, x1: int) -> torch.Tensor:
    """Average the ``(nx, 3, 3)`` slab stress tensor over ``x0 <= x < x1``."""
    return R[x0:x1].mean(dim=0)


def development_length(
    R: torch.Tensor,
    *,
    tol: float = 0.05,
    x_start: int = 1,
    x_stop: int | None = None,
    window: int | None = None,
    require_rate: bool = True,
) -> dict:
    """Self-preserving development length ``L_dev`` from the slab stress series.

    Two stationarity conditions must hold simultaneously for every slab
    ``x' >= L_dev`` up to ``x_stop``:

    1. **Shape stationarity** — the normalised anisotropy shape
       ``Ŝ_ij(x) = R_ij / R_uu`` stays within ``tol`` (entry-wise, relative to
       the O(1) normalised entries) of its far-field value ``Ŝ_ij(x_stop - 1)``.
    2. **Rate stationarity** — the local decay exponent
       ``d ln TKE / d ln x`` stays within ``tol`` (relative) of its far-field
       mean.

    ``tol = 0.05`` is the "95 % self-preservation" criterion of the campaign
    node.  The decay law is fitted per slab from a 3-point log-log stencil.

    Args:
        R: ``(nx, 3, 3)`` slab Reynolds stresses.
        tol: Relative tolerance of the self-preservation criterion.
        x_start: First slab considered (skip the forced/relaxation region).
        x_stop: Last slab considered (excluded); defaults to ``nx - 1`` so the
            outlet closure does not enter the reference state.
        window: Moving-average width in cells used to suppress slab noise
            (default ``max(5, nx//12)``).
        require_rate: Include the rate-stationarity condition.  ``False`` tests
            only the anisotropy *shape*, which is the "95 % self-preservation"
            statement of the campaign node; the rate condition is stricter and
            relaxes only logarithmically, so the two are reported separately.

    Returns:
        Dict with ``L_dev`` (cells, ``nan`` if never reached),
        ``shape_deviation``/``rate_deviation`` arrays and the reference values.
    """
    R = torch.as_tensor(R, dtype=torch.float64)
    nx = R.shape[0]
    # The slab statistics carry a finite-sample error that is large compared
    # with tol=0.05 for a single slab; the self-preservation test is therefore
    # applied to a moving-window average (default width 8 cells).
    w = int(window) if window is not None else max(5, nx // 12)
    w = max(1, min(w, max(1, nx // 4)))
    w += 1 - w % 2  # odd width: keeps the moving average centred on the slab
    # the smoothed series is contaminated within half a window of the ends
    # (replicate padding cannot continue the decay), so neither the admitted
    # window nor the reference state may touch the last w//2 slabs
    x_stop = nx - 1 - w // 2 if x_stop is None else int(x_stop)
    kernel = torch.ones(w, dtype=torch.float64) / w
    R_s = torch.zeros_like(R)
    for i in range(3):
        for j in range(3):
            # replicate padding: zero padding would scale the smoothed stress
            # down at both ends, which fakes a large decay rate there and makes
            # the rate-stationarity test fail even on a perfectly stationary
            # profile.  With the odd width the convolved length is exactly nx.
            padded = torch.nn.functional.pad(
                R[:, i, j].view(1, 1, -1), (w // 2, w // 2), mode="replicate"
            )
            R_s[:, i, j] = torch.nn.functional.conv1d(
                padded, kernel.view(1, 1, -1)
            ).view(-1)
    ruu = R_s[:, 0, 0]
    shape = R_s / ruu.clamp(min=1e-300).view(-1, 1, 1)
    shape_ref = shape[max(x_start, x_stop - w) : x_stop].mean(dim=0)
    shape_dev = (shape - shape_ref).abs().amax(dim=(1, 2))
    tke = 0.5 * (R_s[:, 0, 0] + R_s[:, 1, 1] + R_s[:, 2, 2])

    xs = torch.arange(nx, dtype=torch.float64)
    lnk = torch.log(tke.clamp(min=1e-300))
    lnx = torch.log(xs + 1.0)
    slope = torch.full((nx,), float("nan"), dtype=torch.float64)
    slope[1:-1] = (lnk[2:] - lnk[:-2]) / (lnx[2:] - lnx[:-2])
    window = slope[x_start:x_stop]
    finite = torch.isfinite(window)
    slope_ref = float(window[finite].mean()) if bool(finite.any()) else float("nan")
    if math.isfinite(slope_ref) and abs(slope_ref) > 0.0:
        rate_dev = (slope - slope_ref).abs() / abs(slope_ref)
    else:  # pragma: no cover - only when the energy is flat
        rate_dev = torch.full_like(slope, float("nan"))

    ok = torch.isfinite(shape_dev) & torch.isfinite(rate_dev)
    ok[:x_start] = False
    ok[x_stop:] = False
    ok &= shape_dev <= tol
    if require_rate:
        ok &= rate_dev <= tol
    l_dev = float("nan")
    for i in range(x_start, x_stop):
        # only the admitted window [i, x_stop) must be stationary: ``ok`` is
        # False for x >= x_stop by construction, so scanning ``ok[i:]`` would
        # always fail and report nan.
        if bool(ok[i:x_stop].all()):
            l_dev = float(i + 0.5)
            break
    return {
        "L_dev": l_dev,
        "tol": tol,
        "shape_deviation": shape_dev.numpy().tolist(),
        "rate_deviation": rate_dev.numpy().tolist(),
        "slope_ref": slope_ref,
        "shape_ref": shape_ref.numpy().tolist(),
        "x_start": x_start,
        "x_stop": x_stop,
    }


# ---------------------------------------------------------------------------
# Isotropic decaying turbulence (Comte-Bellot & Corrsin comparison)
# ---------------------------------------------------------------------------


def isotropic_initial_field(
    nx: int,
    ny: int,
    nz: int,
    *,
    k_peak: float = 4.0,
    u_rms: float = 0.05,
    seed: int = 0,
    spectrum: str = "power_law",
    k_min: float = 1.0,
    k_cut: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Divergence-free random velocity field with a Gaussian energy spectrum.

    ``spectrum="power_law"`` (default) sets a modal amplitude
    ``A(k) ∝ k^{-11/6}`` between ``k_min`` and ``k_cut`` (i.e. ``E(k) ∝ k^{-5/3}``)
    with a cosine taper at the cutoff: a broadband initial condition is required
    for a power-law decay measurement, because a narrow-band spectrum decays
    exponentially at rate ``2νk_p²`` instead of transferring energy.
    ``spectrum="gaussian"`` keeps the narrow band around ``k_peak``.

    The field is projected onto the solenoidal subspace with the exact Helmholtz
    projector ``δ_ij - k_i k_j/|k|²`` and rescaled to ``u_rms``.

    Returns:
        ``(3, nz, ny, nx)`` velocity field with zero mean.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, device=device).to(torch.float64)
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny, device=device).to(torch.float64)
    kz = 2.0 * math.pi * torch.fft.fftfreq(nz, device=device).to(torch.float64)
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    k = torch.sqrt(k2)
    kk = k.clone()
    kk[0, 0, 0] = 1.0
    noise = torch.randn((3, nz, ny, nx), generator=gen, device=device, dtype=torch.float64)
    spec = torch.fft.fftn(noise, dim=(-3, -2, -1))
    if spectrum == "gaussian":
        amp = torch.exp(-0.5 * ((k - k_peak) / max(1.0, 0.5 * k_peak)) ** 2)
    elif spectrum == "power_law":
        kc = float(k_cut) if k_cut is not None else 0.25 * min(nx, ny, nz)
        amp = torch.where(
            k >= float(k_min), k.clamp(min=float(k_min)) ** (-11.0 / 6.0), torch.zeros_like(k)
        )
        taper = torch.clamp((kc - k) / max(0.25 * kc, 1e-12), min=0.0, max=1.0)
        amp = amp * taper
    else:  # pragma: no cover - guard
        raise ValueError("spectrum must be 'power_law' or 'gaussian'")
    amp[0, 0, 0] = 0.0
    spec = spec * amp
    kdot = (KX * spec[0] + KY * spec[1] + KZ * spec[2]) / kk**2
    spec[0] -= KX * kdot
    spec[1] -= KY * kdot
    spec[2] -= KZ * kdot
    u = torch.fft.ifftn(spec, dim=(-3, -2, -1)).real
    u = u - u.mean(dim=(-3, -2, -1), keepdim=True)
    rms = torch.sqrt((u**2).mean())
    return (u * (float(u_rms) / float(rms))).to(dtype)


def _shell_spectrum(u: torch.Tensor, n_bins: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Shell-averaged 3-D energy spectrum ``E(k)`` normalised so ``∫E dk = K``."""
    _, nz, ny, nx = u.shape
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx).to(torch.float64)
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny).to(torch.float64)
    kz = 2.0 * math.pi * torch.fft.fftfreq(nz).to(torch.float64)
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
    k = torch.sqrt(KX**2 + KY**2 + KZ**2)
    uh = torch.fft.fftn(u.to(torch.float64), dim=(-3, -2, -1))
    e_hat = 0.5 * (uh[0].abs() ** 2 + uh[1].abs() ** 2 + uh[2].abs() ** 2) / (nx * ny * nz) ** 2
    edges = torch.linspace(0.0, float(k.max()), n_bins + 1, dtype=torch.float64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    E = torch.zeros(n_bins, dtype=torch.float64)
    idx = torch.searchsorted(edges, k.reshape(-1).contiguous()) - 1
    idx = idx.clamp(0, n_bins - 1)
    E.scatter_add_(0, idx, e_hat.reshape(-1))
    dk = float(edges[1] - edges[0])
    return centres, E / dk


def run_isotropic_decay(
    n: int = 64,
    *,
    tau: float = 0.505,
    k_peak: float = 4.0,
    k_cut: float | None = None,
    k_min: float = 1.0,
    spectrum: str = "power_law",
    u_rms: float = 0.05,
    n_steps: int = 2000,
    sample_every: int = 50,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Decay a divergence-free isotropic field in a periodic D3Q19 box.

    Returns:
        Dict with ``t``, ``tke``, ``dissipation`` (spectral strain-rate
        contraction, ``ε = 2ν S_ij S_ij``), ``spectra`` (list of ``{t, k, E}``)
        and ``config``.
    """
    device = torch.device(device)
    u0 = isotropic_initial_field(
        n, n, n, k_peak=k_peak, k_cut=k_cut, k_min=k_min, spectrum=spectrum,
        u_rms=u_rms, seed=seed, device=device, dtype=dtype,
    )
    c, w = _constants(device, dtype)
    rho = torch.ones((n, n, n), dtype=dtype, device=device)
    f = _equilibrium(rho, u0[0], u0[1], u0[2], c, w)
    times: list[int] = []
    tkes: list[float] = []
    diss: list[float] = []
    spectra: list[dict] = []
    kx = 2.0 * math.pi * torch.fft.fftfreq(n, device=device).to(torch.float64)
    KZ, KY, KX = torch.meshgrid(kx, kx, kx, indexing="ij")
    kgrid = torch.stack([KX, KY, KZ])
    nu = (tau - 0.5) / 3.0

    def _measure(step: int) -> None:
        _, ux, uy, uz = _macroscopic(f, c)
        uu = torch.stack([ux, uy, uz]).to(torch.float64)
        tke = 0.5 * float((uu**2).mean())
        uh = torch.fft.fftn(uu, dim=(-3, -2, -1))
        grad = [
            [torch.fft.ifftn(uh[i] * kgrid[j], dim=(-3, -2, -1)).real for j in range(3)]
            for i in range(3)
        ]
        sij2 = 0.0
        for i in range(3):
            for j in range(3):
                s = 0.5 * (grad[i][j] + grad[j][i])
                sij2 += float((s**2).mean())
        times.append(step)
        tkes.append(tke)
        diss.append(2.0 * nu * sij2)
        kk, E = _shell_spectrum(uu)
        spectra.append({"t": step, "k": kk.numpy().tolist(), "E": E.numpy().tolist()})

    _measure(0)
    diverged = False
    for step in range(1, n_steps + 1):
        rho, ux, uy, uz = _macroscopic(f, c)
        f = f - (f - _equilibrium(rho, ux, uy, uz, c, w)) / float(tau)
        f = stream3d_cat(f)
        if step % sample_every == 0:
            _measure(step)
            if not (math.isfinite(tkes[-1]) and tkes[-1] > 0.0):
                # BGK with tau just above 1/2 is only marginally stable at
                # high Re_lambda: report the blow-up instead of poisoning the
                # decay fit with NaN (the finite prefix is kept).
                diverged = True
                times.pop()
                tkes.pop()
                diss.pop()
                spectra.pop()
                break

    return {
        "diverged": diverged,
        "t": times,
        "tke": tkes,
        "dissipation": diss,
        "spectra": spectra,
        "config": {
            "n": n,
            "tau": tau,
            "viscosity": nu,
            "k_peak": k_peak,
            "k_cut": k_cut,
            "spectrum": spectrum,
            "u_rms": u_rms,
            "n_steps": n_steps,
            "sample_every": sample_every,
            "seed": seed,
            "dtype": str(dtype),
        },
    }


def decay_exponent(
    t: list[float] | torch.Tensor,
    tke: list[float] | torch.Tensor,
    *,
    n_fit: int | None = None,
) -> dict:
    """Fit ``K(t) ≈ A (t + t0)^{-n}`` and report the decay exponent ``n``.

    The virtual origin ``t0`` is scanned on a grid and the ``(t0, n)`` pair with
    the smallest residual in log-log space is kept.  The reported ``n_range`` is
    the spread of ``n`` over all ``t0`` whose residual stays within 5 % of the
    optimum — a fit-conditioning proxy, not a statistical error bar.

    Args:
        t: Time samples (LBM steps).
        tke: Kinetic energy samples.
        n_fit: Use only the last ``n_fit`` samples (default: drop the first 10 %).

    Returns:
        Dict with ``n``, ``t0``, ``A``, ``rms_residual``, ``n_range``,
        ``fit_start_step`` and ``n_points``.
    """
    t = torch.as_tensor(t, dtype=torch.float64)
    k = torch.as_tensor(tke, dtype=torch.float64)
    start = max(1, int(0.1 * len(t))) if n_fit is None else max(1, len(t) - int(n_fit))
    tf, kf = t[start:], k[start:]
    lo, hi = -0.5 * float(tf[0]), 3.0 * float(tf[-1])

    def _scan(t0_grid: torch.Tensor, residuals: list[tuple[float, float]]) -> dict:
        best: dict | None = None
        for t0 in t0_grid:
            x = torch.log(tf + t0)
            y = torch.log(kf)
            A = torch.stack([x, torch.ones_like(x)], dim=1)
            sol = torch.linalg.lstsq(A, y).solution
            res = float(torch.sqrt((((A @ sol) - y) ** 2).mean()))
            residuals.append((float(t0), res))
            if best is None or res < best["rms_residual"]:
                best = {
                    "n": float(-sol[0]),
                    "t0": float(t0),
                    "A": float(torch.exp(sol[1])),
                    "rms_residual": res,
                }
        return best

    residuals: list[tuple[float, float]] = []
    best = _scan(torch.linspace(lo, hi, 240), residuals)
    assert best is not None
    # second pass: the residual is flat in t0 near the optimum, so a 240-point
    # coarse grid biases ``n`` by O(1 %) even on an exact power law.  Refine
    # twice on a grid one coarse-spacing wide around the current optimum.
    span = (hi - lo) / 239.0
    for _ in range(2):
        centre = best["t0"]
        coarse = _scan(torch.linspace(centre - span, centre + span, 201), residuals)
        span /= 100.0
        if coarse is not None and coarse["rms_residual"] < best["rms_residual"]:
            best = coarse
    assert best is not None
    tol = 1.05 * best["rms_residual"]
    accepted = []
    for t0, res in residuals:
        if res > tol:
            continue
        x = torch.log(tf + t0)
        y = torch.log(kf)
        A = torch.stack([x, torch.ones_like(x)], dim=1)
        accepted.append(float(-torch.linalg.lstsq(A, y).solution[0]))
    best["n_range"] = [min(accepted), max(accepted)] if accepted else [best["n"], best["n"]]
    best["fit_start_step"] = int(t[start])
    best["n_points"] = int(len(tf))
    return best