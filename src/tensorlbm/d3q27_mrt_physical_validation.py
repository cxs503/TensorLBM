"""D3Q27 MRT **physical validation** probe (R1): analytic-solution anchors.

Where :mod:`tensorlbm.d3q27_composition_evidence` and
:mod:`tensorlbm.d3q27_full_composition_test` establish that the D3Q27 MRT
composition is *executable and internally consistent* (involution, mass
conservation, round-trip), this module establishes that it is *physically
accurate* against closed-form Navier-Stokes solutions, and quantifies the
error.  It is a **cold-path evidence writer**: nothing here is part of a
solver hot loop.

Anchors (all closed form, all executed with the library's own
``collide_mrt27`` / ``stream27_roll`` / ``bounce_back_cells_27`` /
``macroscopic27``):

1. **Shear-wave decay** — ``u_x = u0 sin(k y)`` is an *exact* unsteady
   solution of the incompressible Navier-Stokes equations (the convective
   term vanishes identically, the pressure stays uniform).  Its kinetic
   energy decays as ``exp(-2 nu k^2 t)`` with ``nu = (tau - 1/2)/3``.
   Measured decay rate therefore returns the *effective* kinematic
   viscosity of the discrete composition.  Run on a grid ladder to show the
   error is spatial truncation, not a model defect.
2. **Sound speed** — a right-going linear acoustic mode (``rho' = A
   sin(k x)``, ``u_x = c_s A sin(k x)``) travels at ``c_s = 1/sqrt(3)`` in
   lattice units.  The phase of the ``k``-mode is tracked in time, so the
   measured speed does not assume ``c_s``.
3. **Spectral (k-space) cross-check** — the linearised one-step operator
   ``Lambda(k) = diag(exp(-i k c_x)) . J`` is assembled from the *autograd
   Jacobian* ``J = d(collide_mrt27)/df`` at the resting equilibrium.  Its
   eigenvalues give the shear decay rate, the sound frequency and the sound
   attenuation **without time stepping**.  Agreement with the time-domain
   measurements (1, 2) is a cross-method verification of both the moment
   matrix and the stepper.
4. **Wall-bounded eigenmode** — between two stationary ``bounce_back_cells_27``
   walls, ``u_x = u0 sin(pi (y - y_w) / H) exp(-nu (pi/H)^2 t)`` is an exact
   solution with no-slip at both walls.  Fitting the measured decay rate
   gives the *effective wall position* ``y_w`` of the bounce-back scheme and
   its physical error.  For this full-way bounce-back the walls sit half a
   lattice unit from the solid cell centres, i.e. ``H = ny - 2``.
5. **Conservation** — mass and momentum are invariants of the periodic
   collide + stream composition; measured in ``float64`` over many steps so
   that the tolerance is round-off, not "small enough".
6. **Controls** — a bulk-rate (``s_e``) sweep must leave the shear branch
   untouched, the ``s_e = s_eps = s_q = s_pi = 1/tau`` limit must reproduce
   BGK bit-for-bit, and a ``tau`` control must show the measurement resolves
   a viscosity change (i.e. the acceptance is not vacuous).

Honest scope: this is single-phase, static-geometry, CPU evidence.  It does
**not** validate wall functions, turbulence models, multiphase, free
surface, moving boundaries, AMR, or any GPU/accelerator backend, and it does
not by itself lift the ``general_capability_matrix`` WITHHELD disposition for
the full D3Q27 platform composition (that needs the platform-level
composition path, not a physics anchor).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from .boundaries_d3q27 import bounce_back_cells_27
from .d3q27 import (
    C,
    collide_bgk27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)

ARTIFACT_ID = "d3q27-mrt-physical-validation-r1"
ARTIFACT_VERSION = "d3q27-mrt-physical-validation-r1"

#: Ecosystem accuracy standard: a physics anchor within 3 % is acceptable.
PHYSICS_TOLERANCE = 0.03


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _roll(state: torch.Tensor, tau: float, s_e: float = 1.19) -> torch.Tensor:
    """One periodic collide + stream step (no walls)."""
    return stream27_roll(collide_mrt27(state, tau, s_e=s_e))


def _kinetic_energy(f: torch.Tensor, fluid: torch.Tensor | None = None) -> float:
    """``0.5 <rho |u|^2>`` over fluid cells (unit cell volume)."""
    rho, ux, uy, uz = macroscopic27(f)
    energy = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
    if fluid is not None:
        return float(energy[fluid].mean())
    return float(energy.mean())


def _fit_log_slope(times: list[int], values: list[float]) -> float:
    """Least-squares slope of ``log(values)`` against ``times``."""
    n = len(times)
    if n < 3:
        raise ValueError("at least three samples are required for a slope fit")
    t = torch.tensor(times, dtype=torch.float64)
    y = torch.log(torch.tensor(values, dtype=torch.float64))
    t_mean = t.mean()
    y_mean = y.mean()
    return float(((t - t_mean) * (y - y_mean)).sum() / ((t - t_mean) ** 2).sum())


def _unwrap(phases: list[float]) -> torch.Tensor:
    """Unwrap a phase ramp: accumulate step-wise differences in (-pi, pi]."""
    values = torch.tensor(phases, dtype=torch.float64)
    steps = torch.diff(values)
    wrapped = torch.remainder(steps + math.pi, 2.0 * math.pi) - math.pi
    return torch.cat([values[:1], values[:1] + torch.cumsum(wrapped, dim=0)])


def _rel_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return abs(float(measured))
    return abs(float(measured) - float(reference)) / abs(float(reference))


# ---------------------------------------------------------------------------
# 1. shear-wave decay -> effective kinematic viscosity
# ---------------------------------------------------------------------------


def measure_shear_wave_viscosity(
    n: int = 16,
    tau: float = 0.65,
    *,
    amplitude: float = 0.01,
    steps: int = 300,
    sample_every: int = 20,
    s_e: float = 1.19,
    dtype: torch.dtype = torch.float64,
    collect_series: bool = False,
) -> dict[str, Any]:
    """Measure the effective viscosity from a plane shear wave.

    ``u_x = A sin(k y)`` with ``k = 2 pi / n`` is an exact solution of the
    incompressible NS equations whose energy decays as ``exp(-2 nu k^2 t)``.
    The decay is fitted over ``steps`` steps after discarding nothing (the
    analytic mode is the initial condition, so no transient exists beyond
    the non-hydrodynamic defect modes).

    Returns a dict with ``measured_viscosity``, ``analytic_viscosity``,
    ``rel_error``, ``fit_steps``, ``k`` and the first/last energy samples.
    """
    if n < 8:
        raise ValueError("n must be >= 8 so that the shear mode is resolved")
    k = 2.0 * math.pi / n
    y = torch.arange(n, dtype=dtype)
    profile = torch.sin(k * y).view(1, n, 1).expand(n, n, n)
    rho = torch.ones((n, n, n), dtype=dtype)
    zero = torch.zeros_like(rho)
    f = equilibrium27(rho, amplitude * profile.contiguous(), zero, zero)

    times: list[int] = []
    energies: list[float] = []
    for step in range(1, steps + 1):
        f = _roll(f, tau, s_e)
        if step % sample_every == 0:
            times.append(step)
            energies.append(_kinetic_energy(f))

    slope = _fit_log_slope(times, energies)
    measured = -slope / (2.0 * k * k)
    analytic = (tau - 0.5) / 3.0
    result = {
        "grid": n,
        "lattice_wavenumber": k,
        "tau": tau,
        "s_e": s_e,
        "fit_steps": [times[0], times[-1]],
        "measured_viscosity": measured,
        "analytic_viscosity": analytic,
        "rel_error": _rel_error(measured, analytic),
        "energy_first": energies[0],
        "energy_last": energies[-1],
    }
    if collect_series:
        result["series"] = {"times": times, "energies": energies}
    return result


# ---------------------------------------------------------------------------
# 2. linear acoustic mode -> sound speed
# ---------------------------------------------------------------------------


def measure_sound_speed(
    n: int = 32,
    tau: float = 0.65,
    *,
    amplitude: float = 1e-3,
    steps: int = 300,
    sample_every: int = 5,
    s_e: float = 1.19,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Measure the lattice sound speed and sound attenuation of the composition.

    A right-going linear acoustic mode is initialised as ``rho' = A sin(k x)``,
    ``u_x = c_s A sin(k x)`` (first-order acoustics: ``u = p'/(rho_0 c_s)``).
    The amplitude of the ``k``-Fourier mode of ``rho`` is tracked, so the
    phase velocity is obtained from a phase ramp ``phi(t) = -omega t`` and
    the attenuation from ``log|A_k(t)|``; no lattice constant is assumed.
    """
    k = 2.0 * math.pi / n
    cs_ref = 1.0 / math.sqrt(3.0)
    x = torch.arange(n, dtype=dtype)
    wave = torch.sin(k * x).view(1, 1, n).expand(n, n, n)
    rho = 1.0 + amplitude * wave
    ux = (cs_ref * amplitude) * wave
    zero = torch.zeros_like(rho)
    f = equilibrium27(rho.contiguous(), ux.contiguous(), zero, zero)

    mode = torch.exp(-1j * k * x.to(torch.float64)).to(torch.complex128)
    phases: list[float] = []
    amplitudes: list[float] = []
    times: list[int] = []
    for step in range(1, steps + 1):
        f = _roll(f, tau, s_e)
        if step % sample_every == 0:
            density = macroscopic27(f)[0]
            # mode is x-only: average the identical y,z planes
            profile = density.to(torch.complex128).mean(dim=(0, 1))
            coeff = (profile @ mode) / n
            phases.append(float(torch.angle(coeff)))
            amplitudes.append(float(torch.abs(coeff)))
            times.append(step)

    phase = _unwrap(phases)
    t = torch.tensor(times, dtype=torch.float64)
    t_mean, p_mean = t.mean(), phase.mean()
    omega = -float(((t - t_mean) * (phase - p_mean)).sum() / ((t - t_mean) ** 2).sum())
    measured = omega / k
    attenuation = -_fit_log_slope(times, amplitudes)
    return {
        "grid": n,
        "lattice_wavenumber": k,
        "tau": tau,
        "absorbing_steps": steps,
        "measured_sound_speed": measured,
        "analytic_sound_speed": cs_ref,
        "rel_error": _rel_error(measured, cs_ref),
        "measured_omega": omega,
        "measured_attenuation": attenuation,
        "attenuation_over_nu_k2": attenuation / (((tau - 0.5) / 3.0) * k * k),
    }


# ---------------------------------------------------------------------------
# 3. k-space spectral cross-check
# ---------------------------------------------------------------------------


def spectral_step_eigenvalues(
    n: int = 16,
    tau: float = 0.65,
    *,
    s_e: float = 1.19,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Linearised one-step operator eigenvalues for wavenumber ``k = 2 pi/n``.

    ``J = d(collide_mrt27)/df`` at the resting equilibrium ``feq(rho=1, u=0)``
    (obtained by autograd, i.e. from the implementation itself), composed
    with the streaming phase factors ``exp(-i k c_x)``.  The shear
    (transverse) eigenvalue gives ``nu = -log|lambda| / k^2``; the sound
    eigenvalues give ``c_s = arg(lambda)/k`` and the attenuation
    ``-log|lambda|``.
    """
    k = 2.0 * math.pi / n
    rho = torch.ones((1, 1, 1), dtype=dtype)
    zero = torch.zeros_like(rho)
    base = equilibrium27(rho, zero, zero, zero).reshape(27).clone()

    def collide_vector(raw: torch.Tensor) -> torch.Tensor:
        return collide_mrt27(raw.reshape(27, 1, 1, 1), tau, s_e=s_e).reshape(27)

    jacobian = torch.autograd.functional.jacobian(collide_vector, base.clone(), vectorize=True)
    directions = C.to(dtype=torch.float64)
    phase = torch.exp(-1j * k * directions[:, 0]).to(torch.complex128)
    lam = torch.diag(phase) @ jacobian.to(torch.complex128)
    eigenvalues = torch.linalg.eigvals(lam)
    magnitude = torch.abs(eigenvalues)
    angle = torch.abs(torch.angle(eigenvalues))

    # conserved mass mode: |lambda| = 1 and zero frequency
    hydrodynamic = ~((magnitude - 1.0).abs() < 1e-8)
    # transverse: non-propagating (zero frequency), least damped
    shear_candidates = torch.where(
        hydrodynamic & (angle < 1e-8), magnitude, torch.zeros_like(magnitude)
    )
    shear_index = int(torch.argmax(shear_candidates))
    # sound: propagating, frequency nearest c_s k
    cs_ref = 1.0 / math.sqrt(3.0)
    target = cs_ref * k
    sound_index = int(
        torch.argmin(torch.where(hydrodynamic, (angle - target).abs(), torch.full_like(angle, 1e9)))
    )

    shear_rate = -math.log(float(magnitude[shear_index]))
    sound_omega = float(angle[sound_index])
    return {
        "grid": n,
        "lattice_wavenumber": k,
        "tau": tau,
        "s_e": s_e,
        "shear_eigenvalue": [
            float(eigenvalues[shear_index].real),
            float(eigenvalues[shear_index].imag),
        ],
        "spectral_viscosity": shear_rate / (k * k),
        "sound_eigenvalue": [
            float(eigenvalues[sound_index].real),
            float(eigenvalues[sound_index].imag),
        ],
        "spectral_sound_speed": sound_omega / k,
        "spectral_attenuation": -math.log(float(magnitude[sound_index])),
        "n_eigenvalues": int(eigenvalues.shape[0]),
    }


# ---------------------------------------------------------------------------
# 4. wall-bounded eigenmode -> effective bounce-back wall position
# ---------------------------------------------------------------------------


def measure_wall_mode_decay(
    ny: int = 16,
    tau: float = 0.65,
    *,
    nx: int = 4,
    nz: int = 4,
    amplitude: float = 0.01,
    steps: int = 400,
    sample_every: int = 20,
    wall_offset: float = 0.5,
    s_e: float = 1.19,
    dtype: torch.dtype = torch.float64,
    collect_series: bool = False,
) -> dict[str, Any]:
    """Decay rate of the first no-slip eigenmode between two stationary walls.

    Solid rows are ``y = 0`` and ``y = ny - 1`` (full-way bounce-back).  The
    analytic no-slip planes then sit ``wall_offset`` lattice units from the
    solid cell centres, i.e. at ``y = 0.5`` and ``y = ny - 1.5``, giving a
    channel width ``H = ny - 2``.  The exact solution is

        ``u_x(y, t) = A sin(pi (y - 0.5) / H) exp(-nu (pi / H)^2 t)``.

    Fitting the energy decay returns the effective decay rate, whose ratio to
    ``nu (pi/H)^2`` is the physical error of the wall composition.  The
    implied effective width ``H_eff = pi / sqrt(rate / nu)`` exposes the wall
    position directly, so a wrong wall convention cannot hide in "it looks
    parabolic".
    """
    if ny < 8:
        raise ValueError("ny must be >= 8 so that the wall mode is resolved")
    # solid centres at y = 0 and y = ny - 1; each no-slip plane sits
    # ``wall_offset`` inside, so the span between the two walls is
    # (ny - 1) - 2 * wall_offset
    height = float(ny - 1) - 2.0 * wall_offset
    y = torch.arange(ny, dtype=dtype)
    profile = torch.sin(math.pi * (y - wall_offset) / height)
    rho = torch.ones((nz, ny, nx), dtype=dtype)
    ux = (amplitude * profile).view(1, ny, 1).expand(nz, ny, nx).contiguous()
    zero = torch.zeros_like(rho)
    f = equilibrium27(rho, ux, zero, zero)

    mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    fluid = ~mask

    times: list[int] = []
    energies: list[float] = []
    for step in range(1, steps + 1):
        # NoDynamics inside the solid (the composition used by step27):
        # colliding the solid populations would perturb the reflections.
        f_collided = torch.where(mask.unsqueeze(0), f, collide_mrt27(f, tau, s_e=s_e))
        f = stream27_roll(f_collided)
        f = bounce_back_cells_27(f, mask)
        if step % sample_every == 0:
            times.append(step)
            energies.append(_kinetic_energy(f, fluid))

    slope = _fit_log_slope(times, energies)
    rate = -0.5 * slope  # energy decays at twice the amplitude rate
    nu = (tau - 0.5) / 3.0
    k_mode = math.pi / height
    analytic = nu * k_mode * k_mode
    height_eff = math.pi / math.sqrt(rate / nu)
    result = {
        "ny": ny,
        "channel_width": height,
        "wall_offset": wall_offset,
        "tau": tau,
        "s_e": s_e,
        "measured_decay_rate": rate,
        "analytic_decay_rate": analytic,
        "rel_error": _rel_error(rate, analytic),
        "implied_channel_width": height_eff,
        "implied_wall_offset": 0.5 * (float(ny - 1) - height_eff),
        "fit_steps": [times[0], times[-1]],
    }
    if collect_series:
        result["series"] = {"times": times, "energies": energies}
    return result


# ---------------------------------------------------------------------------
# 5. conservation of the periodic composition
# ---------------------------------------------------------------------------


def check_periodic_conservation(
    n: int = 8,
    tau: float = 0.7,
    *,
    steps: int = 200,
    seed: int = 20261108,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Mass and momentum invariants of periodic collide + stream.

    Population-level mass and momentum are exactly conserved by both the MRT
    collision (rows 0-3 have zero relaxation) and by the roll-based
    streaming (a permutation per direction), so in ``float64`` the drift must
    be round-off.
    """
    generator = torch.Generator().manual_seed(seed)
    # arbitrary positive populations around the resting equilibrium: the
    # invariants must hold for *any* state, not only for near-equilibrium ones
    # uniform drift so that the momentum scale is physical (not noise)
    ones = torch.ones((n, n, n), dtype=torch.float64)
    zeros = torch.zeros((n, n, n), dtype=torch.float64)
    f = equilibrium27(ones, 0.05 * ones, zeros, zeros)
    f = f + 1e-3 * (torch.rand((27, n, n, n), generator=generator, dtype=torch.float64) - 0.5)

    directions = C.to(dtype=torch.float64)

    def invariants(state: torch.Tensor) -> tuple[float, torch.Tensor]:
        mass = float(state.sum())
        momentum = (directions * state.reshape(27, -1).sum(dim=1, keepdim=True)).sum(dim=0)
        return mass, momentum

    mass0, momentum0 = invariants(f)
    for _ in range(steps):
        f = _roll(f, tau)
    mass1, momentum1 = invariants(f)

    mass_scale = abs(mass0) if mass0 != 0.0 else 1.0
    momentum_scale = max(float(torch.linalg.vector_norm(momentum0).item()), 1.0)
    return {
        "grid": n,
        "steps": steps,
        "dtype": str(dtype).replace("torch.", ""),
        "mass_initial": mass0,
        "mass_final": mass1,
        "mass_rel_drift": abs(mass1 - mass0) / mass_scale,
        "momentum_initial": [float(v) for v in momentum0],
        "momentum_final": [float(v) for v in momentum1],
        "momentum_rel_drift": float(torch.linalg.vector_norm(momentum1 - momentum0).item())
        / momentum_scale,
        "finite": bool(torch.isfinite(f).all()),
    }


# ---------------------------------------------------------------------------
# 6. controls
# ---------------------------------------------------------------------------


def check_controls(tau: float = 0.65) -> dict[str, Any]:
    """Negative/positive controls that keep the anchors non-vacuous.

    * ``bgk_limit``: with every non-conserved rate set to ``1/tau`` the MRT
      collision must reproduce :func:`collide_bgk27` bit-for-bit (the MRT
      moment matrix diagonalises BGK in that limit).
    * ``bulk_rate_independence``: the shear branch must not react to the
      energy-mode rate ``s_e``.
    * ``tau_sensitivity``: a ``tau`` control must change the measured
      viscosity by roughly the analytic amount, proving the measurement
      resolves viscosity rather than returning a constant.
    """
    generator = torch.Generator().manual_seed(31415)
    f = torch.rand((27, 4, 5, 6), generator=generator, dtype=torch.float64) + 0.5
    mrt = collide_mrt27(f, tau, s_e=1.0 / tau, s_eps=1.0 / tau, s_q=1.0 / tau, s_pi=1.0 / tau)
    bgk = collide_bgk27(f, tau)
    bgk_delta = float((mrt - bgk).abs().max())

    shear_default = measure_shear_wave_viscosity(16, tau, steps=240, s_e=1.19)
    shear_high_bulk = measure_shear_wave_viscosity(16, tau, steps=240, s_e=1.60)
    bulk_delta = _rel_error(
        shear_high_bulk["measured_viscosity"], shear_default["measured_viscosity"]
    )

    tau_control = 0.68
    shear_control = measure_shear_wave_viscosity(16, tau_control, steps=240)
    return {
        "bgk_limit_max_abs_delta": bgk_delta,
        "bgk_limit_scale": float(f.abs().max()),
        "bulk_rate_independence_rel_delta": bulk_delta,
        "bulk_rate_viscosity_default": shear_default["measured_viscosity"],
        "bulk_rate_viscosity_high": shear_high_bulk["measured_viscosity"],
        "tau_control": tau_control,
        "tau_control_measured_viscosity": shear_control["measured_viscosity"],
        "tau_control_analytic_viscosity": shear_control["analytic_viscosity"],
        "tau_control_rel_error": shear_control["rel_error"],
        "tau_control_rel_shift_vs_base": _rel_error(
            shear_control["measured_viscosity"], shear_default["measured_viscosity"]
        ),
    }


# ---------------------------------------------------------------------------
# artifact assembly
# ---------------------------------------------------------------------------


def _check(
    check_id: str,
    aspect: str,
    measured: float,
    reference: float,
    tolerance: float,
    *,
    units: str = "",
    note: str = "",
) -> dict[str, Any]:
    rel_error = _rel_error(measured, reference)
    return {
        "id": check_id,
        "aspect": aspect,
        "status": "pass" if rel_error <= tolerance else "fail",
        "measured": measured,
        "reference": reference,
        "rel_error": rel_error,
        "tolerance": tolerance,
        "units": units,
        "note": note,
    }


def run_d3q27_mrt_physical_validation_probe(
    *,
    grids: tuple[int, ...] = (16, 24, 32),
    tau: float = 0.65,
    wall_grids: tuple[int, ...] = (16, 24, 32),
) -> dict[str, Any]:
    """Execute every anchor and return a JSON-ready artifact dict.

    Default sizing keeps the whole run to a few tens of seconds on CPU
    (grids 16-32, 240-400 steps): it is a cold-path evidence probe, not a
    production sweep.  Set ``grids``/``wall_grids`` explicitly to trade cost
    for resolution.
    """
    shear = [measure_shear_wave_viscosity(n, tau, steps=300) for n in grids]
    sound = [measure_sound_speed(n, tau, steps=300) for n in (min(grids), max(grids))]
    spectral = [spectral_step_eigenvalues(n, tau) for n in (min(grids), max(grids))]
    walls = [measure_wall_mode_decay(ny, tau, steps=400) for ny in wall_grids]
    conservation = check_periodic_conservation()
    controls = check_controls(tau)

    checks: list[dict[str, Any]] = []
    for row in shear:
        checks.append(
            _check(
                f"shear_viscosity_grid{row['grid']}",
                "streaming_collision_coupling/viscosity",
                row["measured_viscosity"],
                row["analytic_viscosity"],
                PHYSICS_TOLERANCE,
                units="lattice^2/step",
                note=f"shear-wave decay k={row['lattice_wavenumber']:.5f}, "
                f"nu=(tau-1/2)/3 reference",
            )
        )
    for row in sound:
        checks.append(
            _check(
                f"sound_speed_grid{row['grid']}",
                "output/acoustics",
                row["measured_sound_speed"],
                row["analytic_sound_speed"],
                PHYSICS_TOLERANCE,
                units="lattice/step",
                note="right-going linear acoustic mode, phase tracking",
            )
        )
    for spec, row in zip((min(grids), max(grids)), spectral):
        timedomain_shear = next(item for item in shear if item["grid"] == spec)
        timedomain_sound = next(item for item in sound if item["grid"] == spec)
        checks.append(
            _check(
                f"spectral_vs_timedomain_viscosity_grid{spec}",
                "cross_method/viscosity",
                row["spectral_viscosity"],
                timedomain_shear["measured_viscosity"],
                0.01,
                units="lattice^2/step",
                note="autograd Jacobian eigenvalues vs time-domain decay",
            )
        )
        checks.append(
            _check(
                f"spectral_vs_timedomain_sound_speed_grid{spec}",
                "cross_method/acoustics",
                row["spectral_sound_speed"],
                timedomain_sound["measured_sound_speed"],
                0.01,
                units="lattice/step",
                note="eigenvalue frequency vs tracked phase ramp",
            )
        )
        checks.append(
            _check(
                f"spectral_vs_timedomain_attenuation_grid{spec}",
                "cross_method/attenuation",
                row["spectral_attenuation"],
                timedomain_sound["measured_attenuation"],
                0.05,
                units="1/step",
                note="linearised eigenvalue modulus vs amplitude fit",
            )
        )
    for row in walls:
        checks.append(
            _check(
                f"wall_mode_decay_ny{row['ny']}",
                "wall_treatment/no_slip",
                row["measured_decay_rate"],
                row["analytic_decay_rate"],
                PHYSICS_TOLERANCE,
                units="1/step",
                note=f"implied channel width {row['implied_channel_width']:.3f} "
                f"(analytic H={row['channel_width']:.1f})",
            )
        )
        checks.append(
            _check(
                f"wall_position_ny{row['ny']}",
                "wall_treatment/effective_wall_offset",
                row["implied_wall_offset"],
                row["wall_offset"],
                0.05,
                units="lattice",
                note="implied offset of the no-slip plane from the solid cell centre",
            )
        )

    checks.append(
        _check(
            "periodic_mass_conservation",
            "conservation/mass",
            conservation["mass_rel_drift"],
            0.0,
            1e-12,
            units="relative",
            note="float64, 200 steps",
        )
    )
    checks.append(
        _check(
            "periodic_momentum_conservation",
            "conservation/momentum",
            conservation["momentum_rel_drift"],
            0.0,
            1e-12,
            units="relative",
            note="float64, 200 steps",
        )
    )
    checks.append(
        _check(
            "bgk_limit_equivalence",
            "control/bgk_limit",
            controls["bgk_limit_max_abs_delta"],
            0.0,
            1e-10,
            units="max |df|",
            note=f"population scale {controls['bgk_limit_scale']:.3f}",
        )
    )
    checks.append(
        _check(
            "bulk_rate_independence",
            "control/bulk_rate",
            controls["bulk_rate_independence_rel_delta"],
            0.0,
            1e-3,
            units="relative viscosity change",
            note="s_e=1.19 vs 1.60 must leave the shear branch unchanged",
        )
    )
    checks.append(
        _check(
            "tau_sensitivity_control",
            "control/tau_sensitivity",
            controls["tau_control_measured_viscosity"],
            controls["tau_control_analytic_viscosity"],
            0.01,
            units="lattice^2/step",
            note=f"tau={controls['tau_control']} control resolves a "
            f"{controls['tau_control_rel_shift_vs_base'] * 100:.1f}% viscosity shift",
        )
    )

    physically_validated = sorted(
        {
            "streaming_collision_coupling",
            "output",
            "wall_treatment",
            "conservation",
            "cross_method",
            "control",
        }
    )
    artifact: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "version": ARTIFACT_VERSION,
        "lattice": "D3Q27",
        "collision": "MRT",
        "dtype": "float64",
        "device": "cpu",
        "config": {
            "tau": tau,
            "shear_grids": list(grids),
            "sound_grids": [min(grids), max(grids)],
            "wall_grids": list(wall_grids),
            "physics_tolerance": PHYSICS_TOLERANCE,
        },
        "measurements": {
            "shear_wave": shear,
            "sound": sound,
            "spectral": spectral,
            "wall_mode": walls,
            "conservation": conservation,
            "controls": controls,
        },
        "checks": checks,
        "summary": {
            "n_checks": len(checks),
            "n_pass": sum(1 for check in checks if check["status"] == "pass"),
            "n_fail": sum(1 for check in checks if check["status"] == "fail"),
            "max_physics_rel_error": max(
                check["rel_error"] for check in checks if check["tolerance"] == PHYSICS_TOLERANCE
            ),
        },
        "evidence_tier": "physical_validation"
        if all(c["status"] == "pass" for c in checks)
        else "physical_validation_incomplete",
        "physically_validated_aspects": physically_validated,
        "withheld": [
            {
                "aspect": "turbulence_models",
                "reason": "no analytic/LES reference executed here; wall functions, SGS and RANS closures are out of scope",
            },
            {
                "aspect": "multiphase_and_free_surface",
                "reason": "single-phase anchors only",
            },
            {
                "aspect": "moving_boundaries_and_AMR",
                "reason": "static geometry and uniform periodic mesh only",
            },
            {
                "aspect": "accelerator_backends",
                "reason": "CPU float64 evidence only",
            },
            {
                "aspect": "high_reynolds_stability",
                "reason": "anchors run at Re = O(1) in lattice units; no long-time or high-Re stability claim",
            },
        ],
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON rendering (self-hash excluded)."""
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ARTIFACT_ID",
    "ARTIFACT_VERSION",
    "PHYSICS_TOLERANCE",
    "measure_shear_wave_viscosity",
    "measure_sound_speed",
    "spectral_step_eigenvalues",
    "measure_wall_mode_decay",
    "check_periodic_conservation",
    "check_controls",
    "run_d3q27_mrt_physical_validation_probe",
]
