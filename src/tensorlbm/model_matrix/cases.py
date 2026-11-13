"""Analytic verification cases for the model matrix.

Every case is *lattice-generic* and *model-agnostic*: it needs only the
repo-native ``equilibrium``/``macroscopic``/``stream`` trio of the model's
lattice, so one implementation can be applied to every registry entry.  Each
case carries its own reference, its own tolerance and its own enforcement
level; none of them inspects the model internals.

Cases (all periodic, no boundary condition -- so a failure is attributable to
the collision operator alone):

``feq_fixed_point``  (:class:`FeqFixedPoint`)
    The discrete Maxwell-Boltzmann equilibrium is a fixed point of *every*
    collision operator and the collision conserves mass and momentum exactly.
    Reference = exact algebraic identity (no numerical tolerance on physics):
    ``||C(f_eq) - f_eq||_inf`` and ``|rho' - rho|``, ``|u' - u|`` must be at
    round-off level.  Catches operators that mis-handle the conserved moments
    or that secretly change the equation of state.

``shear_decay``  (:class:`ShearDecay`)
    Transverse shear wave ``u_x = U0 cos(k y)`` in a periodic box decays as
    ``U0 exp(-nu k^2 t)`` with ``nu = (tau - 1/2)/3`` -- a closed-form solution
    of the incompressible Navier-Stokes equations and the classical
    "viscosity of a collision operator" check (used e.g. by the Palabos/TCLB
    regression suites).  Metric = fitted decay rate vs the analytic value.
    This is the gate: the molecular viscosity of a model is a property of its
    definition, so a deviation larger than the tolerance is a defect, not a
    numerical accident.

``tg_decay``  (:func:`run_taylor_green_decay`)
    Taylor-Green vortex at ``U0 = 0.1`` (Ma ~ 0.17): ``u = U0 (sin kx cos ky,
    -cos kx sin ky)``, kinetic energy decays as ``E0 exp(-4 nu k^2 t)`` in the
    incompressible limit (the field carries modes (k,k), so nu*lap u = -2 nu k^2 u).  Unlike the shear wave -- where ``(u.grad)u = 0``
    identically, so *no* Mach dependence can be measured -- the TG vortex has a
    genuine ``O(Ma^2)`` compressibility error, which is why it is the
    *advisory* case: the deviation is recorded and reported but does not fail
    the build.  (This replaces the originally planned finite-Mach shear wave,
    whose two Mach numbers returned identical viscosities to 6 digits -- see
    `docs/MODEL_MATRIX_CASE_SELECTION.md`.)

Nothing in this module hard-codes a benchmark result: references are formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .registry import ModelSpec, build_collider, lattice_ops, nu_from_tau


@dataclass(frozen=True)
class CaseResult:
    case: str
    model: str
    status: str
    """PASS / FAIL / GUARD / NOT_APPLICABLE / MISSING / ERROR."""
    metrics: dict[str, float]
    ref: dict[str, Any]
    tolerance: dict[str, Any]
    notes: str = ""
    error: str = ""
    curves: dict[str, list[float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "case": self.case,
            "model": self.model,
            "status": self.status,
            "metrics": self.metrics,
            "ref": self.ref,
            "tolerance": self.tolerance,
            "notes": self.notes,
        }
        if self.error:
            d["error"] = self.error
        if self.curves is not None:
            d["curves"] = self.curves
        return d


# --------------------------------------------------------------------------- #
#  Geometry / common machinery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Grid:
    """Periodic box; ``nz = 1`` keeps the 3-D lattices cheap."""

    nx: int = 32
    ny: int = 32
    nz: int = 1

    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def k(self) -> float:
        """Fundamental wavenumber of the shear wave along y."""
        return 2.0 * math.pi / self.ny


def _shear_field(spec: ModelSpec, grid: Grid, u0: float, dtype: torch.dtype) -> torch.Tensor:
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    y = torch.arange(grid.ny, dtype=dtype).view(1, grid.ny, 1)
    cos = torch.cos(grid.k * y)
    ux = u0 * cos.expand(shape).contiguous()
    zero = torch.zeros(shape, dtype=dtype)
    rho = torch.ones(shape, dtype=dtype)
    return ops["equilibrium"](rho, ux, zero, zero).to(dtype)


def _amplitude(f: torch.Tensor, grid: Grid, spec: ModelSpec) -> float:
    """Project the velocity field onto the cos(k y) mode (mode amplitude)."""
    ops = lattice_ops(spec.lattice)
    _rho, ux, _uy, _uz = ops["macroscopic"](f)
    y = torch.arange(grid.ny, dtype=ux.dtype).view(1, grid.ny, 1)
    cos = torch.cos(grid.k * y)
    return float(2.0 * (ux * cos).mean())


def _mass(f: torch.Tensor, spec: ModelSpec) -> float:
    rho, _ux, _uy, _uz = lattice_ops(spec.lattice)["macroscopic"](f)
    return float(rho.mean())


def _fit_decay_rate(times: list[float], amps: list[float], floor_ratio: float) -> tuple[float, float]:
    """Least-squares slope of ``ln|A|`` vs ``t`` over the useful window.

    Returns ``(rate, r2)`` where ``rate`` is the *decay* rate (positive).  The
    window starts at the first sample (amplitude ``U0``) and stops when the
    amplitude has decayed to ``floor_ratio * U0``, so the fit never leans on
    round-off noise of an almost-dead wave.
    """
    if not times or not amps:
        raise ValueError("empty decay series")
    a0 = abs(amps[0])
    keep = [(t, a) for t, a in zip(times, amps) if abs(a) > floor_ratio * a0]
    if len(keep) < 3:
        raise ValueError(f"decay window too short ({len(keep)} samples)")
    ts = torch.tensor([t for t, _ in keep], dtype=torch.float64)
    ls = torch.log(torch.tensor([abs(a) for _, a in keep], dtype=torch.float64))
    tm, lm = ts.mean(), ls.mean()
    slope = float(((ts - tm) * (ls - lm)).sum() / ((ts - tm) ** 2).sum())
    ss_tot = float(((ls - lm) ** 2).sum())
    ss_res = float(((ls - (lm + slope * (ts - tm))) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return -slope, r2


# --------------------------------------------------------------------------- #
#  Case: equilibrium fixed point
# --------------------------------------------------------------------------- #


def run_feq_fixed_point(spec: ModelSpec, grid: Grid, tau: float, *, dtype=torch.float64) -> dict:
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    rho = torch.full(shape, 1.0 + 0.1, dtype=dtype)
    ux = torch.full(shape, 0.02, dtype=dtype)
    uy = torch.full(shape, -0.01, dtype=dtype)
    uz = torch.full(shape, 0.03, dtype=dtype)
    feq = ops["equilibrium"](rho, ux, uy, uz).to(dtype)
    collider = build_collider(spec.name, tau)
    f_post = collider(feq)
    scale = float(feq.abs().max())
    d_inf = float((f_post - feq).abs().max()) / scale
    rho2, ux2, uy2, uz2 = ops["macroscopic"](f_post)
    d_rho = float((rho2 - rho).abs().max())
    d_u = max(
        float((ux2 - ux).abs().max()),
        float((uy2 - uy).abs().max()),
        float((uz2 - uz).abs().max()),
    )
    tol = 1e-6 if dtype == torch.float64 else 1e-5
    ok = max(d_inf, d_rho, d_u) <= tol
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "rel_feq_fixed_point_error": d_inf,
            "max_abs_rho_error": d_rho,
            "max_abs_velocity_error": d_u,
        },
        "ref": {
            "statement": "f_eq is a fixed point of the collision operator; rho and u are conserved",
            "expected_rel_feq_fixed_point_error": 0.0,
            "expected_max_abs_rho_error": 0.0,
            "expected_max_abs_velocity_error": 0.0,
        },
        "tolerance": {"max_rel_error": tol, "dtype": str(dtype)},
        "notes": "exact algebraic identity; tolerance is round-off of the working dtype",
    }


# --------------------------------------------------------------------------- #
#  Case: shear-wave decay (analytic viscosity)
# --------------------------------------------------------------------------- #


def run_shear_decay(
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    u0: float = 0.01,
    steps: int | None = None,
    dtype=torch.float64,
    tol_pct: float = 3.0,
    enforcement: str = "gate",
    floor_ratio: float = 0.25,
) -> dict:
    k = grid.k
    nu_ref = nu_from_tau(tau)
    omega = nu_ref * k * k
    if steps is None:
        # ~1.6 e-folds of decay: long enough to fit a slope, short enough to
        # stay far from the round-off floor.
        steps = max(40, int(round(1.6 / omega)))

    collider = build_collider(spec.name, tau)
    stream = lattice_ops(spec.lattice)["stream"]

    f = _shear_field(spec, grid, u0, dtype)
    m0 = _mass(f, spec)
    times: list[float] = [0.0]
    amps: list[float] = [_amplitude(f, grid, spec)]
    mass_drift = 0.0
    finite = True
    # Diagnostics (amplitude projection + mass) are sampled, not per step: the
    # invariants are smooth in time and every macroscopic call costs about as
    # much as a time step on these small boxes.
    sample_every = max(1, steps // 16)
    for n in range(1, steps + 1):
        f = stream(collider(f))
        if n % sample_every == 0 or n == steps:
            times.append(float(n))
            amps.append(_amplitude(f, grid, spec))
            mass_drift = max(mass_drift, abs(_mass(f, spec) - m0) / m0)
            if not torch.isfinite(f).all():
                finite = False
                break

    if not finite:
        return {
            "status": "FAIL",
            "metrics": {"nonfinite": 1.0},
            "ref": {"nu": nu_ref, "decay_rate": omega, "u0": u0},
            "tolerance": {"nu_rel_error_pct": tol_pct},
            "notes": "distribution went non-finite before the fit window closed",
        }

    rate, r2 = _fit_decay_rate(times, amps, floor_ratio)
    nu_fit = rate / (k * k)
    # analytic reference series for the same samples (for the figure)
    ref_amps = [u0 * math.exp(-omega * t) for t in times]
    err_pct = 100.0 * (nu_fit - nu_ref) / nu_ref
    field_err = 0.0
    if amps and ref_amps:
        num = sum((a - r) ** 2 for a, r in zip(amps, ref_amps))
        den = sum(r * r for r in ref_amps)
        field_err = 100.0 * math.sqrt(num / den) if den > 0 else 0.0

    # criterion: fitted viscosity within tol of the analytic value AND mass
    # conserved at round-off level AND the fit is a genuine exponential.
    ok = abs(err_pct) <= tol_pct and mass_drift <= 1e-8 and r2 >= 0.999
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "nu_fit": nu_fit,
            "nu_ref": nu_ref,
            "nu_rel_error_pct": err_pct,
            "decay_rate_fit": rate,
            "decay_rate_ref": omega,
            "amplitude_series_rel_l2_pct": field_err,
            "fit_r2": r2,
            "mass_drift_rel": mass_drift,
            "u0": u0,
            "mach": u0 / math.sqrt(1.0 / 3.0),
            "steps": float(steps),
            "k": k,
            "tau": tau,
        },
        "ref": {
            "solution": "u_x(y,t) = u0 cos(k y) exp(-nu k^2 t), periodic",
            "nu": nu_ref,
            "decay_rate": omega,
            "nu_formula": "(tau - 0.5) / 3",
        },
        "tolerance": {
            "nu_rel_error_pct": tol_pct,
            "mass_drift_rel": 1e-8,
            "fit_r2_min": 0.999,
            "enforcement": enforcement,
        },
        "notes": "gate case: molecular viscosity is part of the model definition",
        "curves": {"t": times, "amplitude": amps, "amplitude_analytic": ref_amps},
    }


# --------------------------------------------------------------------------- #
#  Case: Taylor-Green vortex decay (advisory, finite-Mach probe)
# --------------------------------------------------------------------------- #


def _tg_field(spec: ModelSpec, grid: Grid, u0: float, dtype: torch.dtype) -> torch.Tensor:
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    x = torch.arange(grid.nx, dtype=dtype).view(1, 1, grid.nx)
    y = torch.arange(grid.ny, dtype=dtype).view(1, grid.ny, 1)
    k = grid.k
    ux = (u0 * torch.sin(k * x) * torch.cos(k * y)).expand(shape).contiguous()
    uy = (-u0 * torch.cos(k * x) * torch.sin(k * y)).expand(shape).contiguous()
    zero = torch.zeros(shape, dtype=dtype)
    rho = torch.ones(shape, dtype=dtype)
    return ops["equilibrium"](rho, ux, uy, zero).to(dtype)


def _kinetic_energy(f: torch.Tensor, spec: ModelSpec) -> float:
    _rho, ux, uy, uz = lattice_ops(spec.lattice)["macroscopic"](f)
    return 0.5 * float((ux * ux + uy * uy + uz * uz).mean())


def run_taylor_green_decay(
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    u0: float = 0.1,
    steps: int | None = None,
    dtype=torch.float64,
    tol_pct: float = 5.0,
    enforcement: str = "advisory",
    floor_ratio: float = 0.25,
) -> dict:
    k = grid.k
    nu_ref = nu_from_tau(tau)
    # The TG field above carries modes (k, k), so laplacian(u) = -2 k^2 u and the
    # velocity decays at 2 nu k^2; the kinetic energy E ~ u^2 therefore decays
    # twice as fast: dE/dt = -4 nu k^2 E.
    rate_ref = 4.0 * nu_ref * k * k
    if steps is None:
        steps = max(40, int(round(1.4 / rate_ref)))
    collider = build_collider(spec.name, tau)
    stream = lattice_ops(spec.lattice)["stream"]

    f = _tg_field(spec, grid, u0, dtype)
    times = [0.0]
    energies = [_kinetic_energy(f, spec)]
    finite = True
    sample_every = max(1, steps // 16)
    for n in range(1, steps + 1):
        f = stream(collider(f))
        if n % sample_every == 0 or n == steps:
            times.append(float(n))
            energies.append(_kinetic_energy(f, spec))
            if not torch.isfinite(f).all():
                finite = False
                break
    if not finite:
        return {
            "status": "FAIL",
            "metrics": {"nonfinite": 1.0},
            "ref": {"nu": nu_ref, "decay_rate": rate_ref, "u0": u0},
            "tolerance": {"nu_rel_error_pct": tol_pct},
            "notes": "distribution went non-finite before the fit window closed",
        }
    rate, r2 = _fit_decay_rate(times, energies, floor_ratio)
    nu_fit = rate / (4.0 * k * k)
    err_pct = 100.0 * (nu_fit - nu_ref) / nu_ref
    ref_energies = [energies[0] * math.exp(-rate_ref * t) for t in times]
    num = sum((e - r) ** 2 for e, r in zip(energies, ref_energies))
    den = sum(r * r for r in ref_energies)
    energy_err = 100.0 * math.sqrt(num / den) if den > 0 else 0.0
    ok = abs(err_pct) <= tol_pct
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "nu_fit": nu_fit,
            "nu_ref": nu_ref,
            "nu_rel_error_pct": err_pct,
            "decay_rate_fit": rate,
            "decay_rate_ref": rate_ref,
            "energy_series_rel_l2_pct": energy_err,
            "fit_r2": r2,
            "u0": u0,
            "mach": u0 / math.sqrt(1.0 / 3.0),
            "steps": float(steps),
            "k": k,
            "tau": tau,
        },
        "ref": {
            "solution": "Taylor-Green (modes k,k): E(t) = E0 exp(-4 nu k^2 t) (incompressible limit)",
            "nu": nu_ref,
            "decay_rate": rate_ref,
            "nu_formula": "(tau - 0.5) / 3",
        },
        "tolerance": {"nu_rel_error_pct": tol_pct, "enforcement": enforcement},
        "notes": "advisory: deviations carry the O(Ma^2) compressibility error",
        "curves": {"t": times, "energy": energies, "energy_analytic": ref_energies},
    }


# --------------------------------------------------------------------------- #
#  Case registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    enforcement: str
    runner: Callable[..., dict]
    tol_pct: float
    metrics_doc: str


CASES: dict[str, CaseSpec] = {
    "feq_fixed_point": CaseSpec(
        name="feq_fixed_point",
        description="equilibrium fixed point + exact conservation of rho and u (algebraic identity)",
        enforcement="gate",
        runner=run_feq_fixed_point,
        tol_pct=0.0,
        metrics_doc="max relative |C(f_eq)-f_eq|, max |rho'-rho|, max |u'-u|",
    ),
    "shear_decay": CaseSpec(
        name="shear_decay",
        description="periodic shear wave decay: fitted nu vs analytic nu=(tau-1/2)/3, u0=0.01",
        enforcement="gate",
        runner=run_shear_decay,
        tol_pct=3.0,
        metrics_doc="nu_fit, nu_ref, nu_rel_error_pct, fit_r2, mass_drift_rel",
    ),
    "tg_decay": CaseSpec(
        name="tg_decay",
        description="Taylor-Green vortex at u0=0.1: fitted nu vs analytic, advisory (O(Ma^2) probe)",
        enforcement="advisory",
        runner=run_taylor_green_decay,
        tol_pct=5.0,
        metrics_doc="nu_fit, nu_ref, nu_rel_error_pct (advisory), energy_series_rel_l2_pct",
    ),
}


def case_names() -> list[str]:
    return sorted(CASES)


def run_case(
    case_name: str,
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    dtype=torch.float64,
) -> CaseResult:
    cs = CASES[case_name]
    if case_name in spec.inapplicable_reason:
        return CaseResult(
            case=case_name,
            model=spec.name,
            status="NOT_APPLICABLE",
            metrics={},
            ref={},
            tolerance={"note": spec.inapplicable_reason[case_name]},
            notes=spec.inapplicable_reason[case_name],
        )
    if case_name not in spec.cases:
        return CaseResult(
            case=case_name,
            model=spec.name,
            status="MISSING",
            metrics={},
            ref={},
            tolerance={},
            notes="case is not declared for this model (coverage hole)",
        )
    try:
        if case_name == "tg_decay":
            out = cs.runner(spec, grid, tau, dtype=dtype, tol_pct=cs.tol_pct, enforcement="advisory")
        elif case_name == "shear_decay":
            out = cs.runner(spec, grid, tau, dtype=dtype, tol_pct=cs.tol_pct, enforcement="gate")
        else:
            out = cs.runner(spec, grid, tau, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - a crashing model is a matrix result
        return CaseResult(
            case=case_name,
            model=spec.name,
            status="ERROR",
            metrics={},
            ref={},
            tolerance={},
            error=f"{type(exc).__name__}: {exc}",
        )
    if spec.enforcement == "guard" and out["status"] == "PASS" and case_name != "feq_fixed_point":
        out["status"] = "GUARD"
    return CaseResult(
        case=case_name,
        model=spec.name,
        status=out["status"],
        metrics=out["metrics"],
        ref=out["ref"],
        tolerance=out["tolerance"],
        notes=out["notes"],
        curves=out.get("curves"),
    )