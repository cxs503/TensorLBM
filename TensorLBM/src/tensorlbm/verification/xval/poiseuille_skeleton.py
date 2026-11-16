"""Single-phase plane-Poiseuille **baseline skeleton** (OSS cross-validation).

This module holds the three pieces that are shared by every arm of the
cross-validation case ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``
and that must *not* be re-implemented per arm:

1. :func:`oss_units` -- the parameter set of the OSS baseline skeleton, derived
   with the **same unit-conversion semantics** as the reference implementation
   (lettuce ``PoiseuilleFlow2D.make_units``) so that the TensorLBM arm and the
   OSS arm are compared at identical lattice numbers.  Verified against the OSS
   code lines quoted in the docstring (no hand-copied numbers: the test suite
   recomputes them).
2. :func:`analytic_profile` -- the exact steady solution of
   ``nu u'' = -g``, ``u(0) = u(H) = 0``, i.e. the parabola whose zeroes sit on
   the *half-way* wall positions produced by half-way bounce-back.  This is the
   independent reference (it shares no code with either solver).
3. :func:`profile_kpis` / :func:`fit_effective_height` -- the error metrics and
   the geometry check, reduced identically for both arms.

and the harness that makes the skeleton falsifiable:

4. :func:`run_baseline_case` -- runs the in-repo D2Q9 channel from rest to
   steady state and returns the measured profile, its KPIs and its slip
   decomposition (:func:`slip_decomposition`);
5. :func:`convergence_study` -- the grid-refinement sweep with measured orders.

Without (4)/(5) every test of the skeleton compared analytic functions with
analytic functions and could not fail; see
``docs/XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO.md``.

Wall convention (identical in both codes, checked in the case report)
--------------------------------------------------------------------
With ``resolution = N`` lattice nodes across the channel and solid rows at
``i = 0`` and ``i = N-1``, half-way bounce-back places the walls half a lattice
spacing beyond the outermost *fluid* node centres, so the wall-to-wall height is

    H = N - 2      (lattice units)

and the fluid node centres are at ``y = 0.5, 1.5, ..., H - 0.5``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Sequence

import numpy as np

# lattice speed of sound of the D2Q9 stencil; lettuce uses the same 1/sqrt(3)
CS_LU = 1.0 / np.sqrt(3.0)


@dataclass(frozen=True)
class OSSBaseline:
    """Lattice-unit parameter set of the OSS single-phase Poiseuille skeleton.

    Derivation mirrors lettuce ``lettuce/_unit.py`` ``UnitConversion`` and
    ``lettuce/ext/_flows/poiseuille.py`` ``PoiseuilleFlow2D``::

        characteristic_length_lu  = resolution - 1            (make_units)
        characteristic_velocity_lu = cs * mach_number         (UnitConversion)
        viscosity_lu              = L_lu * u_lu / Re          (UnitConversion)
        relaxation_parameter_lu   = nu_lu / cs**2 + 0.5       (UnitConversion)
        acceleration_lu           = a_pu / (u_pu**2 / L_pu) * (u_lu**2 / L_lu)

    ``height_lu`` is *not* part of the OSS conversion: it follows from the
    half-way bounce-back geometry (``resolution - 2``), which is why the OSS
    expected peak velocity uses ``y`` measured from the half-lattice offset.
    """

    resolution: int
    reynolds_number: float
    mach_number: float
    acceleration_pu: float
    characteristic_length_pu: float
    characteristic_velocity_pu: float
    characteristic_length_lu: float
    characteristic_velocity_lu: float
    viscosity_lu: float
    relaxation_parameter_lu: float
    acceleration_lu: float
    height_lu: float
    u_peak_lu: float
    u_peak_pu: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "resolution": self.resolution,
            "reynolds_number": self.reynolds_number,
            "mach_number": self.mach_number,
            "acceleration_pu": self.acceleration_pu,
            "characteristic_length_pu": self.characteristic_length_pu,
            "characteristic_velocity_pu": self.characteristic_velocity_pu,
            "characteristic_length_lu": self.characteristic_length_lu,
            "characteristic_velocity_lu": self.characteristic_velocity_lu,
            "viscosity_lu": self.viscosity_lu,
            "relaxation_parameter_lu": self.relaxation_parameter_lu,
            "acceleration_lu": self.acceleration_lu,
            "height_lu": self.height_lu,
            "u_peak_lu": self.u_peak_lu,
            "u_peak_pu": self.u_peak_pu,
        }


def oss_units(
    resolution: int = 16,
    reynolds_number: float = 1.0,
    mach_number: float = 0.02,
    acceleration_pu: float = 0.001,
    characteristic_length_pu: float = 1.0,
    characteristic_velocity_pu: float = 1.0,
    force_scale: float = 1.0,
) -> OSSBaseline:
    """Parameters of the OSS baseline skeleton at ``resolution`` (identity above).

    ``force_scale`` multiplies the *lattice* acceleration only (sensitivity
    study of the forcing scheme at fixed grid and viscosity); ``1.0`` reproduces
    the OSS skeleton literally.
    """
    if resolution < 4:
        raise ValueError("resolution must be >= 4 nodes (H = resolution - 2 >= 2)")
    res = int(resolution)
    char_length_lu = float(res - 1)
    char_velocity_lu = CS_LU * mach_number
    viscosity_lu = char_length_lu * char_velocity_lu / reynolds_number
    tau_lu = viscosity_lu / CS_LU ** 2 + 0.5
    # lettuce UnitConversion.convert_acceleration_to_lu
    accel_lu = acceleration_pu / (characteristic_velocity_pu ** 2 / characteristic_length_pu) \
        * (char_velocity_lu ** 2 / char_length_lu)
    accel_lu *= force_scale
    height_lu = float(res - 2)
    u_peak_lu = accel_lu * height_lu ** 2 / (8.0 * viscosity_lu)
    u_peak_pu = u_peak_lu / char_velocity_lu * characteristic_velocity_pu
    return OSSBaseline(
        resolution=res,
        reynolds_number=float(reynolds_number),
        mach_number=float(mach_number),
        acceleration_pu=float(acceleration_pu),
        characteristic_length_pu=float(characteristic_length_pu),
        characteristic_velocity_pu=float(characteristic_velocity_pu),
        characteristic_length_lu=char_length_lu,
        characteristic_velocity_lu=char_velocity_lu,
        viscosity_lu=viscosity_lu,
        relaxation_parameter_lu=tau_lu,
        acceleration_lu=accel_lu,
        height_lu=height_lu,
        u_peak_lu=u_peak_lu,
        u_peak_pu=u_peak_pu,
    )


def wall_normal_coordinates(n_nodes: int) -> tuple[np.ndarray, float]:
    """Fluid node distances from the lower half-way wall and the height ``H``.

    ``n_nodes`` is the full lattice extent (fluid rows + 2 solid rows).
    """
    if n_nodes < 4:
        raise ValueError("need at least 4 lattice nodes")
    y = np.arange(1, n_nodes - 1, dtype=float) - 0.5
    return y, float(n_nodes - 2)


def analytic_profile(y: Sequence[float] | np.ndarray, height: float, force: float,
                     nu: float) -> np.ndarray:
    """Exact plane-Poiseuille velocity ``u(y) = g/(2 nu) * y (H - y)``."""
    y = np.asarray(y, dtype=float)
    return force / (2.0 * nu) * y * (height - y)


def analytic_solution_object(height: float, force: float, nu: float):
    """The same parabola via the in-repo layered solver (independent cross-check).

    ``interfaces=()`` reduces :func:`solve_layered` to a single layer; the linear
    solve then reproduces :func:`analytic_profile`.  Returning the object lets
    callers also use ``u_peak()`` / ``flow_rate()``.
    """
    from ..layered_poiseuille.analytic import solve_layered

    return solve_layered(height=height, interfaces=(), nus=(nu,), g=force)


def profile_kpis(y: Sequence[float], u_num: Sequence[float], u_exact: Sequence[float]) -> Dict[str, float]:
    """Normalised error metrics of one cross-section (all dimensionless).

    Only *fluid* node rows should be passed in: outside the half-way walls the
    parabola is not a statement about the discrete solution (the OSS
    ``analytic_solution`` evaluates it there anyway and returns a negative
    velocity at the wall rows -- excluded by construction here).
    """
    y = np.asarray(y, dtype=float)
    un = np.asarray(u_num, dtype=float)
    ue = np.asarray(u_exact, dtype=float)
    if not (y.shape == un.shape == ue.shape):
        raise ValueError("y, u_num, u_exact must have the same shape")
    scale = float(np.max(np.abs(ue)))
    if scale <= 0.0:
        raise ValueError("degenerate reference profile")
    d = un - ue
    q_num = float(np.trapezoid(un, y))
    q_ex = float(np.trapezoid(ue, y))
    pn = float(np.max(np.abs(un)))
    shape_num = un / pn if pn > 0 else un * 0.0
    shape_ex = ue / scale
    return {
        "eps_inf": float(np.max(np.abs(d)) / scale),
        "eps_l2_rel": float(np.linalg.norm(d) / np.linalg.norm(ue)),
        "u_peak_err": float(abs(pn - scale) / scale),
        "q_err": float(abs(q_num - q_ex) / abs(q_ex)),
        "y_peak_abs": float(abs(y[int(np.argmax(un))] - y[int(np.argmax(ue))])),
        "shape_err": float(np.max(np.abs(shape_num - shape_ex))) if pn > 0 else float("nan"),
    }


def fit_effective_height(y: Sequence[float], u: Sequence[float], force: float,
                         nu: float) -> np.ndarray:
    """Per-node effective wall-to-wall height implied by the numerical profile.

    Inverting ``u = g/(2 nu) y (H - y)`` for ``H`` at every node gives
    ``H_i = y_i + 2 nu u_i / (g y_i)``; a solver with the documented half-way
    walls returns ``H_i = H`` for every interior node (geometry check).
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    if force == 0.0:
        raise ValueError("force must be non-zero")
    return y + 2.0 * nu * u / (force * y)


def two_code_diff_metrics(u_a: Sequence[float], u_b: Sequence[float]) -> Mapping[str, float]:
    """Symmetric difference between two *independent* numerical profiles.

    Used for the run-vs-run comparison (OSS binary vs TensorLBM): normalised by
    the mean of the two peaks so that neither code is privileged.
    """
    ua = np.asarray(u_a, dtype=float)
    ub = np.asarray(u_b, dtype=float)
    if ua.shape != ub.shape:
        raise ValueError("profiles must have identical shape (same lattice, same rows)")
    scale = 0.5 * (float(np.max(np.abs(ua))) + float(np.max(np.abs(ub))))
    if scale <= 0.0:
        raise ValueError("degenerate profiles")
    return {
        "tv_rel": float(np.max(np.abs(ua - ub)) / scale),
        "l2_rel": float(np.linalg.norm(ua - ub) / (0.5 * (np.linalg.norm(ua) + np.linalg.norm(ub)))),
        "peak_ratio": float(float(np.max(np.abs(ua))) / float(np.max(np.abs(ub)))),
    }

# ---------------------------------------------------------------------------
# Solver binding: the skeleton as a *run-and-compare* harness
# ---------------------------------------------------------------------------
#
# Everything above describes the case; nothing above ever touches a lattice
# Boltzmann solver.  A baseline skeleton whose only consumer is an analytic
# function compared to another analytic function cannot fail, and therefore
# cannot detect a solver defect (the OSS ``PoiseuilleFlow2D`` skeleton exists
# precisely to *run* the flow and diff the profile).  The two functions below
# close that loop with the in-repo D2Q9 channel solver:
#
#   * :func:`run_baseline_case` -- one resolution, one steady-state run, the
#     same three KPIs used by the cross-validation arms;
#   * :func:`convergence_study` -- the grid-refinement sweep with the measured
#     orders of accuracy.
#
# The initial condition is **rest** (``u = 0``), not the analytic parabola:
# seeding the analytic solution would make the comparison circular.


def slip_decomposition(y: Sequence[float], u_num: Sequence[float],
                       u_exact: Sequence[float]) -> Dict[str, float]:
    """Split the profile error into a uniform slip and the residual shape error.

    Half-way bounce-back combined with a body-force source term does not impose
    ``u = 0`` exactly on the half-way wall: the source term is reflected by the
    boundary and leaves a uniform additive slip ``s``.  Because ``s`` is
    independent of ``y`` it is harmless for the *shape* of the profile but it
    dominates ``eps_inf``.  This function removes it:

        ``s = mean(u_num - u_exact)``,
        ``eps_inf_shape = max|u_num - s - u_exact| / max|u_exact|``.

    ``eps_l2_shape`` is the same removal in the L2 norm.  A solver that
    reproduces the parabola in the interior returns ``eps_inf_shape`` at round-off.
    """
    un = np.asarray(u_num, dtype=float)
    ue = np.asarray(u_exact, dtype=float)
    yy = np.asarray(y, dtype=float)
    if not (yy.shape == un.shape == ue.shape):
        raise ValueError("y, u_num, u_exact must have the same shape")
    scale = float(np.max(np.abs(ue)))
    if scale <= 0.0:
        raise ValueError("degenerate reference profile")
    slip = float(np.mean(un - ue))
    corrected = un - slip
    return {
        "slip": slip,
        "slip_rel": slip / scale,
        "eps_inf_shape": float(np.max(np.abs(corrected - ue)) / scale),
        "eps_l2_shape": float(np.linalg.norm(corrected - ue) / np.linalg.norm(ue)),
    }


@dataclass(frozen=True)
class BaselineRun:
    """Result of one baseline run at one resolution."""

    params: OSSBaseline
    y: np.ndarray
    u_num: np.ndarray
    u_exact: np.ndarray
    kpis: Dict[str, float]
    h_eff: np.ndarray
    steps: int
    converged: bool
    mass_drift: float
    forcing: str
    slip_kpis: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "params": self.params.as_dict(),
            "steps": self.steps,
            "converged": self.converged,
            "forcing": self.forcing,
            "mass_drift": self.mass_drift,
            "kpis": dict(self.kpis),
            "slip_kpis": dict(self.slip_kpis),
            "h_eff_spread": float(np.ptp(self.h_eff)),
            "h_eff_mean": float(np.mean(self.h_eff)),
            "u_peak_num": float(np.max(self.u_num)),
            "u_peak_exact": float(np.max(self.u_exact)),
        }


def _build_skeleton_channel(params: OSSBaseline, nx: int, forcing: str):
    """Channel with exactly one solid row per wall -> H = resolution - 2 fluid rows."""
    from ...ice_hydro.lbm_channel import ChannelConfig, D2Q9Channel

    n_nodes = params.resolution
    cfg = ChannelConfig(
        nx=int(nx),
        ny=int(n_nodes),
        tau=float(params.relaxation_parameter_lu),
        steps=0,
        skip=1,
        cover_amplitude=0.0,
        cover_depth=float(n_nodes - 1),   # top solid row only
        bc="no_slip",
        force=float(params.acceleration_lu),
        forcing=forcing,
    )
    return D2Q9Channel(cfg)


def run_baseline_case(
    resolution: int = 16,
    *,
    reynolds_number: float = 1.0,
    mach_number: float = 0.02,
    acceleration_pu: float = 0.001,
    force_scale: float = 1.0,
    nx: int = 2,
    forcing: str = "guo",
    nu_lu: float | None = None,
    force_lu: float | None = None,
    max_steps: int = 400_000,
    tol: float = 1e-10,
    window: int = 200,
) -> BaselineRun:
    """Run the OSS baseline case at ``resolution`` and diff against the parabola.

    The flow is driven from rest by the uniform body force of
    :func:`oss_units`; steady state is declared when the wall-normal peak
    velocity changes by less than ``tol`` (relative) over ``window`` steps, and
    the reported profile is the time average over one further window.

    ``nu_lu`` / ``force_lu`` override the OSS unit conversion so that the grid
    can be refined at *fixed* lattice viscosity and force (the textbook
    refinement).  The OSS convention fixes Re and Ma instead, which makes
    ``nu`` grow with the resolution -- the two refinements answer different
    questions and both are reported.

    Returns a :class:`BaselineRun` carrying the sampled profile, the analytic
    reference on the same nodes, the KPI dict of :func:`profile_kpis` and the
    per-node effective height of :func:`fit_effective_height`.
    """
    from ...ice_hydro.lbm_channel import CX

    params = oss_units(
        resolution,
        reynolds_number=reynolds_number,
        mach_number=mach_number,
        acceleration_pu=acceleration_pu,
        force_scale=force_scale,
    )
    if nu_lu is not None or force_lu is not None:
        nu_eff = float(nu_lu) if nu_lu is not None else params.viscosity_lu
        g_eff = float(force_lu) if force_lu is not None else params.acceleration_lu
        params = replace(
            params,
            viscosity_lu=nu_eff,
            relaxation_parameter_lu=nu_eff / CS_LU ** 2 + 0.5,
            acceleration_lu=g_eff,
            u_peak_lu=g_eff * params.height_lu ** 2 / (8.0 * nu_eff),
        )
    ch = _build_skeleton_channel(params, nx=nx, forcing=forcing)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    if n_fluid != int(params.height_lu):
        raise RuntimeError(
            f"geometry mismatch: {n_fluid} fluid rows vs H = {params.height_lu}"
        )

    y, height = wall_normal_coordinates(params.resolution)
    u_exact = analytic_profile(y, height, params.acceleration_lu, params.viscosity_lu)

    def profile() -> tuple[np.ndarray, float]:
        rho = ch.f.sum(axis=0)
        ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
        return ux[:, rows].mean(axis=0), float(rho[ch.fluid].mean())

    prev_peak = 0.0
    steps = 0
    converged = False
    while steps < max_steps:
        for _ in range(window):
            ch.f = ch._stream_from(ch._collide(ch.f))
        steps += window
        prof, _ = profile()
        peak = float(np.max(np.abs(prof)))
        if prev_peak > 0.0 and abs(peak - prev_peak) <= tol * peak:
            converged = True
            break
        prev_peak = peak

    acc = np.zeros(n_fluid)
    rho_acc = 0.0
    for _ in range(window):
        ch.f = ch._stream_from(ch._collide(ch.f))
        prof, rho_mean = profile()
        acc += prof
        rho_acc += rho_mean
    u_num = acc / window
    rho_mean = rho_acc / window

    return BaselineRun(
        params=params,
        y=y,
        u_num=u_num,
        u_exact=u_exact,
        kpis=profile_kpis(y, u_num, u_exact),
        h_eff=fit_effective_height(y, u_num, params.acceleration_lu, params.viscosity_lu),
        steps=steps,
        converged=converged,
        mass_drift=rho_mean - 1.0,
        forcing=forcing,
        slip_kpis=slip_decomposition(y, u_num, u_exact),
    )


def convergence_study(
    resolutions: Sequence[int] = (8, 16, 32, 64),
    **kwargs,
) -> Dict[str, object]:
    """Grid-refinement sweep; returns per-resolution KPIs and measured orders.

    Each resolution doubles the previous one, so the observed order of the error
    norm ``e`` follows from ``p = log2(e_coarse / e_fine)``.
    """
    res_list = [int(r) for r in resolutions]
    runs = [run_baseline_case(r, **kwargs) for r in res_list]
    table = []
    for r, run in zip(res_list, runs):
        row = {"resolution": r}
        row.update(run.kpis)
        row.update({
            "steps": run.steps,
            "converged": run.converged,
            "mass_drift": run.mass_drift,
            "h_eff_spread": float(np.ptp(run.h_eff)),
        })
        table.append(row)
    orders = []
    for i in range(1, len(table)):
        prev, cur = table[i - 1], table[i]
        factor = cur["resolution"] / prev["resolution"]
        order = {}
        if not np.isclose(factor, 2.0):
            order = {"order_inf": float("nan"), "order_l2": float("nan")}
        else:
            for key in ("eps_inf", "eps_l2_rel"):
                num, den = prev[key], cur[key]
                order[f"order_{key}"] = float(np.log(num / den) / np.log(2.0)) if den > 0 else float("nan")
        orders.append({"resolution_pair": (prev["resolution"], cur["resolution"]), **order})
    return {"table": table, "orders": orders, "runs": runs}
