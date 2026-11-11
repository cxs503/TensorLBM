"""D3Q27 MRT square-duct Poiseuille anchor (physical-accuracy evidence).

Where :mod:`tensorlbm.d3q27_mrt_physical_validation` anchors the *periodic*
operator (shear-wave viscosity, sound speed, wall eigenmode, spectral
eigenvalues) and :mod:`tensorlbm.d3q27_full_composition_test` proves the
wall/geometry/streaming composition is *executable and self-consistent*, this
module supplies the missing anchor class:

    **a steady, doubly confined flow driven by a body force and checked
    against a closed-form solution plus a literature integral value.**

That combination is exactly the case that exposes corner behaviour in a
three-dimensional lattice (two bounce-back operators superposed at the duct
corners, wall shear contributed by two directions), and it was covered for
D3Q19 (``tests/test_lbm_verification.py`` Step E, 0.416 % at ``L = 10``) but
never for D3Q27.

Physics setup (all lattice units, ``rho = 1``):

* square duct of side ``L`` (fluid cross-section ``L x L`` in ``y``/``z``),
  periodic along ``x`` with an odd number of cells in the middle plane;
* three-dimensional static solid mask (two walls in ``y``, two in ``z``),
  half-way bounce-back via the library's own ``OPPOSITE27`` reflection;
* constant body force ``g`` along ``x`` injected as a population increment
  ``w_i * 3 * c_ix * g`` (the same convention Step E uses for D3Q19), applied
  to fluid cells only unless ``force_on_solid=True`` (disclosed control);
* collision ``collide_mrt27``; the ``s_e = s_eps = s_q = s_pi = 1/tau`` limit
  must reproduce ``collide_bgk27`` (the limit is *measured*, not assumed).

Reference solution (Boussinesq series, ``a = L/2``, walls at ``+-a``)::

    u(Y, Z) = 16 (g/nu) a^2 / pi^3 * sum_{n odd}
              (-1)^((n-1)/2) / n^3 * [1 - cosh(n pi Z / (2a)) / cosh(n pi / 2)]
              * cos(n pi Y / (2a))

The series is *itself* checked before it is used as a reference: wall values
vanish, four-fold symmetry holds, and the analytically differentiated
Laplacian equals ``-g/nu`` (no finite differences), so a coefficient or sign
error cannot masquerade as a small lattice error.  ``cosh`` ratios are
evaluated in exponential form to avoid overflow at large ``n``.

The literature anchor is the square-duct Poiseuille number ``f * Re = 56.91``
(Shah & London 1978 laminar duct tables; equivalently the dimensionless
resistance ``R = g L^2 / (nu * u_mean) = 28.455``), which is an *integral*
functional independent of the pointwise profile comparison.

Honest scope: single-phase, static geometry, ``cpu``, ``float64``; no
turbulence, multiphase, free surface, moving boundary, AMR or accelerator
backend.  This module is a cold-path evidence writer, not a solver hot path,
and it does **not** lift the ``general_capability_matrix`` WITHHELD
disposition for the complete D3Q27 platform composition (that requires the
platform-level composition path and its own acceptance record).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .boundaries_d3q27 import OPPOSITE27
from .d3q27 import C, W, collide_bgk27, collide_mrt27, equilibrium27, macroscopic27, stream27_roll

ARTIFACT_ID = "duct-poiseuille-validation-r1"
ARTIFACT_VERSION = "duct-poiseuille-validation-r1"

#: Ecosystem accuracy standard: a physics anchor within 3 % is acceptable.
PHYSICS_TOLERANCE = 0.03
#: Pointwise profile criterion used by the in-repo analytic ladder (stricter).
PROFILE_TOLERANCE = 0.01
#: Square-duct laminar Poiseuille number, Shah & London (1978) tables.
LITERATURE_POISEUILLE_NUMBER = 56.91
LITERATURE_RESISTANCE = LITERATURE_POISEUILLE_NUMBER / 2.0  # 28.455


# ---------------------------------------------------------------------------
# closed-form reference
# ---------------------------------------------------------------------------


def duct_coords(length: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Fluid-cell coordinates for a duct of side ``length`` (half-way walls)."""
    if length < 4:
        raise ValueError("length must be >= 4 to resolve the duct profile")
    return torch.arange(1, length + 1, dtype=dtype) - 0.5 * (length + 1.0)


def duct_series(
    Y: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    length: int,
    body_force: float,
    nu: float,
    *,
    n_terms: int = 201,
    order: int = 0,
) -> np.ndarray:
    """Boussinesq series for fully developed square-duct Poiseuille flow.

    ``order=0`` returns ``u``, ``order=2`` returns the *analytically*
    differentiated ``d2u/dY2 + d2u/dZ2`` (used to self-check the series
    against ``-g/nu`` without finite differences).
    """
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    a = 0.5 * float(length)
    prefactor = 16.0 * (body_force / nu) * a * a / math.pi**3
    total = np.zeros_like(np.broadcast_arrays(Y, Z)[0])
    for n in range(1, n_terms + 1, 2):
        sign = 1.0 if (n // 2) % 2 == 0 else -1.0
        k = n * math.pi / (2.0 * a)
        # cosh(k Z) / cosh(k a) in exponential form: no overflow at large n.
        kz = k * np.abs(Z)
        ratio = np.exp(-(k * a - kz)) * (1.0 + np.exp(-2.0 * kz)) / (1.0 + np.exp(-2.0 * k * a))
        cos_y = np.cos(k * Y)
        if order == 0:
            total = total + sign / n**3 * (1.0 - ratio) * cos_y
        elif order == 2:
            # d2/dY2 + d2/dZ2 of (1 - cosh(kZ)/cosh(ka)) cos(kY) = -k^2 cos(kY)
            total = total + sign / n**3 * (-(k * k)) * cos_y
        else:  # pragma: no cover - guarded for API clarity
            raise ValueError("order must be 0 or 2")
    return prefactor * total


def duct_profile_exact(
    length: int, body_force: float, nu: float, *, n_terms: int = 201
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u_exact[iz, iy], Y, Z)`` on the fluid-cell grid."""
    coord = duct_coords(length).numpy()
    Zg, Yg = np.meshgrid(coord, coord, indexing="ij")
    return duct_series(Yg, Zg, length, body_force, nu, n_terms=n_terms), Yg, Zg


def check_series(length: int = 10, body_force: float = 1.0 / 600.0, nu: float = 1.0 / 6.0,
                 *, n_terms: int = 401) -> dict[str, Any]:
    """Self-check the reference before trusting it as a reference."""
    coord = duct_coords(length).numpy()
    a = 0.5 * float(length)

    # 1. no-slip on all four walls.  Half-way bounce-back puts the wall at
    # +-a = +-L/2, i.e. half a lattice unit beyond the last fluid-cell centre
    # (which sits at +-(a-0.5)); evaluating *outside* that is meaningless and
    # makes cosh(kZ)/cosh(ka) overflow.
    wall = np.array([-a, a])
    Wg, Wz = np.meshgrid(wall, wall, indexing="ij")
    wall_max = float(np.max(np.abs(duct_series(Wg, Wz, length, body_force, nu, n_terms=n_terms))))

    # 2. four-fold symmetry
    Zg, Yg = np.meshgrid(coord, coord, indexing="ij")
    u = duct_series(Yg, Zg, length, body_force, nu, n_terms=n_terms)
    sym = float(
        max(
            np.max(np.abs(u - duct_series(-Yg, Zg, length, body_force, nu, n_terms=n_terms))),
            np.max(np.abs(u - duct_series(Yg, -Zg, length, body_force, nu, n_terms=n_terms))),
            np.max(np.abs(u - duct_series(Zg, Yg, length, body_force, nu, n_terms=n_terms))),
        )
    )

    # 3. analytically differentiated Laplacian == -g/nu
    lap = duct_series(Yg, Zg, length, body_force, nu, n_terms=n_terms, order=2)
    lap_err = float(np.max(np.abs(lap - (-body_force / nu))) / (body_force / nu))

    # 4. resolution of the series: window-averaged resistance vs literature
    u_mean = float(u.mean())
    resistance = body_force * length**2 / (nu * u_mean)
    return {
        "length": length,
        "n_terms": n_terms,
        "wall_value_max": wall_max,
        "symmetry_max_abs_diff": sym,
        "laplacian_rel_error": lap_err,
        "u_max": float(u.max()),
        "u_mean": u_mean,
        "resistance": resistance,
        "resistance_two_parameter_line": resistance,
    }


# ---------------------------------------------------------------------------
# D3Q27 simulation of the duct
# ---------------------------------------------------------------------------


def _duct_state(
    length: int, nx: int, *, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(f, solid_mask)`` for the duct; wall-solid shell included."""
    ny = nz = length + 2
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    rho = torch.ones((nz, ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho)
    f = equilibrium27(rho, zero, zero, zero)
    return f, solid


def run_duct_case(
    length: int = 10,
    *,
    nx: int = 4,
    steps: int = 6000,
    tau: float = 1.0,
    u_nominal: float = 0.05,
    collision: str = "mrt",
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
    force_on_solid: bool = False,
    dtype: torch.dtype = torch.float64,
    collect_profile: bool = True,
) -> dict[str, Any]:
    """Run one duct case and measure the error against the closed form."""
    if collision not in ("mrt", "bgk"):
        raise ValueError("collision must be 'mrt' or 'bgk'")
    nu = (tau - 0.5) / 3.0
    body_force = 8.0 * nu * u_nominal / length**2
    f, solid = _duct_state(length, nx, dtype=dtype)
    fluid = (~solid).to(dtype)

    c = C.to(dtype)
    w = W.to(dtype)
    accel = (w * 3.0 * c[:, 0]).view(27, 1, 1, 1) * body_force
    if not force_on_solid:
        accel = accel * fluid.unsqueeze(0)
    opp = OPPOSITE27.to(torch.long)

    u_max_prev = 0.0
    steady_tail: list[float] = []
    for step in range(1, steps + 1):
        f_pre = f
        if collision == "mrt":
            f = collide_mrt27(f_pre, tau, s_e=s_e, s_eps=s_eps, s_q=s_q, s_pi=s_pi)
        else:
            f = collide_bgk27(f_pre, tau)
        # No dynamics + half-way bounce-back on solid cells: the solid carries
        # the reflected *pre-collision* populations (post-collision snapshot
        # would inject spurious momentum at the wall).
        f = torch.where(solid.unsqueeze(0), f_pre[opp], f)
        f = f + accel
        f = stream27_roll(f)
        if step % max(1, steps // 20) == 0 or step == steps:
            _, ux, _, _ = macroscopic27(f)
            u_max = float(ux[~solid].max()) if bool((~solid).any()) else 0.0
            steady_tail.append(u_max)
            u_max_prev = u_max

    rho, ux, uy, uz = macroscopic27(f)
    u_num = ux.detach().numpy()[1:-1, 1:-1, nx // 2]
    u_exact, _, _ = duct_profile_exact(length, body_force, nu)
    u_max_exact = float(u_exact.max())
    rel_err = float(np.max(np.abs(u_num - u_exact)) / u_max_exact)
    rel_err_l1 = float(np.mean(np.abs(u_num - u_exact)) / u_max_exact)

    u_mean_num = float(u_num.mean())
    u_mean_exact = float(u_exact.mean())
    resistance_num = body_force * length**2 / (nu * u_mean_num)
    resistance_exact = body_force * length**2 / (nu * u_mean_exact)
    f_re_num = 2.0 * resistance_num
    f_re_ref = 2.0 * resistance_exact
    mass_drift = float(abs(rho.sum()) - rho.numel()) / rho.numel()

    tail = steady_tail[-5:]
    steady_rel_change = 0.0 if len(tail) < 2 else abs(tail[-1] - tail[0]) / abs(tail[-1])

    result: dict[str, Any] = {
        "case": "square_duct_poiseuille_27",
        "collision": collision,
        "length": length,
        "nx": nx,
        "steps": steps,
        "tau": tau,
        "nu": nu,
        "body_force": body_force,
        "s_e": s_e if collision == "mrt" else None,
        "s_eps": s_eps if collision == "mrt" else None,
        "s_q": s_q if collision == "mrt" else None,
        "s_pi": s_pi if collision == "mrt" else None,
        "force_on_solid": force_on_solid,
        "dtype": str(dtype).replace("torch.", ""),
        "u_max_num": float(u_num.max()),
        "u_max_exact": u_max_exact,
        "rel_error_max": rel_err,
        "rel_error_mean": rel_err_l1,
        "u_mean_num": u_mean_num,
        "u_mean_exact": u_mean_exact,
        "resistance_num": resistance_num,
        "resistance_exact": resistance_exact,
        "resistance_rel_error": abs(resistance_num - resistance_exact) / resistance_exact,
        "poiseuille_number_num": f_re_num,
        "poiseuille_number_exact": f_re_ref,
        "mass_drift": mass_drift,
        "steady_rel_change_tail": steady_rel_change,
        "max_uy": float(uy.detach().abs().max()),
        "max_uz": float(uz.detach().abs().max()),
    }
    if collect_profile:
        mid = nx // 2
        result["profile_y"] = duct_coords(length).tolist()
        result["profile_num_midz"] = ux.detach().numpy()[1:-1, 1:-1, mid][mid].tolist()
        result["profile_exact_midz"] = u_exact[mid].tolist()
    return result


# ---------------------------------------------------------------------------
# controls: does the acceptance actually discriminate?
# ---------------------------------------------------------------------------


def control_wrong_viscosity(length: int = 10, **kwargs: Any) -> dict[str, Any]:
    """Compare the (correct) flow against a *mis-specified* reference."""
    case = run_duct_case(length=length, collect_profile=True, **kwargs)
    wrong = duct_profile_exact(length, case["body_force"], case["nu"] * 1.25)[0]
    u_num = np.asarray(case["profile_num_midz"], dtype=np.float64)
    err = float(np.abs(u_num - wrong[len(wrong) // 2]).max() / wrong.max())
    return {
        "control": "viscosity_mis_specification",
        "nu_ratio": 1.25,
        "rel_error": err,
        "correct_rel_error": case["rel_error_max"],
        "discriminates": err > PROFILE_TOLERANCE,
    }


def control_partial_steps(
    length: int = 10, steps: int = 3000, *, full_error: float | None = None, **kwargs: Any
) -> dict[str, Any]:
    """A quarter-length run must not be better than the converged one.

    ``full_error`` is the error already measured on the converged rung, so the
    control costs one short run instead of repeating the full ladder point.
    """
    short = run_duct_case(length=length, steps=max(100, steps // 4), collect_profile=False, **kwargs)
    reference = short["rel_error_max"] if full_error is None else float(full_error)
    return {
        "control": "not_yet_converged",
        "steps_short": short["steps"],
        "rel_error_short": short["rel_error_max"],
        "steps_full": steps if full_error is not None else short["steps"],
        "rel_error_full": reference,
        "discriminates": short["rel_error_max"] > reference,
    }


def control_mrt_bgk_limit(length: int = 10, steps: int = 1200, tau: float = 1.0) -> dict[str, Any]:
    """``s_e = s_eps = s_q = s_pi = 1/tau`` must reproduce BGK bit-for-bit."""
    rate = 1.0 / tau
    mrt = run_duct_case(
        length=length,
        steps=steps,
        tau=tau,
        s_e=rate,
        s_eps=rate,
        s_q=rate,
        s_pi=rate,
        collect_profile=True,
    )
    bgk = run_duct_case(length=length, steps=steps, tau=tau, collision="bgk", collect_profile=True)
    diff = float(
        np.max(np.abs(np.asarray(mrt["profile_num_midz"]) - np.asarray(bgk["profile_num_midz"])))
    )
    scale = max(abs(mrt["u_max_num"]), 1e-30)
    return {
        "control": "mrt_reduces_to_bgk",
        "rel_error_mrt": mrt["rel_error_max"],
        "rel_error_bgk": bgk["rel_error_max"],
        "max_abs_profile_diff": diff,
        "rel_diff": diff / scale,
        "bitwise_identical_error_values": bool(mrt["rel_error_max"] == bgk["rel_error_max"]),
        "discriminates": diff / scale < 1e-10,
    }


def control_force_on_solid(length: int = 8, steps: int = 2000) -> dict[str, Any]:
    """Disclosed variant: body force added on solid cells as well."""
    fluid_only = run_duct_case(length=length, steps=steps, force_on_solid=False, collect_profile=False)
    everywhere = run_duct_case(length=length, steps=steps, force_on_solid=True, collect_profile=False)
    return {
        "control": "body_force_on_solid_cells",
        "rel_error_fluid_only": fluid_only["rel_error_max"],
        "rel_error_force_everywhere": everywhere["rel_error_max"],
        "delta": everywhere["rel_error_max"] - fluid_only["rel_error_max"],
    }


# ---------------------------------------------------------------------------
# probe assembly
# ---------------------------------------------------------------------------


def _canonical_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _check(name: str, measured: float, threshold: float, *, lower_is_better: bool = True,
           note: str = "") -> dict[str, Any]:
    passed = measured < threshold if lower_is_better else measured > threshold
    return {
        "name": name,
        "measured": measured,
        "threshold": threshold,
        "lower_is_better": lower_is_better,
        "passed": bool(passed),
        "note": note,
    }


#: Default resolution ladder and step counts (stationarity scales with L^2/nu).
LADDER: tuple[tuple[int, int], ...] = ((8, 2000), (10, 3000), (14, 6000))


def run_duct_poiseuille_validation(
    *, ladder: tuple[tuple[int, int], ...] = LADDER, controls: bool = True
) -> dict[str, Any]:
    """Run the series self-check, the lattice ladder and the controls."""
    reference = check_series(ladder[0][0])

    cases = [
        run_duct_case(length=length, steps=steps) for length, steps in ladder
    ]
    errors = [case["rel_error_max"] for case in cases]
    lengths = [case["length"] for case in cases]
    # Observed order from consecutive ladder rungs: e ~ L^-p.
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(lengths[i + 1] / lengths[i])
        for i in range(len(errors) - 1)
    ]
    finest = cases[-1]

    checks = [
        _check("series_wall_no_slip", reference["wall_value_max"], 1e-10,
               note="all four wall values of the closed form must vanish"),
        _check("series_four_fold_symmetry", reference["symmetry_max_abs_diff"], 1e-6,
               note="u(Y,Z)=u(-Y,Z)=u(Y,-Z)=u(Z,Y)"),
        _check("series_laplacian_matches_body_force", reference["laplacian_rel_error"], 1e-2,
               note="analytically differentiated Laplacian vs -g/nu (no finite differences)"),
        _check("duct_profile_error_finest", finest["rel_error_max"], PROFILE_TOLERANCE,
               note=f"max|u-u_exact|/u_max at L={finest['length']}"),
        _check("duct_resistance_literature", finest["resistance_rel_error"], PHYSICS_TOLERANCE,
               note="R = gL^2/(nu*ubar) vs Boussinesq-series exact for the same L"),
        _check("duct_poiseuille_number_literature",
               abs(finest["poiseuille_number_num"] - LITERATURE_POISEUILLE_NUMBER)
               / LITERATURE_POISEUILLE_NUMBER,
               0.03,
               note="f*Re = 2R vs Shah & London (1978) 56.91"),
        _check("duct_mass_drift", finest["mass_drift"], 1e-8,
               note="sum(rho) drift in the steadied state"),
        _check("duct_steady_tail", finest["steady_rel_change_tail"], 5e-3,
               note="u_max change over the last fifth of the run"),
        _check("duct_transverse_velocity_small",
               max(finest["max_uy"], finest["max_uz"]) / finest["u_max_num"], 5e-2,
               note="a fully developed duct flow must have negligible cross-flow"),
    ]

    control_records: list[dict[str, Any]] = []
    if controls:
        c_wrong = control_wrong_viscosity(length=10)
        converged_l10 = next((c["rel_error_max"] for c in cases if c["length"] == 10), None)
        c_short = control_partial_steps(length=10, steps=3000, full_error=converged_l10)
        c_limit = control_mrt_bgk_limit()
        c_solid = control_force_on_solid()
        control_records = [c_wrong, c_short, c_limit, c_solid]
        checks.append(
            _check("control_mis_specified_viscosity_is_rejected",
                   c_wrong["rel_error"], PROFILE_TOLERANCE, lower_is_better=False,
                   note="a 25 % viscosity error must not satisfy the 1 % gate")
        )
        checks.append(
            _check("control_short_run_is_rejected",
                   c_short["rel_error_short"], c_short["rel_error_full"], lower_is_better=False,
                   note="a quarter-length run must be worse than the converged one")
        )
        checks.append(
            _check("control_mrt_equals_bgk_in_limit", c_limit["rel_diff"], 1e-10,
                   note="s_e=s_eps=s_q=s_pi=1/tau reproduces BGK")
        )

    payload: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "version": ARTIFACT_VERSION,
        "library": "TensorLBM",
        "lattice": "D3Q27",
        "collision": "MRT",
        "reference": {
            "profile": "Boussinesq series, square duct, no-slip on four walls",
            "literature_poiseuille_number": LITERATURE_POISEUILLE_NUMBER,
            "literature_resistance": LITERATURE_RESISTANCE,
            "self_check": reference,
        },
        "conventions": {
            "forcing": "f += w_i*3*c_ix*g on fluid cells (Step E convention)",
            "walls": "solid cells carry opposite pre-collision populations (half-way bounce-back)",
            "nx": "4 periodic cells along the duct axis",
            "dtype": "float64",
            "device": "cpu",
        },
        "cases": cases,
        "ladder_lengths": lengths,
        "ladder_errors": errors,
        "observed_orders": orders,
        "checks": checks,
        "controls": control_records,
        "all_checks_passed": all(check["passed"] for check in checks),
        "general_capability_matrix_effect": (
            "none: D3Q27 complete platform composition stays WITHHELD; this anchor is a "
            "prerequisite record, not a disposition change"
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "artifact_sha256"}
    )
    return payload


def write_artifact(path: str | Path, payload: dict[str, Any] | None = None) -> Path:
    data = run_duct_poiseuille_validation() if payload is None else payload
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        default="artifacts/duct_poiseuille_validation/results.json",
        help="artifact path",
    )
    parser.add_argument("--quick", action="store_true", help="single rung, no controls")
    args = parser.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    # Tiny 3D tensors: 128 CPU threads cost far more in dispatch than they save.
    torch.set_num_threads(1)
    if args.quick:
        payload = run_duct_poiseuille_validation(ladder=((8, 3000),), controls=False)
    else:
        payload = run_duct_poiseuille_validation()
    target = write_artifact(args.json, payload)
    print(json.dumps({"artifact": str(target), "all_checks_passed": payload["all_checks_passed"],
                      "ladder_errors": payload["ladder_errors"],
                      "observed_orders": payload["observed_orders"]}, indent=2))
    return 0 if payload["all_checks_passed"] else 1


__all__ = [
    "ARTIFACT_ID",
    "ARTIFACT_VERSION",
    "LADDER",
    "LITERATURE_POISEUILLE_NUMBER",
    "LITERATURE_RESISTANCE",
    "PHYSICS_TOLERANCE",
    "PROFILE_TOLERANCE",
    "check_series",
    "control_force_on_solid",
    "control_mrt_bgk_limit",
    "control_partial_steps",
    "control_wrong_viscosity",
    "duct_coords",
    "duct_profile_exact",
    "duct_series",
    "main",
    "run_duct_case",
    "run_duct_poiseuille_validation",
    "write_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())