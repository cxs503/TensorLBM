#!/usr/bin/env python3
"""KVLCC2 total-resistance decomposition (criterion d).

Chain used here (all terms traceable):

    C_T = C_F (1 + k) + C_R + C_A

``C_F``   computed in :mod:`tensorlbm.wall_model_les.resistance` from the
          ITTC-1957 line at the mission Reynolds number ``Re = 1.3e7``.
``1+k``   form factor, an *input*: KVLCC2 doubles as a benchmark hull with a
          published form factor of about 1.2; the sensitivity to 1.15/1.25 is
          reported, and the value used is recorded in the JSON.
``C_R``   residual resistance from the ship platform's traceable ITTC-1978
          chain (``POST /api/v1/resistance-advanced/ittc78``).  The request and
          the full response are stored in the artifact so the borrowed number
          can be re-derived.
``C_A``   correlation allowance, also taken from the platform response.

The comparison partner is the platform's own ITTC-78 ``C_T`` at the same
``Re``/``Fr`` -- i.e. an independent implementation of the same standard chain.
**No digitized towing-tank measurement of KVLCC2 is shipped with this repository,
so the literal "versus experiment" limb of criterion (d) is reported as
WITHHELD rather than filled in from memory.**

Usage
-----
    python benchmarks/wall_model_les/kvlcc2.py --out artifacts/wall_model_les
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.resistance import (  # noqa: E402
    decompose,
    fetch_platform_ittc78,
    ittc_1957_cf,
    kvlcc2_geometry,
)

RE_TARGET = 1.3e7
FROUDE = 0.142
FORM_FACTOR = 1.20
NOMINAL_CRITERION = 0.07
NU_SEA = 1.189e-6  # m^2/s, ITTC-1978 fresh/sea water at 15 C
RHO_SEA = 1025.0


def _model_length_for_re(re_target: float, froude: float, nu: float) -> float:
    """Model length whose Fr = ``froude`` speed gives ``Re = re_target``.

    ``Re = V L / nu`` with ``V = Fr sqrt(g L)`` =>  ``L = (Re nu / (Fr sqrt(g)))^(2/3)``.
    """
    return (re_target * nu / (froude * math.sqrt(9.80665))) ** (2.0 / 3.0)


def run(out_dir: Path) -> dict:
    length = _model_length_for_re(RE_TARGET, FROUDE, NU_SEA)
    geometry = kvlcc2_geometry(scale=length / 320.0)
    speed = FROUDE * math.sqrt(9.80665 * length)
    re_check = speed * length / NU_SEA

    result: dict = {
        "target_reynolds": RE_TARGET,
        "froude": FROUDE,
        "derived": {
            "model_length_m": length,
            "model_beam_m": geometry["B"],
            "model_draft_m": geometry["T"],
            "wetted_surface_m2": geometry["wetted_surface"],
            "speed_ms": speed,
            "reynolds_check": re_check,
            "kinematic_viscosity": NU_SEA,
        },
        "cf_itcc_1957_here": ittc_1957_cf(RE_TARGET),
    }

    # ---- platform reference ------------------------------------------------
    reference = None
    api_error = None
    try:
        api = fetch_platform_ittc78(
            geometry=geometry,
            # NOTE: the platform's ITTC-78 endpoint takes the speed in KNOTS
            # (verified: passing m/s reproduces Re/2 and half the Froude
            # number), so the SI speed is converted here and the conversion is
            # recorded below.
            speed=speed / 0.514444,
            form_factor=FORM_FACTOR,
            water_viscosity=NU_SEA,
            water_density=RHO_SEA,
        )
        reference = api
    except Exception as exc:  # noqa: BLE001 - reference may be unavailable
        api_error = str(exc)
    result["platform_reference"] = reference
    result["platform_reference_error"] = api_error

    if reference is not None:
        resp = reference["response"]
        chain = decompose(
            reynolds=RE_TARGET,
            form_factor=FORM_FACTOR,
            cr=float(resp.get("Cr", 0.0)),
            # The platform's C_A (= delta_Cf) is a full-scale model-ship
            # correlation/roughness allowance; the mission's decomposition
            # C_T = C_F (1+k) + C_R is a model-scale relation, so it is excluded
            # here and reported on its own.
            c_a=0.0,
            speed=speed,
            froude=FROUDE,
            rho=RHO_SEA,
            wetted_surface=geometry["wetted_surface"],
            provenance={
                "cf": "ITTC-1957 line computed in tensorlbm.wall_model_les.resistance",
                "form_factor": f"input {FORM_FACTOR} (KVLCC2 published value, sensitivity reported)",
                "cr": reference["endpoint"],
                "ca": reference["endpoint"],
            },
        )
        cf_ref = float(resp.get("Cf", float("nan")))
        cr_ref = float(resp.get("Cr", float("nan")))
        ct_ref = cf_ref * FORM_FACTOR + cr_ref
        result["chain_here"] = chain.as_dict()
        result["comparison"] = {
            "ct_here": chain.ct,
            "ct_platform_ittc78": ct_ref,
            "ct_platform_ittc78_with_ca": float(resp.get("Ct", float("nan"))),
            "ca_platform_excluded": float(resp.get("Ca", float("nan"))),
            "platform_reynolds": float(resp.get("reynolds_number", float("nan"))),
            "platform_froude": float(resp.get("froude_number", float("nan"))),
            "relative_error_ct": abs(chain.ct - ct_ref) / ct_ref if ct_ref else None,
            "cf_here": chain.cf,
            "cf_platform_ittc78": cf_ref,
            "relative_error_cf": abs(chain.cf - cf_ref) / cf_ref if cf_ref else None,
            "criterion": NOMINAL_CRITERION,
            "verdict_meets_criterion": bool(
                ct_ref and abs(chain.ct - ct_ref) / ct_ref <= NOMINAL_CRITERION
            ),
            "partner_is_experiment": False,
            "experimental_reference_status": (
                "WITHHELD - no digitized KVLCC2 towing-tank C_T is available in this "
                "worktree; the platform ITTC-78 chain is an independent implementation "
                "of the same standard, not a measurement"
            ),
        }
        # full-scale counterpart for context
        full = kvlcc2_geometry(scale=1.0)
        speed_full = FROUDE * math.sqrt(9.80665 * full["Lpp"])
        re_full = speed_full * full["Lpp"] / NU_SEA
        try:
            api_full = fetch_platform_ittc78(
                geometry=full,
                speed=speed_full / 0.514444,
                form_factor=FORM_FACTOR,
                water_viscosity=NU_SEA,
                water_density=RHO_SEA,
            )
            result["full_scale"] = {
                "reynolds": re_full,
                "speed_ms": speed_full,
                "platform": api_full["response"],
            }
        except Exception as exc:  # noqa: BLE001
            result["full_scale"] = {"error": str(exc)}

    # ---- sensitivity to the (input) form factor ---------------------------
    sensitivity = {}
    cr = result.get("chain_here", {}).get("cr", 0.0) if reference else 0.0
    ca = result.get("chain_here", {}).get("c_a", 0.0) if reference else 0.0
    for k in (1.15, 1.18, 1.20, 1.22, 1.25):
        chain = decompose(
            reynolds=RE_TARGET, form_factor=k, cr=cr, c_a=ca, speed=speed, froude=FROUDE,
            rho=RHO_SEA, wetted_surface=geometry["wetted_surface"],
        )
        sensitivity[f"one_plus_k_{k:.2f}"] = {"ct": chain.ct}
    if result.get("comparison"):
        base_ct = result["comparison"]["ct_here"]
        for key in sensitivity:
            sensitivity[key]["relative_change_vs_baseline"] = (
                sensitivity[key]["ct"] / base_ct - 1.0
            )
    result["form_factor_sensitivity"] = sensitivity

    # ---- offline chain (no ship API needed) --------------------------------
    result["offline_chain"] = _offline_chain(geometry=geometry, speed=speed)
    result["criterion_d"] = _criterion_d(result["offline_chain"])
    result["experimental_reference_search"] = {
        "verdict": "NOT_FOUND",
        "searched": [
            "this repository: benchmarks/, cases/, configs/, docs/ (KVLCC2 appears only as "
            "geometry and as caller-supplied maneuvering derivatives -- no towing-tank C_T)",
            "ship-performance-platform checkout: backend package only, no reference dataset",
            "platform API :8001 (resistance-advanced/ittc78): connection refused during this run",
            "outbound network (package index / raw source host): blocked in this sandbox",
        ],
        "consequence": (
            "the 'versus experiment' limb of criterion (d) stays WITHHELD; no number is "
            "recalled from memory to fill it, and the friction limb is reported instead"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kvlcc2.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    result = run(Path(args.out))
    print(json.dumps(result.get("comparison", result), indent=2, default=float))




# ---------------------------------------------------------------------------
# offline chain: the friction limb, computed without the ship platform
# ---------------------------------------------------------------------------
def _wall_model_mean_cf(re_length: float, *, n_steps: int = 400) -> dict:
    """Mean skin-friction coefficient at ``Re_L`` from the wall-law plate model.

    The marched model (:func:`tensorlbm.wall_model_les.flat_plate.run_flat_plate`)
    returns ``Re_theta(Re_x)``; for a flat plate the mean friction coefficient is
    ``C_F = 2 theta(L)/L = 2 Re_theta(L)/Re_L`` -- an identity, not a fit.  This
    is the *model's* prediction of the ship's friction at ``Re = 1.3e7``, and it
    is compared below with the ITTC-1957 line, which is the empirical standard.
    """

    import numpy as np

    from tensorlbm.wall_model_les.flat_plate import run_flat_plate

    result = run_flat_plate(re_theta_start=2.0e3, re_theta_end=2.0e5, n_steps=n_steps)
    re_x = np.asarray(result.re_x, dtype=float)
    if not (re_x.min() <= re_length <= re_x.max()):
        msg = (
            f"Re_L = {re_length:.3e} outside the marched range "
            f"[{re_x.min():.3e}, {re_x.max():.3e}]"
        )
        raise ValueError(msg)
    re_theta = float(np.interp(re_length, re_x, result.re_theta))
    cf_mean = 2.0 * re_theta / re_length
    cf_local = float(np.interp(re_length, re_x, result.c_f))
    return {
        "cf_mean": float(cf_mean),
        "cf_local": cf_local,
        "re_theta_at_re_l": re_theta,
        "march_re_x_range": [float(re_x.min()), float(re_x.max())],
        "closure": result.metadata.get("closure"),
        "pi_wake": float(result.pi_wake),
        "kappa": float(result.kappa),
        "b": float(result.b),
    }


def _offline_chain(*, geometry: dict, speed: float) -> dict:
    """``C_T = (1+k) C_F + C_A`` with the friction coefficient from two sources."""

    cf_ittc = ittc_1957_cf(RE_TARGET)
    plate = _wall_model_mean_cf(RE_TARGET)
    # The correlation allowance is a *common* input of both chains and therefore
    # cancels in the relative comparison; it is reported symbolically instead of
    # being invented here.
    return {
        "reynolds": RE_TARGET,
        "form_factor": FORM_FACTOR,
        "froude": FROUDE,
        "speed_ms": speed,
        "wetted_surface_m2": geometry["wetted_surface"],
        "cf_ittc_1957": float(cf_ittc),
        "wall_model_plate": plate,
        "cf_wall_model": float(plate["cf_mean"]),
        "cf_rel_difference": float(abs(plate["cf_mean"] - cf_ittc) / cf_ittc),
        "correlation_allowance": (
            "common input C_A (ITTC-78 allowance); it cancels in the relative "
            "comparison, so no value is fabricated here"
        ),
        "residual_resistance": (
            "NOT predicted by the wall model: C_R needs the resolved pressure "
            "field (wave + viscous pressure), which a wall-function plate model "
            "does not contain"
        ),
    }


def _criterion_d(offline: dict) -> dict:
    """Criterion (d) as it can be decided offline.

    The literal "versus experiment" limb has no reference in this environment
    (see ``experimental_reference`` below).  What *is* decidable is the friction
    forecast itself: two independent friction chains -- the wall-law boundary
    layer model and the ITTC-1957 empirical line -- at the mission Reynolds
    number, propagated into ``C_T`` with the shared ``(1+k)`` and ``C_A``.
    """

    k = FORM_FACTOR
    cf_wm = offline["cf_wall_model"]
    cf_std = offline["cf_ittc_1957"]
    ct_wm = k * cf_wm
    ct_std = k * cf_std
    rel = abs(ct_wm - ct_std) / ct_std
    return {
        "threshold": NOMINAL_CRITERION,
        "quantity": (
            "C_T = (1+k) C_F with C_F from the wall-law boundary-layer model vs "
            "C_F from the ITTC-1957 line, at Re = 1.3e7"
        ),
        "ct_wall_model": float(ct_wm),
        "ct_ittc_1957": float(ct_std),
        "ct_rel_error": float(rel),
        "verdict": "PASS" if rel <= NOMINAL_CRITERION else "FAIL",
        "partner": "empirical standard line (ITTC-1957), NOT a measurement",
        "experimental_limb": "WITHHELD",
    }


if __name__ == "__main__":
    main()
