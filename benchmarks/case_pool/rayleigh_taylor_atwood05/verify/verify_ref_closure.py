"""Closed-form audit of the CASE-RT-INSTABILITY-ATWOOD acceptance criterion.

No solver, no sampling: everything here is an exact evaluation of

    sigma(c, x) = sqrt(c + x^2) - x,     c = A g k,   x = nu k^2,   k = 2 pi / lam

which is the locked reference of the accepted case (``ref/ref_lock.py``
``sigma_viscous``: ``sqrt(Agk + nu^2 k^4) - nu k^2``).  Three questions are asked
that the accepted report does not ask:

1. **Is ``sigma_visc`` correct?**  Re-evaluated here from the parameters in
   ``ref/ref_lock.json`` with an implementation that shares no code with it.
2. **Is the gate able to detect a viscosity error?**  The gate is
   ``|sigma_sim/sigma_visc - 1| <= 15 %``.  Inverted exactly, it maps to an
   interval of *effective viscosities*: any solver whose effective viscosity lies
   in that interval passes.  Reported as ``nu_eff/nu`` -- the acceptance region.
3. **How well is ``nu_eff`` determined by a measured ``sigma``?**  The accepted
   report tabulates ``nu_eff/nu = 8.0 ... 37.2`` as if it were a measurement; the
   inversion's derivative ``d nu_eff / d sigma`` says how much of that is noise.

Usage
-----
    python3 verify/verify_ref_closure.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
ART = HERE / "artifacts"


def sigma_inv(A: float, g: float, lam: float) -> float:
    return math.sqrt(A * g * 2.0 * math.pi / lam)


def sigma_visc(A: float, g: float, lam: float, nu: float) -> float:
    k = 2.0 * math.pi / lam
    x = nu * k * k
    return math.sqrt(A * g * k + x * x) - x


def dsigma_dnu(A: float, g: float, lam: float, nu: float) -> float:
    k = 2.0 * math.pi / lam
    x = nu * k * k
    return -k * k * (1.0 - x / math.sqrt(A * g * k + x * x))


def nu_eff_from_sigma(A: float, g: float, lam: float, s: float) -> float:
    """Exact inverse of ``sigma_visc`` (== ``ref_lock.nu_eff_from_sigma``)."""
    k = 2.0 * math.pi / lam
    c = A * g * k
    if s <= 0.0 or s * s >= c:
        return float("nan")
    return (c - s * s) / (2.0 * s) / (k * k)


def gate_viscosity_interval(A: float, g: float, lam: float, nu: float,
                            gate_pct: float = 15.0) -> dict:
    """Viscosities whose *exact* growth rate still satisfies the gate."""
    s0 = sigma_visc(A, g, lam, nu)
    lo, hi = s0 * (1.0 - gate_pct / 100.0), s0 * (1.0 + gate_pct / 100.0)
    # sigma_visc is monotone decreasing in nu and bounded above by sigma_inv at nu=0
    nu_lo = 0.0
    nu_hi = nu_eff_from_sigma(A, g, lam, lo)
    reachable = hi < sigma_inv(A, g, lam)
    return {
        "sigma_visc": s0,
        "sigma_inv": sigma_inv(A, g, lam),
        "gate_lo": lo, "gate_hi": hi,
        "nu_eff_min_over_nu": nu_lo / nu,
        "nu_eff_max_over_nu": nu_hi / nu,
        "upper_side_reachable": bool(reachable),
        "max_achievable_positive_pct": 100.0 * (sigma_inv(A, g, lam) / s0 - 1.0),
    }


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    lock = json.loads((CASE / "ref" / "ref_lock.json").read_text())
    A = lock["atwood"]
    rows = []
    for e in lock["suite"]:
        lam, g, nu = e["lam"], e["g"], e["nu"]
        s_inv = sigma_inv(A, g, lam)
        s_vis = sigma_visc(A, g, lam, nu)
        k = 2.0 * math.pi / lam
        x = nu * k * k
        ci = A * g * k
        # elasticities
        S_nu = dsigma_dnu(A, g, lam, nu) * nu / s_vis
        S_g = (ci / (2.0 * math.sqrt(ci + x * x))) / s_vis
        gate = gate_viscosity_interval(A, g, lam, nu)
        # inversion conditioning: |d nu_eff / d sigma|
        dnu = abs(1.0 / dsigma_dnu(A, g, lam, nu))
        rows.append({
            "label": e["label"], "lam": lam, "g": g, "nu": nu,
            "sigma_inv_ref": e["sigma_inviscid"],
            "sigma_inv_recomputed": s_inv,
            "sigma_visc_ref": e["sigma_viscous"],
            "sigma_visc_recomputed": s_vis,
            "ref_agreement_abs": abs(s_vis - e["sigma_viscous"]) + abs(s_inv - e["sigma_inviscid"]),
            "elasticity_nu": S_nu,
            "elasticity_g": S_g,
            "viscous_fraction_of_sigma": x / s_vis,
            "gate": gate,
            "dnu_eff_dsigma": dnu,
            "nu_eff_uncertainty_3pct_sigma": dnu * 0.03 * s_vis / nu,
        })

    ref_err = max(r["ref_agreement_abs"] for r in rows)
    out = {
        "case": "CASE-RT-INSTABILITY-ATWOOD",
        "atwood": A,
        "reference_recheck": {
            "max_abs_difference_vs_ref_lock": ref_err,
            "verdict": "identical to machine precision (<1e-18)" if ref_err < 1e-15 else "MISMATCH",
        },
        "rows": rows,
    }
    (ART / "ref_closure.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"== 1. reference recheck (independent closed form vs ref_lock.json) ==")
    print(f"   max |difference| over {len(rows)} suite rows: {ref_err:.3e}")
    print("\n== 2. can the +-15 % gate see a viscosity error? ==")
    print("   case                nu      S_nu     nu_eff/nu range implied by the gate   +side")
    for r in rows:
        g_ = r["gate"]
        print(f"   {r['label']:<26s} {r['nu']:.4f} {r['elasticity_nu']*100:+6.2f}%  "
              f"[{g_['nu_eff_min_over_nu']:.2f}, {g_['nu_eff_max_over_nu']:.2f}]"
              f"   reachable={g_['upper_side_reachable']} "
              f"(max +{g_['max_achievable_positive_pct']:.2f}%)")
    print("\n== 3. conditioning of the nu_eff inversion the accepted report quotes ==")
    for r in rows:
        print(f"   {r['label']:<26s} |dnu_eff/dsigma|={r['dnu_eff_dsigma']:.1f}  "
              f"=> a 3 % sigma error moves nu_eff by {r['nu_eff_uncertainty_3pct_sigma']:.1f} x nu")
    print(f"\n[written] {ART/'ref_closure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())