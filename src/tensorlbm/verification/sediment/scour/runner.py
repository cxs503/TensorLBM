"""End-to-end runner for the sediment / scour verification face.

Runs the three cases, evaluates the six machine-checkable acceptance criteria of
campaign node ``P84W2-N04``, writes ``artifacts/sediment_scour/{case,results}.json``
and the figures, and prints a summary.  Everything the report quotes is written
by this module -- no number is copied by hand.

Usage::

    python -m tensorlbm.verification.sediment.scour.runner --out artifacts/sediment_scour
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .cases import pile_scour_case, rouse_case, shields_case
from .sediment import G, KAPPA

# Sumer & Fredsoe (2002), *The Mechanics of Scour in the Marine Environment*,
# ch. 3: equilibrium scour depth of a slender pile in a steady current,
# z_s/D = 1.0-1.5 for h/D >= 2 in live-bed conditions.  The node's C3 datum.
ZS_D_BAND = (1.0, 1.5)


def gci_study(values: list[float], ratios: list[float], p_assumed: float = 2.0) -> dict:
    """Roache grid-convergence index from three (coarse, medium, fine) values.

    ``GCI = 1.25 |eps| / (r^p - 1)`` with ``eps`` the relative difference between
    the two finest grids and ``p`` the observed order (Richardson), falling back
    to ``p_assumed`` when the observed order is not in ``[0.5, 4]``.
    """
    f3, f2, f1 = float(values[0]), float(values[1]), float(values[2])  # coarse, medium, fine
    r21, r32 = float(ratios[1]), float(ratios[0])
    eps = (f2 - f1) / f1 if abs(f1) > 0 else float("nan")
    denom = f2 - f3
    if abs(denom) > 1e-30 and abs(f2 - f1) > 1e-30:
        ratio = (f2 - f1) / denom
        if 1e-3 < ratio < 100.0 and r21 > 1.0:
            p = abs(math.log(ratio) / math.log(r21))
        else:
            p = p_assumed
    else:
        p = p_assumed
    if not (0.5 <= p <= 4.0):
        p = p_assumed
    gci_fine = 1.25 * abs(eps) / (r21**p - 1.0) if r21 > 1.0 else float("nan")
    return {"values": [f3, f2, f1], "refinement_ratios": [r32, r21], "observed_order": p,
            "eps_fine": eps, "gci_fine": gci_fine}


def run_all(
    *,
    fast: bool = False,
    skip_scour_grids: bool = False,
    out_dir: str | Path | None = None,
) -> dict:
    """Run every case and return the full result dictionary."""
    t0 = time.time()
    timings: dict[str, float] = {}

    t = time.time()
    shields = shields_case(steps=20000 if fast else 30000)
    timings["shields_s"] = time.time() - t

    t = time.time()
    rouse = rouse_case(
        steps=20000 if fast else 30000,
        n_particles=60_000 if fast else 200_000,
        seed=20261109,
    )
    timings["rouse_s"] = time.time() - t

    # --- pile scour: grid study + depth-feedback sensitivity ---------------
    levels = [8, 10, 12] if not fast else [8, 10]
    scour_runs = []
    t = time.time()
    for k, cpd in enumerate(levels):
        res = pile_scour_case(
            cells_per_D=cpd,
            n_relax=2500 if fast else 3000,
            n_cycle=8,
            n_per_cycle=200 if fast else 250,
            depth_feedback=True,
            max_scour_D=6.0,
        )
        res["level"] = k
        scour_runs.append(res)
    timings["scour_grids_s"] = time.time() - t

    t = time.time()
    scour_nofb = pile_scour_case(
        cells_per_D=levels[1],
        n_relax=2500 if fast else 3000,
        n_cycle=8,
        n_per_cycle=200 if fast else 250,
        depth_feedback=False,
        max_scour_D=6.0,
    )
    timings["scour_nofeedback_s"] = time.time() - t

    primary = scour_runs[-1]
    zs_values = [r["z_s_over_D"] for r in scour_runs]
    ratios = [
        levels[1] / levels[0],
        levels[2] / levels[1] if len(levels) > 2 else levels[1] / levels[0],
    ]
    gci = gci_study(zs_values, ratios) if len(levels) >= 3 else {"gci_fine": float("nan")}

    # --- acceptance criteria ------------------------------------------------
    zs = primary["z_s_over_D"]
    lo, hi = ZS_D_BAND
    datum = 0.5 * (lo + hi)
    # "within 25 % of the Sumer & Fredsoe band": inside the band, or within 25 %
    # of the nearer band edge.
    if lo <= zs <= hi:
        c3_dev = 0.0
    else:
        c3_dev = min(abs(zs - lo), abs(zs - hi)) / (lo if zs < lo else hi)

    criteria = {
        "C1_shields": {
            "value": shields["theta_cr_model"],
            "reference": shields["theta_cr_classical"],
            "deviation": shields["theta_cr_deviation"],
            "threshold": 0.15,
            "pass": bool(shields["theta_cr_deviation"] < 0.15),
            "note": "theta_cr(d=0.2 mm) via the Soulsby-Whitehouse parameterisation of "
                    "the Shields curve, with u_* from the LBM momentum balance",
        },
        "C2_rouse": {
            "value": rouse["deviation_lpt"],
            "value_fv": rouse["deviation_fv"],
            "threshold": 0.20,
            "pass": bool(rouse["deviation_lpt"] < 0.20),
            "note": "max relative deviation of the Lagrangian Rouse profile from the "
                    "closed form over the comparison window; the finite-volume solve "
                    "gives the discretisation error separately",
        },
        "C3_scour_depth": {
            "value": zs,
            "band": list(ZS_D_BAND),
            "deviation": c3_dev,
            "threshold": 0.25,
            "pass": bool(c3_dev < 0.25),
            "converged": primary["converged"],
            "cap_hit": primary["max_scour_cap_hit"],
            "note": "2-D depth-averaged LBM+Exner; no finite equilibrium is reached "
                    "(see honest-FAIL note) so the value is a lower bound set by the "
                    "run-time cap",
        },
        "C4_mass_balance": {
            "value": max(r["mass_balance_rel_error"] for r in scour_runs),
            "threshold": 1e-4,
            "pass": bool(max(r["mass_balance_rel_error"] for r in scour_runs) < 1e-4),
            "note": "discrete Exner update vs the net face-flux integral "
                    "(telescoping check, all morphodynamic cycles)",
        },
        "C5_grid_convergence": {
            "value": gci.get("gci_fine"),
            "threshold": 0.10,
            "pass": bool(gci.get("gci_fine") is not None and gci["gci_fine"] < 0.10),
            "detail": gci,
            "note": "GCI on z_s/D over three grids; meaningful only because the "
                    "8-cycle state is before the runaway, see the report",
        },
        "C6_json_output": {
            "value": ["shields_curve", "rouse_profile", "scour_depth", "mass_balance"],
            "pass": True,
            "note": "artifacts/sediment_scour/results.json carries all four blocks",
        },
    }

    results = {
        "node": "P84W2-N04",
        "title": "TensorLBM suspended-sediment transport and bed-scour verification face",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": time.time() - t0,
        "timings": timings,
        "criteria": criteria,
        "all_pass": all(v.get("pass", False) for v in criteria.values()),
        "shields_curve": shields,
        "rouse_profile": rouse,
        "scour_depth": {
            "primary": primary,
            "by_grid": [
                {"cells_per_D": cpd, "grid": r["parameters"]["grid"],
                 "z_s_over_D": r["z_s_over_D"], "converged": r["converged"],
                 "cap_hit": r["max_scour_cap_hit"], "history": r["history"],
                 "mass_balance_rel_error": r["mass_balance_rel_error"]}
                for cpd, r in zip(levels, scour_runs)
            ],
            "no_depth_feedback": {
                "z_s_over_D": scour_nofb["z_s_over_D"],
                "cap_hit": scour_nofb["max_scour_cap_hit"],
                "history": scour_nofb["history"],
            },
            "sumer_fredsoe_band": list(ZS_D_BAND),
        },
        "mass_balance": {
            "rel_error": criteria["C4_mass_balance"]["value"],
            "note": "sum(dz_b) matches the net flux integral to round-off because the "
                    "face fluxes are shared between the update and the accounting",
        },
    }
    if out_dir is not None:
        write_artifacts(results, Path(out_dir))
    return results


def write_artifacts(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    (out / "case.json").write_text(
        json.dumps(_case_only(results), indent=1), encoding="utf-8"
    )
    (out / "results.json").write_text(
        json.dumps(_strip_heavy(results), indent=1), encoding="utf-8"
    )
    try:
        make_figures(results, fig_dir)
    except Exception as exc:  # pragma: no cover - plotting must never gate the run
        print(f"[warn] figure generation failed: {exc}")


def _case_only(results: dict) -> dict:
    """``case.json`` -- the inputs, so the run is reproducible."""
    return {
        "node": results["node"],
        "shields_parameters": results["shields_curve"]["parameters"],
        "rouse_parameters": results["rouse_profile"]["parameters"],
        "scour_parameters": results["scour_depth"]["primary"]["parameters"],
        "scour_grid_levels": [g["cells_per_D"] for g in results["scour_depth"]["by_grid"]],
        "sediment": {"d_m": 2.0e-4, "rho_s": 2650.0, "rho_f": 1000.0, "nu": 1.0e-6,
                     "porosity": 0.4, "repose_deg": 32.0},
        "references": {
            "shields": "Shields (1936); Soulsby & Whitehouse (1991) fit; Soulsby (1997) eq. 4.14",
            "rouse": "Rouse (1937)",
            "scour": "Sumer & Fredsoe (2002), The Mechanics of Scour in the Marine Environment",
            "bedload": "Meyer-Peter & Muller (1948)",
            "suspended": "van Rijn (1984) reference concentration",
            "settling": "Schiller & Naumann (1933) drag correction",
        },
    }


def _strip_heavy(results: dict) -> dict:
    """Drop the 2-D fields (kept only in the figure step) from results.json."""
    keep_scour = {}
    for key, val in results["scour_depth"].items():
        if key in ("primary", "no_depth_feedback"):
            v = {k: x for k, x in val.items() if k not in ("bed", "u_mean_field", "tau_pa", "pile_mask")}
            keep_scour[key] = v
        elif key == "by_grid":
            keep_scour[key] = [
                {k: x for k, x in g.items() if k not in ("bed",)}
                for g in val
            ]
        else:
            keep_scour[key] = val
    out = dict(results)
    out["scour_depth"] = keep_scour
    return out


def make_figures(results: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    # ---- figure 1: Shields curve + Rouse profile --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = results["shields_curve"]
    ax = axes[0]
    ax.loglog(sc["shields_curve"]["d_star"], sc["shields_curve"]["theta_cr"], "-",
              label="Soulsby-Whitehouse fit (this module)")
    ax.loglog(sc["shields_curve"]["digitised_d_star"], sc["shields_curve"]["digitised_theta_cr"],
              "o", mfc="none", label="Shields (1936) digitised readings")
    d02 = sc["parameters"]["d_star"]
    ax.plot([d02], [sc["theta_cr_model"]], "r*", ms=14,
            label=f"d=0.2 mm: theta_cr={sc['theta_cr_model']:.4f}")
    ax.axhline(sc["theta_cr_classical"], color="k", ls=":", lw=1,
               label=f"classical 0.045")
    ax.set_xlabel("D* [-]"); ax.set_ylabel(r"$\theta_{cr}$ [-]")
    ax.set_title("(a) Shields threshold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    rp = results["rouse_profile"]["profile"]
    ax = axes[1]
    zz = np.array(rp["z_over_h"])
    ax.semilogy(rp["C_analytic"], zz, "-", label="Rouse (1937) closed form")
    ax.semilogy(rp["C_fv"], zz, "--", label=f"FV advection-diffusion (dev {results['rouse_profile']['deviation_fv']*100:.1f} %)")
    ax.semilogy(rp["C_lpt"], zz, "o", ms=3,
                label=f"Lagrangian grains (dev {results['rouse_profile']['deviation_lpt']*100:.1f} %)")
    ax.set_xlabel("C / C_a [-]"); ax.set_ylabel("z / h [-]")
    ax.set_title(f"(b) Rouse profile, Z={results['rouse_profile']['parameters']['rouse_number']:.2f}")
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p1 = fig_dir / "shields_rouse.png"
    fig.savefig(p1, dpi=140); plt.close(fig); made.append(str(p1))

    # ---- figure 2: scour history + bed map -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for g in results["scour_depth"]["by_grid"]:
        hist = np.array(g["history"]) / results["scour_depth"]["primary"]["parameters"]["D_m"]
        ax.plot(np.arange(1, hist.size + 1), hist, "o-", ms=3, label=f"{g['cells_per_D']} cells/D")
    lo, hi = ZS_D_BAND
    ax.axhspan(lo, hi, color="g", alpha=0.15, label="Sumer & Fredsoe (2002) band")
    ax.set_xlabel("morphodynamic cycle"); ax.set_ylabel(r"$z_s/D$ [-]")
    ax.set_title("(c) Scour depth history (2-D depth-averaged)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    bed = np.array(results["scour_depth"]["primary"]["bed"])
    D = results["scour_depth"]["primary"]["parameters"]["D_m"]
    im = ax.imshow((bed / D).T, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xlabel("x [cells]"); ax.set_ylabel("y [cells]")
    ax.set_title("(d) Bed elevation change $z_b/D$ (negative = scour)")
    fig.colorbar(im, ax=ax, label="$z_b/D$")
    fig.tight_layout()
    p2 = fig_dir / "scour.png"
    fig.savefig(p2, dpi=140); plt.close(fig); made.append(str(p2))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/sediment_scour")
    ap.add_argument("--fast", action="store_true", help="reduced resolution/steps")
    args = ap.parse_args(argv)
    res = run_all(fast=args.fast, out_dir=args.out)
    print(json.dumps({k: v for k, v in res["criteria"].items()}, indent=1, default=str))
    print(f"\nall_pass = {res['all_pass']}   runtime = {res['runtime_s']:.1f} s")
    print(f"artifacts -> {Path(args.out).resolve()}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())