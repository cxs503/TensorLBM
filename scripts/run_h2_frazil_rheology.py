#!/usr/bin/env python3
"""H2-FRAZIL-RHEOLOGY entry point: volume fraction -> effective viscosity.

Runs the four machine-checked criteria of
:mod:`tensorlbm.frazil_suspension_rheology` and writes the run artifacts:

* ``campaign.json``  -- every number produced by this run (single source for
  the report; nothing is transcribed by hand),
* ``benchmark.md``   -- ref-vs-got tables with errors and pass/fail,
* ``fig1_mu_r_closures.png``, ``fig2_shear_rate_sweep.png``,
  ``fig3_lbm_profiles.png``,
* ``repro.json``     -- provenance (commit, module hash, versions, command).

Reproduce from the repository root::

    PYTHONPATH=src python scripts/run_h2_frazil_rheology.py --out artifacts/h2_frazil_rheology

Exit code is 0 only when every criterion passes; a failing run still writes the
artifacts (honest FAIL is data, not a crash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib  # noqa: I001
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensorlbm import frazil_suspension_rheology as rheo  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_mu_r_closures(report: dict, out: Path) -> Path:
    phi_max = report["parameters"]["phi_max"]
    phis = np.linspace(0.0, phi_max * 0.98, 240)
    kd = np.array([rheo.krieger_dougherty_relative_viscosity(p, phi_max=phi_max) for p in phis])
    ein = 1.0 + rheo.EINSTEIN_INTRINSIC * phis
    bat = ein + rheo.BATCHELOR_K2 * phis**2
    th_mask = phis <= 0.50  # Thomas (1965) is valid up to phi ~ 0.5
    th = np.array([rheo.thomas_relative_viscosity(p) for p in phis[th_mask]])

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax[0].semilogy(phis, kd, "k-", lw=2.0, label=r"Krieger-Dougherty $\phi_{max}=%.2f$" % phi_max)
    ax[0].semilogy(phis[th_mask], th, "C3--", lw=1.6, label="Thomas (1965) empirical")
    ax[0].semilogy(phis, bat, "C1-.", lw=1.3, label=r"Batchelor-Green $1+2.5\phi+6.2\phi^2$")
    ax[0].semilogy(phis, ein, "C0:", lw=1.3, label=r"Einstein $1+2.5\phi$")
    dil = report["dilute_limit"]["rows"]
    mid = report["mid_concentration"]["rows"]
    ax[0].plot([r["phi"] for r in dil], [r["got_model"] for r in dil], "o", ms=5, mfc="none",
               color="C2", label="model chain (c1 points)")
    ax[0].plot([r["phi"] for r in mid], [r["got_model"] for r in mid], "s", ms=5, mfc="none",
               color="C4", label="model chain (c2 points)")
    ax[0].axvspan(0.0, rheo.DILUTE_LIMIT, color="C2", alpha=0.06)
    ax[0].axvspan(*rheo.MID_CONCENTRATION_RANGE, color="C4", alpha=0.06)
    ax[0].set_xlabel(r"ice volume fraction $\phi$")
    ax[0].set_ylabel(r"relative viscosity $\mu_r$")
    ax[0].set_xlim(0.0, phi_max)
    ax[0].set_ylim(1.0, 1.0e3)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title("(a) closure families and the model chain")
    ax[0].legend(fontsize=7.5, loc="upper left")

    ax[1].axhline(10.0, color="r", ls="--", lw=1.0, label="10 % criterion")
    ax[1].semilogy([r["phi"] for r in dil], [100 * r["rel_error"] for r in dil], "o-", color="C2",
                   label="c1: model vs Einstein")
    ax[1].semilogy([r["phi"] for r in mid], [100 * r["rel_error_kd"] for r in mid], "s-", color="C4",
                   label="c2: model vs KD")
    ax[1].semilogy([r["phi"] for r in mid],
                   [max(100 * r["rel_error_thomas"], 1e-4) for r in mid], "^--", color="C3",
                   label="c2 cross-check: model vs Thomas (1965)")
    ax[1].set_xlabel(r"ice volume fraction $\phi$")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("(b) pointwise error against the analytic / literature anchors")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.suptitle("H2-FRAZIL-RHEOLOGY  phi -> mu_r : %d/%d criteria pass"
                 % (report["acceptance"]["n_pass"], report["acceptance"]["n_defined"]), y=1.0)
    fig.tight_layout()
    path = out / "fig1_mu_r_closures.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_shear_rate_sweep(report: dict, out: Path) -> Path:
    sweep = report["shear_rate_sweep"]
    g = np.asarray(sweep["gamma_dots"])
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(sweep["curves"])))
    for colour, curve in zip(colours, sweep["curves"]):
        mu = np.asarray(curve["mu_app"])
        ax[0].loglog(g, mu, "o-", color=colour, ms=4,
                     label=r"$\phi=%.2f$, $n=%.3f$, slope=%.3f"
                     % (curve["phi"], curve["n"], curve["fitted_slope"]))
        if curve["phi"] > 0.0:
            guide = mu[0] * (g / g[0]) ** (curve["n"] - 1.0)
            ax[0].loglog(g, guide, ":", color=colour, lw=0.9)
    ax[0].set_xlabel(r"shear rate $\dot\gamma$ [1/s]")
    ax[0].set_ylabel(r"apparent viscosity $\mu_{app}$ [Pa s]")
    ax[0].set_title("(a) power-law rheology: model vs $K\\dot\\gamma^{n-1}$ (dotted)")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    phis = [c["phi"] for c in sweep["curves"]]
    ns = [c["n"] for c in sweep["curves"]]
    ax[1].plot(phis, ns, "o-", color="C0")
    ax[1].axhline(1.0, color="k", ls=":", label="Newtonian")
    ax[1].axhline(report["parameters"]["n_packed"], color="r", ls="--",
                  label=r"$n_{packed}=%.2f$" % report["parameters"]["n_packed"])
    ax[1].set_xlabel(r"ice volume fraction $\phi$")
    ax[1].set_ylabel(r"flow index $n(\phi)$")
    ax[1].set_title("(b) W9-1215 flow index: thinning sets in with concentration")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("H2-FRAZIL-RHEOLOGY shear-rate scan: max |slope - (n-1)| = %.2e"
                 % sweep["max_slope_error"], y=1.0)
    fig.tight_layout()
    path = out / "fig2_shear_rate_sweep.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_lbm_profiles(report: dict, out: Path) -> Path | None:
    rows = [r for r in report["channels"]["rows"] if r.get("ran") and "profile" in r]
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colours = plt.cm.plasma(np.linspace(0.05, 0.8, len(rows)))
    for colour, row in zip(colours, rows):
        y = np.asarray(row["y_phys"])
        ax[0].plot(np.asarray(row["profile"]), y, "-", color=colour, lw=1.6,
                   label=r"LBM $\phi=%.2f$" % row["phi"])
        ax[0].plot(np.asarray(row["u_exact"]), y, ":", color=colour, lw=1.2)
    ax[0].set_xlabel("streamwise velocity $u$ [lattice]")
    ax[0].set_ylabel("wall-normal coordinate $y$ [lattice]")
    ax[0].set_title("(a) LBM profile (solid) vs exact power-law solution (dotted)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    ax[1].semilogy([r["phi"] for r in rows], [100 * r["l2_relative"] for r in rows], "o-")
    ax[1].axhline(100 * report["channels"]["threshold"], color="r", ls="--",
                  label="W9-1215 node criterion (2 %)")
    ax[1].set_xlabel(r"ice volume fraction $\phi$")
    ax[1].set_ylabel("L2 velocity error [%]")
    ax[1].set_title("(b) slurry channel residual")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.suptitle("H2-FRAZIL-RHEOLOGY  LBM slurry channel: worst L2 = %.3f %%"
                 % (100 * report["channels"]["worst_l2_relative"]), y=1.0)
    fig.tight_layout()
    path = out / "fig3_lbm_profiles.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown benchmark
# ---------------------------------------------------------------------------


def write_benchmark_md(report: dict, out: Path, repro_cmd: str, figures: dict) -> Path:
    p = report["parameters"]
    acc = report["acceptance"]
    lines = []
    lines.append("# H2-FRAZIL-RHEOLOGY — ref vs got benchmark\n")
    lines.append("Generated by `%s` (do not edit by hand).\n" % Path(__file__).name)
    lines.append("## Criteria\n")
    lines.append("| id | metric | threshold | measured | verdict |")
    lines.append("|---|---|---|---|---|")
    for cid, c in acc["criteria"].items():
        measured = c["measured"]
        if isinstance(measured, dict):
            measured = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in measured.items())
        elif isinstance(measured, float):
            measured = f"{measured:.6g}"
        verdict = "PASS" if c["passed"] is True else ("**FAIL**" if c["passed"] is False else "undefined")
        lines.append("| %s | %s | %.3g | %s | %s |"
                     % (cid, c["metric"], c["threshold"], measured, verdict))
    lines.append("")
    lines.append("**Verdict: %s (%d/%d)**\n"
                 % (acc["verdict"], acc["n_pass"], acc["n_defined"]))

    lines.append("## c1 — dilute limit vs Einstein (1906), phi <= 0.05, 10 %%\n")
    lines.append("| phi | ref Einstein | got model | abs err | rel err | pass |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["dilute_limit"]["rows"]:
        lines.append("| %.3f | %.6f | %.6f | %.3e | %+.3f %% | %s |"
                     % (r["phi"], r["ref_einstein"], r["got_model"], r["abs_error"],
                        100 * r["rel_error"], "yes" if r["passed"] else "no"))
    lines.append("\nmax rel err = %.4f %% (threshold %.1f %%)\n"
                 % (100 * report["dilute_limit"]["max_rel_error"],
                    100 * report["dilute_limit"]["threshold"]))

    lines.append("## c2 — mid concentration vs Krieger-Dougherty, 0.10 <= phi <= 0.30, 10 %%\n")
    lines.append("| phi | got model | ref KD | rel err KD | ref Thomas (1965) | rel err Thomas | pass |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["mid_concentration"]["rows"]:
        lines.append("| %.3f | %.6f | %.6f | %.2e %% | %.6f | %+.3f %% | %s |"
                     % (r["phi"], r["got_model"], r["ref_krieger_dougherty"],
                        100 * r["rel_error_kd"], r["ref_thomas_1965"],
                        100 * r["rel_error_thomas"], "yes" if r["passed"] else "no"))
    lines.append("\nThe KD column is a **code-path** check (the chain reproduces the closure it "
                 "is built on, hence round-off zero); the independent content of c2 is the "
                 "Thomas (1965) cross-check (max %.3f %%).\n"
                 % (100 * report["mid_concentration"]["max_rel_error_thomas"]))

    lines.append("## c3 — W9-1215 physical root (quoted criteria)\n")
    lines.append("| check | ref | got | rel err | pass |")
    lines.append("|---|---|---|---|---|")
    for c in report["root_chain"]["checks"]:
        lines.append("| %s | %s | %.6g | %.3e | %s |"
                     % (c["name"], c["ref"], c["got"], c["rel_error"],
                        "yes" if c["passed"] else "no"))
    lines.append("")
    lines.append("### c3b — LBM slurry channel vs exact Ostwald-de Waele solution (root criterion 2 %)\n")
    lines.append("| phi | n(phi) | K [Pa s^n] | nu_wall [lattice] | rescale | L2 err | peak err | tau_max | steps | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in report["channels"]["rows"]:
        if not r.get("ran"):
            lines.append("| %.3f | - | - | - | - | - | - | - | - | not run |" % r["phi"])
            continue
        lines.append("| %.3f | %.4f | %.6g | %.4g | %.4g | %.4f %% | %.4f %% | %.3f | %d | %s |"
                     % (r["phi"], r["flow_index"], r["consistency_index"], r["viscosity_wall"],
                        r["lattice_rescale"], 100 * r["l2_relative"], 100 * r["peak_relative"],
                        r["tau_relaxation_max"], r["steps"], "yes" if r["passed"] else "no"))
    lines.append("\nworst L2 = %.4f %% (threshold %.1f %%)\n"
                 % (100 * report["channels"]["worst_l2_relative"],
                    100 * report["channels"]["threshold"]))
    routed = [r for r in report["channels"]["rows"] if r.get("ran")]
    lines.append("solver-route consistency (solver-internal `n`, `mu_r` == node API): %d/%d rows\n"
                 % (sum(1 for r in routed if r.get("route_consistent")), len(routed)))

    lines.append("## c4 — shear-rate scan 1e-3 <= gdot <= 1e3\n")
    lines.append("| phi | n(phi) | K | fitted slope | expected n-1 | slope err | thinning ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in report["shear_rate_sweep"]["curves"]:
        lines.append("| %.3f | %.4f | %.6g | %.6f | %.6f | %.2e | %.2f |"
                     % (c["phi"], c["n"], c["K"], c["fitted_slope"], c["expected_slope"],
                        c["slope_error"], c["shear_thinning_ratio"]))
    lines.append("\nNewtonian case slope = %.2e (must be 0).\n"
                 % report["shear_rate_sweep"]["newtonian_slope"])

    lines.append("## External anchors\n")
    for k, v in report["anchors"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Parameters\n")
    lines.append("```json")
    lines.append(json.dumps(p, indent=2, sort_keys=True))
    lines.append("```\n")
    lines.append("## Reproduce\n")
    lines.append("```bash")
    lines.append(repro_cmd)
    lines.append("```\n")
    if figures:
        lines.append("## Figures\n")
        for name, caption in figures.items():
            lines.append(f"- `{name}` — {caption}")
    path = out / "benchmark.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/h2_frazil_rheology", help="artifact directory")
    ap.add_argument("--H", type=int, default=16, help="channel half-height in cells")
    ap.add_argument("--nx", type=int, default=2, help="streamwise cells (flow is x-homogeneous)")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1.0e-4)
    ap.add_argument("--check-every", type=int, default=200)
    ap.add_argument("--phi-channel", type=float, nargs="*",
                    default=[0.05, 0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--skip-channels", action="store_true",
                    help="skip the LBM channel runs (criterion c3b then counts as undefined)")
    ap.add_argument("--mu-water", type=float, default=1.0e-3, help="water viscosity [Pa s]")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    report = rheo.build_report(
        phis_channel=args.phi_channel,
        mu_water=args.mu_water,
        run_channels=not args.skip_channels,
        return_profiles=True,
        H=args.H,
        nx=args.nx,
        max_steps=args.max_steps,
        tol=args.tol,
        check_every=args.check_every,
    )
    wall = time.time() - t0

    module_path = Path(rheo.__file__)
    repro_cmd = ("PYTHONPATH=src python scripts/run_h2_frazil_rheology.py "
                 "--out %s --H %d --nx %d --max-steps %d --tol %g"
                 % (args.out, args.H, args.nx, args.max_steps, args.tol))
    report["repro_cmd"] = repro_cmd
    report["provenance"] = {
        "commit": _git_commit(),
        "module": str(module_path.relative_to(_REPO_ROOT)) if module_path.is_relative_to(_REPO_ROOT) else str(module_path),
        "module_sha256": _sha256(module_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "argv": sys.argv[1:],
        "wall_time_s": round(wall, 2),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    fig1 = fig_mu_r_closures(report, out)
    fig2 = fig_shear_rate_sweep(report, out)
    fig3 = fig_lbm_profiles(report, out)
    figures = {
        fig1.name: "relative viscosity closures with the model-chain points and the "
                   "pointwise error against Einstein (c1) and KD / Thomas (c2)",
        fig2.name: "apparent viscosity vs shear rate for phi = 0 ... 0.50 with the "
                   "power-law guide K*gdot^(n-1) and the W9-1215 flow index n(phi)",
    }
    if fig3 is not None:
        figures[fig3.name] = "LBM slurry-channel velocity profiles vs the exact power-law " \
                             "solution and the 2 % node criterion"

    (out / "campaign.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "repro.json").write_text(json.dumps(report["provenance"], indent=2, sort_keys=True) + "\n")
    bm = write_benchmark_md(report, out, repro_cmd, figures)

    acc = report["acceptance"]
    print(json.dumps({"verdict": acc["verdict"], "n_pass": acc["n_pass"],
                      "n_defined": acc["n_defined"], "out": str(out),
                      "benchmark": str(bm), "wall_time_s": report["provenance"]["wall_time_s"]},
                     indent=2))
    for cid, c in acc["criteria"].items():
        flag = "PASS " if c["passed"] is True else ("FAIL " if c["passed"] is False else "undef")
        print(f"  {cid:28s} {flag}  {c['metric']}")
    return 0 if acc["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
