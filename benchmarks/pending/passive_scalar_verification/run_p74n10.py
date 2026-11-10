#!/usr/bin/env python
"""P74-N10 passive-scalar transport surface — driver.

Runs the resumable case set of
:mod:`tensorlbm.passive_scalar_transport_verification`, evaluates the eight
gates (A–H) and writes the machine-readable artifact plus figures.

Usage
-----
    python .../run_p74n10.py --list                 # enumerate the jobs
    python .../run_p74n10.py --only erf_L0,erf_L1   # one chunk (resumable)
    python .../run_p74n10.py --assemble             # gates + artifact + figures
    python .../run_p74n10.py --all                  # everything (minutes)

Every case result is cached under ``artifacts/passive_scalar_p74n10/cases/``
so a heavy ``Pe_r = 40`` cell can be run in its own chunk / on a compute node
and the verdict assembled afterwards.  All numbers in the artifact trace back
to one of those cache files (zero hand transcription).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

# These grids are tiny (N_y x 128): multithreaded torch only adds contention on
# a shared control-plane box, so the driver is single-threaded by default
# (override with P74N10_THREADS=0 to keep torch's default).
_threads = os.environ.get("P74N10_THREADS", "1")
if _threads.isdigit() and int(_threads) > 0:
    import torch

    torch.set_num_threads(int(_threads))

from tensorlbm.passive_scalar_transport_verification import (  # noqa: E402
    TransportSpec,
    evaluate_gates,
    run_surface,
    surface_cases,
    write_artifact,
)

CASE_DIR = REPO_ROOT / "artifacts/passive_scalar_p74n10/cases"
ARTIFACT = REPO_ROOT / "artifacts/passive_scalar_p74n10.json"
FIG_DIR = REPO_ROOT / "artifacts/passive_scalar_p74n10/figures"


def _load_cases() -> list[dict]:
    rows = []
    for path in sorted(CASE_DIR.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def _print_gate_summary(report: dict) -> None:
    print("=" * 84)
    print(f"P74-N10 passive-scalar transport surface   verdict: {report['verdict']}")
    print("=" * 84)
    for key, gate in report["gates"].items():
        flag = "PASS" if gate["pass"] else "FAIL"
        print(f"[{key}] {gate['name']}: {flag}")
        if key == "A":
            print(
                f"      finest rel L2 = {gate['finest_grid_rel_pct']:.4f} % "
                f"(nx={gate['finest_nx']})  orders={['%.3f' % o for o in gate['orders']]}"
            )
        elif key == "B":
            for row in gate["per_pe"]:
                print(
                    f"      Pe={row['pe']:5g} err%={['%+.2f' % e for e in row['err_pct']]}"
                    f" N_y={row['ny']} nx={row['nx']} rr={['%.3f' % r for r in row['meter_rate_ratio']]}"
                    f"  best={row['best_err_pct']:.2f} % @N_y={row['best_ny']}"
                )
        elif key == "C":
            for row in gate["table"]:
                print(
                    f"      Pe={row['pe']:5g} D={row['diffusivity']:.4f} u={row['u_mean']:.5f}"
                    f"  D_eff/D={row['ratio_sim']:.6f} dev={row['dev_from_unity_pct']:+.3f} %"
                )
        elif key == "D":
            for row in gate["table"]:
                print(
                    f"      Sc={row['sc']:5g} D={row['diffusivity']:.4f} tau_d={row['tau_d']:.3f}"
                    f"  D_eff/D={row['ratio_sim']:.6f}  err={row['rel_err_vs_closed_pct']:+.2f} %"
                )
            print(
                f"      spread={gate['spread_pct']:.3f} % "
                f"(bias-corrected {gate['spread_vs_discrete_pct']:.3f} %)"
            )
        elif key == "E":
            print(
                f"      max drift per 1e4 steps = {gate['max_rate']:.3e} "
                f"({gate['max_rate_case']}), raw max {gate['raw_max_drift']:.3e}"
            )
        elif key == "F":
            print(f"      u=0: D_eff/D = {gate['ratio_sim']:.6f} band={gate['band']}")
        if key == "A" or key == "B":
            pass
        elif key == "G":
            print(
                f"      erf monotone={gate['erf']['monotone_error_decay']} "
                f"R2={gate['erf']['loglog_r2']:.4f}; "
                f"dispersion R2_min={gate['dispersion']['r2_min']:.5f} "
                f"monotone={gate['dispersion']['monotone_all']}"
            )
        elif key == "H":
            for cell in gate["cells"]:
                print(
                    f"      Pe={cell['pe']:5g} N_y={cell['ny']:3d} nx={cell['nx']:4d} "
                    f"err={cell['err_pct']:+7.2f} %  rate_ratio={cell['meter_rate_ratio']:.3f}"
                    f"  {cell['status']}"
                )
    diagnostics = report.get("diagnostics", {})
    if diagnostics.get("rows"):
        print("diagnostics: fixed (Pe,N_y), varying D (tau_d)")
        for row in diagnostics["rows"]:
            print(
                f"      u={row['u_mean']:.3f} D={row['diffusivity']:.3f} tau_d={row['tau_d']:.3f}"
                f"  err={row['err_pct']:+.3f} %  increment_err={row['increment_err_pct']:+.2f} %"
            )
    naive = diagnostics.get("naive_sc_reading")
    if naive and naive.get("rows"):
        print(f"diagnostics: naive Sc reading (nu pinned), spread={naive['spread_pct']:.2f} %")
        for row in naive["rows"]:
            print(
                f"      Sc={row['sc']:5g} D={row['diffusivity']:.3f} tau_d={row['tau_d']:.3f}"
                f"  D_eff/D={row['ratio_sim']:.5f}  err={row['err_pct']:+.2f} %"
            )
    diffusion = diagnostics.get("diffusion_magnitude")
    if diffusion and diffusion.get("rows"):
        print("diagnostics: pure diffusion (u=0) vs D — the tau_d accuracy domain")
        for row in diffusion["rows"]:
            print(
                f"      D={row['diffusivity']:.3f} tau_d={row['tau_d']:.3f}"
                f"  D_eff/D={row['ratio_sim']:.5f}  err={row['err_pct']:+.2f} %"
            )
    print("=" * 84)


def _make_figures(report: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cases = _load_cases()

    # ---- fig 1: erf front + convergence -----------------------------------
    erf_rows = sorted([c for c in cases if c["kind"] == "erf"], key=lambda r: r["nx"])
    if erf_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        ref = erf_rows[-1]
        x = np.asarray(ref["x"])
        exact = np.asarray(ref["c_exact"])
        win = np.asarray(ref["window"], dtype=bool)
        for row in erf_rows:
            xn = np.asarray(row["x"])
            cn = np.asarray(row["c_numeric"])
            m = np.asarray(row["window"], dtype=bool)
            ax.plot(xn[m], cn[m], ".", ms=3, label=f"LBM nx={row['nx']}")
        ax.plot(x[win], exact[win], "k-", lw=1.2, label=r"$0.5\,\mathrm{erfc}(x/\sqrt{4Dt})$")
        ax.set_xlabel("x")
        ax.set_ylabel("C")
        ax.set_title("erf diffusion front (D=0.1, t=200)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        dx = np.asarray([r["dx"] for r in erf_rows])
        err = np.asarray([r["l2_abs"] for r in erf_rows])
        ax.loglog(dx, err, "o-", color="C3", label="LBM L2 error")
        ax.loglog(dx, err[-1] * (dx / dx[-1]) ** 2, "k--", lw=1.0, label="slope 2")
        for k, order in enumerate(report["gates"]["A"]["orders"]):
            ax.annotate(f"p={order:.2f}", (dx[k] * 1.1, err[k] * 1.05), fontsize=8)
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$L_2$ error")
        ax.set_title(
            f"min order {report['gates']['A']['min_order']:.2f} (gate "
            f"{report['gates']['A']['threshold_order']:g})"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = FIG_DIR / "fig1_erf_diffusion.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # ---- fig 2: Pe ladder + accuracy domain --------------------------------
    disp = [c for c in cases if c["kind"] == "dispersion"]
    if disp:
        gate_b = report["gates"]["B"]
        gate_h = report["gates"]["H"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        pe_grid = np.logspace(0.3, 1.8, 200)
        ax.loglog(
            pe_grid, [1.0 + pe**2 / 210.0 for pe in pe_grid], "k-", label=r"$1+Pe_r^2/210$"
        )
        colours = ["C0", "C1", "C2", "C3", "C4"]
        for k, row in enumerate(gate_b["per_pe"]):
            cells = [c for c in gate_h["cells"] if c["pe"] == row["pe"]]
            cells = sorted(cells, key=lambda c: c["ny"])
            ax.loglog(
                [row["pe"]],
                [1.0 + row["pe"] ** 2 / 210.0],
                "k*",
                ms=9,
            )
            ax.loglog(
                [row["pe"]] * len(cells),
                [c["ratio_sim"] for c in cells],
                "o",
                color=colours[k % 5],
                label=f"LBM $N_y$={cells[0]['ny']}..{cells[-1]['ny']}",
            )
        ax.set_xlabel(r"$Pe_r = U H / D$")
        ax.set_ylabel(r"$D_\mathrm{eff}/D$")
        ax.set_title("Taylor–Aris dispersion, time-marched LBM (stars = analytic)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

        ax = axes[1]
        pes = gate_h["pe_nodes"]
        nys = gate_h["ny_ladder"]
        err = np.zeros((len(pes), len(nys)))
        for i, pe in enumerate(pes):
            for j, ny in enumerate(nys):
                sel = [c for c in gate_h["cells"] if c["pe"] == pe and c["ny"] == ny]
                err[i, j] = sel[0]["err_pct"] if sel else np.nan
        limit = max(5.0, float(np.nanmax(np.abs(err))))
        im = ax.imshow(
            np.abs(err), origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=limit
        )
        ax.set_xticks(range(len(nys)))
        ax.set_xticklabels([str(n) for n in nys])
        ax.set_yticks(range(len(pes)))
        ax.set_yticklabels([f"{p:g}" for p in pes])
        for i in range(len(pes)):
            for j in range(len(nys)):
                val = err[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}%",
                    ha="center",
                    va="center",
                    color="w" if abs(val) < 0.6 * limit else "k",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, label=r"|error| vs $1+Pe^2/210$  [%]")
        ax.set_xlabel(r"$N_y$ (transverse nodes)")
        ax.set_ylabel(r"$Pe_r$")
        ax.set_title("accuracy domain (gate 5 %)")
        fig.tight_layout()
        path = FIG_DIR / "fig2_taylor_aris_accuracy_domain.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # ---- fig 3: Schmidt + mass budget --------------------------------------
    sc_rows = [c for c in cases if c["kind"] == "schmidt"]
    ctrl_rows = [c for c in cases if c["kind"] == "negative_control"]
    if sc_rows:
        gate_d = report["gates"]["D"]
        gate_e = report["gates"]["E"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        scs = [r["sc"] for r in gate_d["table"]]
        ratios = [r["ratio_sim"] for r in gate_d["table"]]
        closed = 1.0 + gate_d["pe"] ** 2 / 210.0
        ax.semilogx(scs, ratios, "o-", label=r"LBM $D_\mathrm{eff}/D$")
        ax.axhline(closed, color="k", ls="--", label="closed form")
        if ctrl_rows:
            ax.axhline(
                ctrl_rows[0]["ratio_sim"], color="C2", ls=":", label="u=0 control"
            )
            ax.axhline(1.0, color="C2", ls="-.", lw=0.8)
        ax.set_xlabel(r"$Sc = \nu/D$")
        ax.set_ylabel(r"$D_\mathrm{eff}/D$")
        ax.set_title(
            f"Schmidt independence, $Pe_r$={gate_d['pe']:g}: spread "
            f"{gate_d['spread_pct']:.2f} % (gate {gate_d['threshold_pct']:g} %)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        ax = axes[1]
        names = list(gate_e["rates"].keys())
        rates = [gate_e["rates"][n] for n in names]
        order = np.argsort(rates)
        ax.barh([names[i] for i in order], [rates[i] for i in order], color="C0")
        ax.axvline(gate_e["threshold"], color="r", ls="--", label="gate 1e-10 / 1e4 steps")
        ax.set_xscale("log")
        ax.set_xlabel("relative mass drift per 1e4 steps")
        ax.set_title("scalar mass budget (fp64)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="x", which="both")
        fig.tight_layout()
        path = FIG_DIR / "fig3_schmidt_mass.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P74-N10 transport surface")
    parser.add_argument("--list", action="store_true", help="list the jobs")
    parser.add_argument("--only", default=None, help="comma separated case names")
    parser.add_argument("--all", action="store_true", help="run every missing case")
    parser.add_argument("--force", action="store_true", help="recompute cached cases")
    parser.add_argument("--assemble", action="store_true", help="gates + artifact + figures")
    parser.add_argument("--no-figs", action="store_true", help="skip figures")
    parser.add_argument("--quick", action="store_true", help="reduced spec (pytest budget)")
    args = parser.parse_args(argv)

    if args.quick:
        from tensorlbm.passive_scalar_transport_verification import quick_transport_spec

        spec = quick_transport_spec()
    else:
        spec = TransportSpec()

    if args.list:
        for case in surface_cases(spec):
            print(f"{case['name']:24s} {case['kind']:16s} {case}")
        return 0

    if args.all or args.only:
        only = None
        if args.only:
            only = [name.strip() for name in args.only.split(",") if name.strip()]
        started = time.time()
        run_surface(spec, case_dir=CASE_DIR, only=only, force=args.force, verbose=True)
        print(f"[p74n10] chunk finished in {time.time() - started:.1f} s", flush=True)
        if not args.assemble:
            return 0

    cases = _load_cases()
    if not cases:
        print("no cached cases: run with --all first")
        return 2
    report = evaluate_gates(cases, spec)
    report.update(
        {
            "node": "P74-N10",
            "label": spec.label,
            "spec": spec.to_json(),
            "generated_unix": time.time(),
            "case_files": [f"artifacts/passive_scalar_p74n10/cases/{c['name']}.json" for c in cases],
            "environment": {"python": sys.version.split()[0], "repo": str(REPO_ROOT)},
        }
    )
    if not args.no_figs:
        figs = _make_figures(report)
        report["figures"] = [str(p.relative_to(REPO_ROOT)) for p in figs]
    write_artifact(report, ARTIFACT)
    _print_gate_summary(report)
    print(f"artifact: {ARTIFACT}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())