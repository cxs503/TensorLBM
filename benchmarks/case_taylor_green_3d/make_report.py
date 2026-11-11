#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-3D report builder: JSON artefacts -> tables + figures + summary.md.

Every number in the generated tables/figures is read from
``runs/case_taylor_green_3d/*.json`` (written by ``run_case.py``); nothing is
typed by hand.  Figures go to ``runs/case_taylor_green_3d/figures/``.

Usage::

    python benchmarks/case_taylor_green_3d/make_report.py [--runs DIR] [--figs DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs", "case_taylor_green_3d")
FIGS = os.path.join(RUNS, "figures")

TOL_PCT = 8.0
TOL_MASS = 1e-12
TOL_ORDER = 1.8


def load(name: str) -> dict | None:
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def mark(value: float, tol: float, lower_is_better: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "[N/A]"
    ok = abs(value) <= tol if lower_is_better else value >= tol
    return "[OK]" if ok else "[FAIL]"


def fmt(x: float, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (abs(x) < 1e-3 and x != 0.0):
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
def table_verification(v: dict) -> str:
    rows = [
        "| case | curl(C)/\\|C\\| | convective term is a gradient | \\|lap(u)+k2u\\|/\\|k2u\\| | NS residual (rel) | exact solution |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("tg", "tg3d", "abc"):
        e = v[key]
        rows.append(
            "| {k} | {c} | {g} | {l} | {r} | {s} |".format(
                k=key,
                c=fmt(e["curl_of_convective_term_over_norm"]),
                g="yes" if e["convective_term_is_pure_gradient"] else "**no**",
                l=fmt(e.get("laplacian_identity_rel_residual")),
                r=fmt(e.get("ns_momentum_residual_rel")),
                s=e.get("exact_solution") or "**none (no closed form)**",
            )
        )
    rows.append("")
    rows.append(
        f"ABC is Beltrami: `||curl(u)-k u||/||curl(u)||` = "
        f"{v['abc_beltrami_rel_residual']:.2e} (machine-checked, N={v['tg']['n']})."
    )
    return "\n".join(rows)


def table_primary(d: dict) -> str:
    rows = [
        "| metric | tolerance | got | |",
        "|---|---|---|---|",
        f"| M1 energy decay ratio rel. error (T={d['steps']}, N={d['n']}) | < {TOL_PCT:.0f}% | "
        f"{d['M1_energy_decay_rel_err_pct']:+.3e}% | {mark(d['M1_energy_decay_rel_err_pct'], TOL_PCT)} |",
        f"| M2 energy decay-exponent rel. error (fit over 0..{d['steps']}) | < {TOL_PCT:.0f}% | "
        f"{d['M2_decay_exponent_rel_err_pct']:+.3e}% | {mark(d['M2_decay_exponent_rel_err_pct'], TOL_PCT)} |",
        f"| M3 relative L2 velocity error at T | < {TOL_PCT:.0f}% | "
        f"{d['M3_l2_field_rel_err_pct']:.3e}% | {mark(d['M3_l2_field_rel_err_pct'], TOL_PCT)} |",
        f"| M5 mass drift | < {TOL_MASS:.0e} | {d['M5_mass_drift_rel']:.2e} | "
        f"{mark(d['M5_mass_drift_rel'], TOL_MASS)} |",
        f"| finite field | required | {d['finite']} | {'[OK]' if d['finite'] else '[FAIL]'} |",
    ]
    return "\n".join(rows)


def table_convergence(c: dict) -> str:
    rows = [
        "| N | k=2π/N | \\|k\\|² | L2 rel. error | observed order vs previous | wall [s] |",
        "|---|---|---|---|---|---|",
    ]
    prev = None
    for r in c["runs"]:
        e = r["M3_l2_field_rel_err_pct"] / 100.0
        order = "" if prev is None else f"{math.log2(prev / e):.3f}"
        rows.append(
            f"| {r['n']} | {2*math.pi/r['n']:.5f} | {r['k2']:.6f} | {e:.4e} | {order} | {r['wall_s']:.1f} |"
        )
        prev = e
    rows.append("")
    rows.append(
        f"M4 observed order (24→48) = **{c['M4_observed_order']:.3f}** "
        f"{mark(c['M4_observed_order'], TOL_ORDER, lower_is_better=False)} "
        f"(target ≥ {TOL_ORDER}); 24→32 = {fmt(c['M4_order_24_32'])}, 48→64 = {fmt(c['M4_order_48_64'])}."
    )
    return "\n".join(rows)


def table_tau(t: dict) -> str:
    rows = [
        "| τ | ν=(τ-0.5)/3 | Re=A/(ν√\\|k\\|²) | λ_exact=2ν\\|k\\|² | λ_fit | rel. error | |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t["runs"]:
        rows.append(
            f"| {r['tau']:.2f} | {r['nu']:.5f} | {r['re_lattice']:.2f} | {r['lambda_exact']:.6e} | "
            f"{r['lambda_fit']:.6e} | {r['M2_decay_exponent_rel_err_pct']:+.3e}% | "
            f"{mark(r['M2_decay_exponent_rel_err_pct'], TOL_PCT)} |"
        )
    return "\n".join(rows)


def table_amp(a: dict) -> str:
    rows = [
        "| A | Ma=A√3 | M1 rel. error | M2 rel. error | M3 L2 rel. error |",
        "|---|---|---|---|---|",
    ]
    for r in a["runs"]:
        rows.append(
            f"| {r['amp']:.1e} | {r['ma']:.4f} | {r['M1_energy_decay_rel_err_pct']:+.3e}% | "
            f"{r['M2_decay_exponent_rel_err_pct']:+.3e}% | {r['M3_l2_field_rel_err_pct']:.3e}% |"
        )
    return "\n".join(rows)


def table_abc(x: dict) -> str:
    rows = [
        "| N | |k|² | M1 energy rel. error | M2 exponent rel. error | M3 L2 rel. error |",
        "|---|---|---|---|---|",
    ]
    for r in x["runs"]:
        rows.append(
            f"| {r['n']} | {r['k2']:.6f} | {r['M1_energy_decay_rel_err_pct']:+.3e}% | "
            f"{r['M2_decay_exponent_rel_err_pct']:+.3e}% | {r['M3_l2_field_rel_err_pct']:.3e}% |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
def figures(v, d, c, t, a, x, figs_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figs_dir, exist_ok=True)
    made = []

    if d:
        ts = [h["t"] for h in d["history"]]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.semilogy(ts, [h["E"] for h in d["history"]], "o", ms=3, label="D3Q27 LBM")
        ax.semilogy(ts, [h["E_exact"] for h in d["history"]], "-", lw=1.2,
                    label=r"exact $\frac{1}{2}\langle|u|^2\rangle_0 e^{-2\nu|k|^2t}$")
        ax.set_xlabel("lattice steps")
        ax.set_ylabel("kinetic energy per volume")
        ax.set_title(f"Taylor-Green 3D decay, N={d['n']}, τ={d['tau']}, A={d['amp']}")
        ax.legend()
        ax.grid(alpha=0.3)
        p = os.path.join(figs_dir, "energy_decay.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(ts, [h["E_rel_err_pct"] for h in d["history"]], ".-")
        ax.axhline(TOL_PCT, ls="--", c="r", lw=1, label=f"tolerance ±{TOL_PCT:.0f}%")
        ax.axhline(-TOL_PCT, ls="--", c="r", lw=1)
        ax.set_xlabel("lattice steps")
        ax.set_ylabel("energy rel. error [%]")
        ax.set_title("instantaneous energy error vs exact solution")
        ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(figs_dir, "energy_error.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    if c:
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        ns = [r["n"] for r in c["runs"]]
        es = [r["M3_l2_field_rel_err_pct"] / 100.0 for r in c["runs"]]
        ax.loglog(ns, es, "o-", label="measured L2 error")
        ref = es[0] * (ns[0] / ns[-1]) ** 2
        ax.loglog([ns[0], ns[-1]], [es[0], ref], "k--", label=r"$N^{-2}$ reference")
        ax.set_xlabel("N"); ax.set_ylabel(r"$\|u_N-u_{ex}\|_2/\|u_{ex}\|_2$")
        ax.set_title(f"grid convergence (order 24→48 = {c['M4_observed_order']:.2f})")
        ax.legend(); ax.grid(alpha=0.3, which="both")
        p = os.path.join(figs_dir, "grid_convergence.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    if t:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.plot([r["tau"] for r in t["runs"]], [r["lambda_exact"] for r in t["runs"]], "k--",
                label=r"exact $2\nu|k|^2$")
        ax.plot([r["tau"] for r in t["runs"]], [r["lambda_fit"] for r in t["runs"]], "o",
                label="fitted from LBM energy history")
        ax.set_xlabel("τ"); ax.set_ylabel("energy decay exponent [1/step]")
        ax.set_title("relaxation-time sensitivity"); ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(figs_dir, "tau_sensitivity.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    if a:
        ma = [r["ma"] for r in a["runs"]]
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.loglog(ma, [r["M3_l2_field_rel_err_pct"] for r in a["runs"]], "o-", label="L2 field error")
        ax.loglog(ma, [abs(r["M1_energy_decay_rel_err_pct"]) for r in a["runs"]], "s-",
                  label="|energy decay error|")
        ref = [a["runs"][0]["M3_l2_field_rel_err_pct"] * (m / ma[0]) ** 2 for m in ma]
        ax.loglog(ma, ref, "k--", label=r"$Ma^2$ reference")
        ax.set_xlabel("Mach number  Ma = A√3"); ax.set_ylabel("relative error [%]")
        ax.set_title("amplitude / compressibility sensitivity"); ax.legend(); ax.grid(alpha=0.3, which="both")
        p = os.path.join(figs_dir, "amp_sensitivity.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    if x:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for r in x["runs"]:
            ts = [h["t"] for h in r["history"]]
            ax.semilogy(ts, [h["E"] for h in r["history"]], "-", lw=1.0, label=f"LBM N={r['n']}")
        r0 = x["runs"][0]
        ax.semilogy([h["t"] for h in r0["history"]], [h["E_exact"] for h in r0["history"]], "k--",
                    lw=1.2, label="exact (Beltrami decay)")
        ax.set_xlabel("lattice steps"); ax.set_ylabel("kinetic energy per volume")
        ax.set_title("z-dependent companion: ABC (Beltrami) exact decay")
        ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(figs_dir, "abc_decay.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    return made


def main() -> int:
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--figs", default=FIGS)
    args = ap.parse_args()
    RUNS = args.runs

    v = load("exactness_verification.json")
    d = load("primary_decay.json")
    c = load("convergence.json")
    t = load("tau_sensitivity.json")
    a = load("amp_sensitivity.json")
    x = load("abc_zdependent.json")

    figs = figures(v, d, c, t, a, x, args.figs)
    parts = ["# CASE-TAYLOR-GREEN-3D — generated evidence tables", ""]
    if v:
        parts += ["## 0. Reference exactness (machine-checked, spectral derivatives)", "", table_verification(v), ""]
    if c:
        parts += ["## 1. Grid convergence (M4)", "", table_convergence(c), ""]
    if d:
        parts += ["## 2. Primary decay case (M1/M2/M3/M5)", "", table_primary(d), ""]
    if t:
        parts += ["## 3. τ sensitivity (S1)", "", table_tau(t), ""]
    if a:
        parts += ["## 4. Amplitude / Mach sensitivity (S2)", "", table_amp(a), ""]
    if x:
        parts += ["## 5. z-dependent exact companion: Beltrami/ABC flow", "", table_abc(x), ""]
    if figs:
        parts += ["## Figures", ""] + [f"- `{os.path.relpath(p, ROOT)}`" for p in figs] + [""]
    md = "\n".join(parts)
    out = os.path.join(RUNS, "summary.md")
    os.makedirs(RUNS, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(md + "\n")
    print(md)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())