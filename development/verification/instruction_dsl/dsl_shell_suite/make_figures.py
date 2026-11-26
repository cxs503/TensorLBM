#!/usr/bin/env python3
"""Figures + numbers for the shell-suite report (every value is computed here).

The report must not contain hand-typed numbers, so this script re-runs the two
physics cases of the conformance suite, writes
``artifacts/figures_data.json`` (the exact numbers inserted into the report) and
renders:

* ``fig_poiseuille.png``  -- channel velocity profile, DSL result vs the
  analytical parabola u(y) = g (y-1/2)(H-(y-1/2))/(2 nu);
* ``fig_tg_decay.png``    -- Taylor-Green kinetic-energy decay, DSL result vs
  the analytical exp(-4 nu k^2 t) line;
* ``fig_case_margins.png``-- per-case relative margin against the acceptance
  bands used by the suite (3 % physics / 1e-10 exact), read back from
  ``artifacts/results.tsv``-independent re-execution of each numerical claim.

Usage: python3 make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorlbm.instruction_dsl import decay_rate_from_history, parse, poiseuille_error, run

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
CASES = HERE / "cases"


def program(name: str) -> str:
    return (CASES / name / "program.lbm").read_text(encoding="utf-8")


def poiseuille_figure(data: dict) -> Path:
    prog = parse(program("v04_channel_poiseuille"))
    res = run(prog, sample_every=1000)
    ny = prog.ny
    nu = res.scheme.nu
    g = prog.force[0]
    H = ny - 2
    y = np.arange(ny) + 0.5  # cell centres; walls at 0.5 and ny-0.5
    ux = res.ux.mean(axis=0)
    wall = np.clip(y - 0.5, 0.0, H)
    analytic = g * wall * (H - wall) / (2.0 * nu)

    fluid = slice(1, ny - 1)
    err = poiseuille_error(res.ux, ny, nu=nu, body_force=g)

    data["poiseuille"] = {
        "ny": ny,
        "H": H,
        "nu_declared": nu,
        "g": g,
        "steps": prog.steps,
        "u_max_numeric": float(ux[fluid].max()),
        "u_max_analytic": float(analytic[fluid].max()),
        "u_max_rel_error": float(
            abs(ux[fluid].max() - analytic[fluid].max()) / analytic[fluid].max()
        ),
        "profile_max_rel_error": float(err["max_rel_error"]),
        "converged": bool(res.converged),
        "y": y.tolist(),
        "ux_numeric": ux.tolist(),
        "ux_analytic": analytic.tolist(),
    }

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(ux, y, "o", ms=4, label="instruction DSL (12000 steps)")
    ax.plot(analytic, y, "-", lw=1.4, label="analytic $g\\,y(H-y)/(2\\nu)$")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.axhline(ny - 0.5, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("$u_x$ (lattice units)")
    ax.set_ylabel("$y$ (lattice units)")
    ax.set_title(f"Channel Poiseuille, {prog.nx}x{ny}, max err {err['max_rel_error']*100:.2f} %")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ART / "fig_poiseuille.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def tg_figure(data: dict) -> Path:
    prog = parse(program("v03_tg_decay_32"))
    res = run(prog, sample_every=2)
    nx = prog.nx
    k = 2.0 * math.pi / nx
    amp = prog.init["amp"]
    nu = res.scheme.nu
    steps = np.array([h["step"] for h in res.history], dtype=float)
    ke = np.array([h["kinetic_energy"] for h in res.history], dtype=float)
    ke0 = 0.25 * amp * amp
    ke_ana = ke0 * np.exp(-4.0 * nu * k * k * steps)
    nu_fit, r2 = decay_rate_from_history(res.history, k)
    rel = abs(nu_fit - nu) / nu
    mask = ke > 0
    ke_rel = float(np.max(np.abs(ke[mask] - ke_ana[mask]) / ke_ana[mask]))

    data["taylor_green"] = {
        "nx": nx,
        "amp": amp,
        "steps": int(steps[-1]),
        "nu_declared": nu,
        "nu_fitted": float(nu_fit),
        "nu_rel_error": float(rel),
        "r2": float(r2),
        "ke_max_rel_error_vs_analytic": ke_rel,
        "k": k,
        "history_steps": steps.tolist(),
        "history_ke": ke.tolist(),
        "ke_analytic": ke_ana.tolist(),
    }

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax.semilogy(steps, ke, "o", ms=3, label="instruction DSL")
    ax.semilogy(steps, ke_ana, "-", lw=1.3, label="analytic $\\frac{A^2}{4}e^{-4\\nu k^2 t}$")
    ax.set_xlabel("step")
    ax.set_ylabel("kinetic energy")
    ax.set_title("Taylor-Green decay, 32$^2$, $A$=0.05")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ln_ke = np.log(ke)
    slope, intercept = np.polyfit(steps, ln_ke, 1)
    ax2.plot(steps, ln_ke, "o", ms=3, label="ln KE (DSL)")
    ax2.plot(steps, slope * steps + intercept, "-", lw=1.2,
             label=f"fit slope {slope:.6f}")
    ax2.plot(steps, -4.0 * nu * k * k * steps + ln_ke[0], "--", lw=1.2,
             label=f"analytic slope {-4 * nu * k * k:.6f}")
    ax2.set_xlabel("step")
    ax2.set_ylabel("ln(kinetic energy)")
    ax2.set_title(f"$\\nu$ fit {nu_fit:.6f} vs declared {nu:.6f} (R$^2$={r2:.6f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = ART / "fig_tg_decay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def margins_figure(data: dict) -> Path:
    """Per-case margin of every numerical claim against its acceptance band."""
    summary = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    rows = []
    for case in sorted(CASES.iterdir()):
        expect = case / "expect.conf"
        if not expect.is_file():
            continue
        expect_text = expect.read_text(encoding="utf-8")
        if "error_class=" in expect_text:  # rejection cases carry no numeric claim
            continue
        program_text = (case / "program.lbm").read_text(encoding="utf-8")
        record = run(parse(program_text), sample_every=0)
        got = {
            "u_max": float(record.summary()["u_max"]),
            "kinetic_energy": float(record.kinetic_energy),
            "nu": float(record.scheme.nu),
            "tau": float(record.scheme.tau),
            "cs2": float(record.scheme.cs2),
        }
        for line in expect.read_text(encoding="utf-8").splitlines():
            if not line.startswith(("approx.", "abs.")):
                continue
            mode, rest = line.split(".", 1)
            key, spec = rest.split("=", 1)
            ref_s, tol_s = spec.split(":")
            ref, tol = float(ref_s), float(tol_s)
            if key not in got:
                continue
            d = abs(got[key] - ref)
            if mode == "approx" and ref != 0:
                d = d / abs(ref)
            rows.append((f"{case.name}:{key}", mode, d, tol))

    data["case_margins"] = [
        {"claim": c, "mode": m, "error": e, "band": t,
         "margin_decades": float(math.log10(t / e)) if e > 0 else 99.0}
        for c, m, e, t in rows
    ]

    labels = [r[0].replace("v0", "").replace(":", "\n") for r in rows]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    x = np.arange(len(rows))
    for i, (claim, mode, err, tol) in enumerate(rows):
        used = err if err > 0 else 1e-17
        ax.bar(i, used, color="#2b6cb0", width=0.6)
        ax.plot([i - 0.35, i + 0.35], [tol, tol], color="#c53030", lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("measured error (log)")
    ax.set_title("Per-claim error vs acceptance band (red bar = allowed tolerance)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = ART / "fig_case_margins.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    data["suite"] = {"total": summary["total"], "passed": summary["passed"],
                     "failed": summary["failed"]}
    return out


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if not (ART / "results.json").is_file():
        raise SystemExit("run ./run_cases.sh first: artifacts/results.json is the suite record")
    figs = [poiseuille_figure(data), tg_figure(data), margins_figure(data)]
    data["figures"] = [p.name for p in figs]
    (ART / "figures_data.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "case_margins"}, indent=2,
                     sort_keys=True)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())