"""Entry point: cross-validate the two-component Shan-Chen model on layered Poiseuille.

The script runs a pre-registered case matrix with :class:`TwoComponentSCSolver`
and compares every run against the *exact* layered plane-Poiseuille solution of
:func:`..analytic.solve_layered` (piecewise quadratic, velocity and shear-stress
continuity at the interface).  It writes one JSON artifact (profiles + metrics +
gate verdicts) and a figure, and prints a table.

Usage
-----
::

    cd TensorLBM
    PYTHONPATH=src python -m tensorlbm.verification.layered_poiseuille.run_two_component_sc_validation \
        --matrix core --out artifacts/xval_two_component_sc_layered_poiseuille_edm

``--case NAME`` runs a single case (used by the cluster shard driver).

Pre-registered gates
--------------------
``GATE`` per case kind, fixed before the runs (owner standard: >15% error is not
acceptable; self target 3%):

* ``control``  (one component, G = 0): ``eps_inf <= 3%``  -- proves the viscous
  solution is actually reached (the defect that sank the previous delivery was
  ``u_max = g*t``, i.e. no viscous braking at all).
* ``layered``  (two components, G > 0): ``u_peak_err <= 15%``, ``Q_err <= 15%``,
  ``eps_l2_rel <= 15%``, ``eps_inf <= 15%`` (full field, no cells excluded).
* ``static``   (two components, g = 0): spurious current ``max|u| / u_ref <= 15%``
  (``u_ref`` = the analytic peak of the same configuration at ``g_x = 5e-5``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .analytic import solve_layered
from .metrics import layer_fluxes, peak_metrics, profile_errors, rel_err
from .two_component_sc import (TwoComponentSCConfig, TwoComponentSCSolver,
                               layered_reference)

#: owner standard: a deviation above this is not acceptable
GATE_OWNER = 0.15
#: publication target used by the sibling layered-Poiseuille anchor
GATE_TARGET = 0.03

CASE_MATRIX: Dict[str, Dict[str, Any]] = {
    # -- controls: the solver must reproduce a single-fluid Poiseuille profile ----
    "control_single": dict(
        kind="control",
        cfg=dict(ny=34, nx=1, tau_a=0.8, tau_b=0.8, G=0.0, g_x=1e-4, g_y=0.0, steps=6000,
                 init_rho_a=2.0, init_rho_b=0.0, init_layer_frac=1.0, interface_width=1.0),
        note="one component (A only), nu=0.1, exact single-fluid Poiseuille",
    ),
    "control_mixture": dict(
        kind="control",
        cfg=dict(ny=34, nx=1, tau_a=0.8, tau_b=0.8, G=0.0, g_x=1e-4, g_y=0.0, steps=6000,
                 init_rho_a=2.0, init_rho_b=2.0, init_layer_frac=0.5, interface_width=2.0),
        note="two components, G=0 -> they mix to rho_total=2, equal tau; exact single-fluid Poiseuille",
    ),
    # -- anchor cases: two-component layered Poiseuille --------------------------
    "layered_r167": dict(
        kind="layered",
        cfg=dict(ny=34, nx=1, tau_a=0.8, tau_b=1.0, G=2.5, g_x=5e-5, g_y=0.0, steps=12000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=2000),
        note="nu_A=0.1000 (bottom), nu_B=0.1667 (top): viscosity ratio 1.67",
    ),
    "layered_r233": dict(
        kind="layered",
        cfg=dict(ny=34, nx=1, tau_a=0.8, tau_b=1.2, G=2.5, g_x=5e-5, g_y=0.0, steps=12000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=2000),
        note="nu_A=0.1000 (bottom), nu_B=0.2333 (top): viscosity ratio 2.33",
    ),
    "layered_r167_h48": dict(
        kind="layered",
        cfg=dict(ny=50, nx=1, tau_a=0.8, tau_b=1.0, G=2.5, g_x=5e-5, g_y=0.0, steps=16000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=2000),
        note="resolution check: same physics, H=48 instead of H=32",
    ),
    # -- static interface: spurious currents -------------------------------------
    "layered_static": dict(
        kind="static",
        cfg=dict(ny=34, nx=1, tau_a=0.8, tau_b=1.0, G=2.5, g_x=0.0, g_y=0.0, steps=6000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=2000),
        note="g=0: residual velocity must stay far below the driven peak",
    ),
}

MATRICES = {
    "core": ["control_single", "control_mixture", "layered_r167", "layered_r233", "layered_static"],
    "full": list(CASE_MATRIX),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git_commit(repo_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return "unknown"


def _finite_or_none(obj):
    """Recursively map non-finite floats to ``None`` (JSON ``null``).

    Board-audit m21704 F5/R3: the artifact must be strict-JSON / PostgreSQL
    ``jsonb``-safe, so ``NaN``/``Inf`` (e.g. an unlocatable interface or an
    under-resolved viscosity fit) are serialised as ``null`` instead of bare
    ``NaN`` tokens that break ``json.loads`` strict parsing and ``jsonb`` ingest.
    """
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def interface_position(phi: np.ndarray, y: np.ndarray, skip_wall: int = 2) -> float:
    """``y`` where the A-fraction crosses 0.5 (linear interpolation, walls skipped)."""
    sel = slice(skip_wall, len(y) - skip_wall)
    yy, pp = y[sel], phi[sel]
    below = np.where(pp >= 0.5)[0]
    if len(below) == 0 or below[-1] + 1 >= len(pp):
        return float("nan")
    k = below[-1]
    t = (pp[k] - 0.5) / (pp[k] - pp[k + 1])
    return float(yy[k] + t * (yy[k + 1] - yy[k]))


def layer_viscosity_fit(u: np.ndarray, y: np.ndarray, g: float, lo: float, hi: float) -> float:
    """Effective kinematic viscosity of a layer from its own curvature.

    In a steady unidirectional flow with constant viscosity ``u'' = -g/nu``,
    so a quadratic fit inside one layer returns ``nu = -g / (2 a2)``.  The fit
    window stays clear of the walls and of the interface.
    """
    sel = (y >= lo) & (y <= hi)
    if int(sel.sum()) < 4:
        return float("nan")
    a2, _, _ = np.polyfit(y[sel], u[sel], 2)
    return float(-g / (2.0 * a2)) if a2 != 0.0 else float("nan")


def case_reference(case: Dict[str, Any], cfg: TwoComponentSCConfig):
    """Exact reference solution for one case."""
    if case["kind"] == "control":
        # both components share tau -> single-fluid channel of the whole height
        nu = cfg.nu_a
        sol = solve_layered(cfg.height, [0.5 * cfg.height], [nu, nu], cfg.g_x)
        return sol, [0.5 * cfg.height], nu
    if case["kind"] in ("layered", "static"):
        sol = layered_reference(cfg)
        return sol, [0.5 * cfg.height], None
    raise ValueError(case["kind"])


# --------------------------------------------------------------------------- #
# one case
# --------------------------------------------------------------------------- #
def run_case(name: str, *, verbose: bool = True) -> Dict[str, Any]:
    """Run one of the pre-registered :data:`CASE_MATRIX` cases by name."""
    return run_case_spec(CASE_MATRIX[name], name=name, verbose=verbose)


def run_case_spec(case: Dict[str, Any], *, name: Optional[str] = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """Run an arbitrary case specification (``{"kind", "cfg", "note"}``).

    Factored out of :func:`run_case` so other callers - the paper-evidence
    benchmark harness and the cross-validation registry - drive exactly the
    same scoring code instead of re-implementing the reference comparison.
    """
    name = name or case.get("case", "unnamed")
    cfg = TwoComponentSCConfig(**case["cfg"])
    solver = TwoComponentSCSolver(cfg)
    t0 = time.time()
    out = solver.run()
    wall = time.time() - t0
    prof = out["final_profile"]
    y = prof["y"]
    u = prof["ux_total"]

    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(prof["rho_a"])):
        return {"case": name, "kind": case["kind"], "status": "DIVERGED",
                "note": case["note"], "config": case["cfg"], "wall_seconds": wall}

    sol, interfaces, nu_single = case_reference(case, cfg)
    u_ref = sol.u(y)
    errs = profile_errors(u, u_ref)
    pm = peak_metrics(u, y)
    pk_ref = float(np.max(u_ref))
    q_num = layer_fluxes(u, y, interfaces)
    q_ref = layer_fluxes(u_ref, y, interfaces)
    phi = prof["rho_a"] / (prof["rho_a"] + prof["rho_b"] + 1e-12)

    rec: Dict[str, Any] = {
        "case": name,
        "kind": case["kind"],
        "note": case["note"],
        "status": "RAN",
        "config": case["cfg"],
        "wall_seconds": round(wall, 1),
        "height": cfg.height,
        "nu_a": cfg.nu_a,
        "nu_b": cfg.nu_b,
        "metrics": {
            "u_peak_err": rel_err(pm["u_peak_num"], pk_ref),
            "u_peak_num": pm["u_peak_num"],
            "u_peak_ref": pk_ref,
            "y_peak_num": pm["y_peak_num"],
            "eps_inf": errs["eps_inf"],
            "eps_l2_rel": errs["eps_l2_rel"],
            "eps_mean": errs["eps_mean"],
            "Q_err": rel_err(q_num[0] + q_num[1], q_ref[0] + q_ref[1]),
            "Q_layer_err": [rel_err(a, b) for a, b in zip(q_num, q_ref)],
            "Q_num": q_num,
            "Q_ref": q_ref,
            "y_iface": interface_position(phi, y),
            "y_iface_ref": interfaces[0],
            "y_iface_err": abs(interface_position(phi, y) - interfaces[0]) / cfg.height,
            "rho_total_min": float(prof["rho_total"].min()),
            "rho_total_max": float(prof["rho_total"].max()),
            "mass_a": float(prof["rho_a"].sum()),
            "mass_b": float(prof["rho_b"].sum()),
            "steady_drift": float(out["max_profile_drift"]),
        },
        "profiles": {k: [float(v) for v in prof[k]] for k in
                     ("y", "rho_a", "rho_b", "rho_total", "ux_total")},
        "u_ref": [float(v) for v in u_ref],
        "phi": [float(v) for v in phi],
    }

    # layer-wise effective viscosity (layered/static cases only)
    if case["kind"] in ("layered",):
        half = 0.5 * cfg.height
        rec["metrics"]["nu_fit_lower"] = layer_viscosity_fit(u, y, cfg.g_x, 2.0, half - 4.0)
        rec["metrics"]["nu_fit_upper"] = layer_viscosity_fit(u, y, cfg.g_x, half + 4.0, cfg.height - 2.0)
        rec["metrics"]["nu_fit_err_lower"] = rel_err(rec["metrics"]["nu_fit_lower"], cfg.nu_a)
        rec["metrics"]["nu_fit_err_upper"] = rel_err(rec["metrics"]["nu_fit_upper"], cfg.nu_b)

    # ---- pre-registered gates -------------------------------------------- #
    m = rec["metrics"]
    gates: Dict[str, Any] = {}
    if case["kind"] == "control":
        gates["control_eps_inf<=3%"] = m["eps_inf"] <= GATE_TARGET
        gates["control_u_peak<=3%"] = m["u_peak_err"] <= GATE_TARGET
    elif case["kind"] == "layered":
        gates["u_peak_err<=15%"] = m["u_peak_err"] <= GATE_OWNER
        gates["Q_err<=15%"] = m["Q_err"] <= GATE_OWNER
        gates["eps_l2_rel<=15%"] = m["eps_l2_rel"] <= GATE_OWNER
        gates["eps_inf<=15%"] = m["eps_inf"] <= GATE_OWNER
        gates["interface_position<=5%H"] = m["y_iface_err"] <= 0.05
    elif case["kind"] == "static":
        # reference scale: the analytic peak of the same configuration, driven
        driven = solve_layered(cfg.height, [0.5 * cfg.height], [cfg.nu_a, cfg.nu_b], 5e-5)
        u_scale = driven.u_peak()
        m["spurious_ratio"] = float(np.max(np.abs(u)) / u_scale)
        gates["spurious_ratio<=15%"] = m["spurious_ratio"] <= GATE_OWNER
    rec["gates"] = gates
    rec["gate_pass"] = all(gates.values()) if gates else None
    if verbose:
        print(f"[{name:18s}] {case['kind']:8s} " + " ".join(
            f"{k}={v*100:.2f}%" for k, v in m.items()
            if isinstance(v, float) and k.startswith(("u_peak_err", "eps_", "Q_err", "y_iface_err", "spurious"))
        ) + f"  gates={'PASS' if rec['gate_pass'] else 'FAIL'}  {wall:.0f}s", flush=True)
    return rec


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(records: List[Dict[str, Any]], png_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ran = [r for r in records if r.get("status") == "RAN"]
    if not ran:
        raise RuntimeError("no completed case to plot")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for r in ran:
        if r["kind"] in ("layered", "control"):
            ax.plot(r["profiles"]["ux_total"], r["profiles"]["y"], lw=1.4, label=f"{r['case']} (num)")
            ax.plot(r["u_ref"], r["profiles"]["y"], "k--", lw=0.9)
    ax.set_xlabel(r"$u_x$ [lu]")
    ax.set_ylabel("wall distance y [lu]")
    ax.set_title("velocity profiles (dashed = exact layered solution)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in ran:
        if r["kind"] == "layered":
            ax.plot(r["phi"], r["profiles"]["y"], lw=1.4, label=f"{r['case']} $\\phi_A$")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\phi_A=\rho_A/(\rho_A+\rho_B)$")
    ax.set_ylabel("wall distance y [lu]")
    ax.set_title("demixing: A below, B above")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for r in ran:
        if r["kind"] in ("layered", "control"):
            u = np.asarray(r["profiles"]["ux_total"])
            ref = np.asarray(r["u_ref"])
            ax.plot(100 * (u - ref) / np.max(ref), r["profiles"]["y"], lw=1.3, label=r["case"])
    ax.axvline(0.0, color="k", lw=0.6)
    ax.axvline(15.0, color="r", ls="--", lw=0.8, label="±15% owner gate")
    ax.axvline(-15.0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("local error [% of exact peak]")
    ax.set_ylabel("wall distance y [lu]")
    ax.set_title("pointwise deviation from the exact solution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="core", choices=sorted(MATRICES))
    ap.add_argument("--case", default=None, help="run a single case by name")
    ap.add_argument("--out", default="artifacts/xval_two_component_sc_layered_poiseuille_edm",
                    help="artifact directory (created if missing)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="rebuild the figure from results.json")
    ap.add_argument("--json-only", default=None, help="write just this case to <out>/<name>.json")
    args = ap.parse_args(argv)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    def _load(path):
        if os.path.exists(path):
            with open(path) as fh:
                return {r["case"]: r for r in json.load(fh).get("records", [])}
        return {}

    json_path0 = os.path.join(out_dir, "results.json")
    if args.figure_only:
        old = list(_load(json_path0).values())
        make_figure(old, os.path.join(out_dir, "two_component_sc_layered.png"))
        print(f"figure rebuilt from {len(old)} records")
        return 0

    names = [args.case] if args.case else MATRICES[args.matrix]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    acc = _load(json_path0)
    for name in names:
        if name not in CASE_MATRIX:
            raise SystemExit(f"unknown case {name}; choose from {sorted(CASE_MATRIX)}")
        rec = run_case(name)
        acc[name] = rec
        if args.json_only:
            with open(os.path.join(out_dir, args.json_only), "w") as fh:
                json.dump(_finite_or_none(rec), fh, indent=1)

    records = [acc[k] for k in CASE_MATRIX if k in acc]
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    payload = {
        "benchmark": "xval_two_component_sc_layered_poiseuille_edm",
        "anchor": "XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM",
        "model": {
            "kind": "two-component Shan-Chen (MCMP) D2Q9 BGK, one f per component",
            "pseudopotential": "psi = rho0 (1 - exp(-rho/rho0)), rho0 = 1",
            "interaction": "F_sigma = -G psi_sigma sum_i w_i psi_sigma'(x+c_i) c_i (G>0 repulsive)",
            "force_scheme": "EDM (Kupershtokh 2004): f += feq(rho, u*+F/rho) - feq(rho, u*)",
            "u_eq": "viscous (Martys & Chen 1996 viscosity-weighted coupling)",
            "walls": "halfway bounce-back at rows 0 / ny-1, wall_mode=mirror (neutral wetting)",
            "reference": "exact piecewise-quadratic layered solution, analytic.solve_layered "
                         "(2N x 2N LU solve, independent of the solver)",
        },
        "gates": {"owner_standard": GATE_OWNER, "self_target": GATE_TARGET,
                  "rule": "control eps_inf<=3%; layered u_peak/Q/eps_l2/eps_inf<=15%, "
                          "interface position<=5%H; static spurious<=15% of driven peak"},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "git_commit": _git_commit(repo_root),
            "argv": sys.argv,
        },
        "records": records,
        "summary": {
            "n_cases": len(records),
            "n_gate_pass": sum(1 for r in records if r.get("gate_pass")),
            "n_diverged": sum(1 for r in records if r.get("status") != "RAN"),
            "all_gates_pass": all(bool(r.get("gate_pass")) for r in records),
        },
    }

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as fh:
        json.dump(_finite_or_none(payload), fh, indent=1)
    digest = hashlib.sha256(open(json_path, "rb").read()).hexdigest()
    payload["sha256"] = digest
    with open(os.path.join(out_dir, "sha256.txt"), "w") as fh:
        fh.write(digest + "  results.json\n")

    if not args.no_figure:
        make_figure(records, os.path.join(out_dir, "two_component_sc_layered.png"))

    print(json.dumps(payload["summary"], indent=1))
    print(f"artifact: {json_path} (sha256 {digest[:16]}...)")
    return 0 if payload["summary"]["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())