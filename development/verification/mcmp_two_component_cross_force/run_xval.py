#!/usr/bin/env python3
"""XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE - driver.

Judges the two-component Shan-Chen *cross force* of TensorLBM
(``tensorlbm.multiphase.sc_two_component_force``, the force used by
``collide_sc_two_component`` at ``multiphase.py:788``) against three
references that do not share its code path:

* ``oss``      - NumPy transcription of the published forward-gather stencil
                 (Palabos ``shanChenProcessor3D.hh:304``, listLBM
                 ``mcmp_mod.f90``): an *implementation* cross-check.
* ``symbol``   - closed-form D2Q9 symbol of the same stencil: the discrete
                 operator's truncation is predictable, not tabulated.
* ``spectral`` - exact gradient of the sampled field: the *continuum* reference
                 that the acceptance criterion (<15%) is judged against.

Arms (all numbers land in ``artifacts/summary.json``; nothing is retyped):

A. ``mode_sweep``    single Fourier modes, constant companion density; measured
                     relative L2 error vs the closed-form ``|1 - S(k)|``.
B. ``diagonal``      same sweep along the (1,1) lattice direction (isotropy of
                     the symbol).
C. ``slab``          real solver state: relaxed two-component planar interface
                     (n = 64/96, G_12 = -2.5/-3.0) compared with the spectral
                     reference on the *simulated* densities, for both force
                     discretisations (``standard`` and ``edm``).
D. ``momentum``      action-reaction defect ``|sum(F_1+F_2)| / sum(|F_1|+|F_2|)``
                     on non-uniform fields: the ``edm`` split is a pure discrete
                     gradient and must telescope to zero.
E. ``invariants``    uniform-field nil, D2Q9 symmetry equivariance,
                     OSS-stencil identity and symbol identity.
F. ``mutation``      planted defects in the reference implementation, to show
                     the metric is sensitive (a test that cannot fail is not a
                     test).

Usage::

    PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xval_lib as L  # noqa: E402

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.mcmp_verification import simulate_planar_slab  # noqa: E402
from tensorlbm.multiphase import sc_two_component_force  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)

ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Predeclared thresholds (copied verbatim into predeclared.json before the run;
# not adjusted after seeing results).
# ---------------------------------------------------------------------------
ACCEPT_PCT = 15.0
THRESHOLDS = {
    "accept_pct": ACCEPT_PCT,               # owner accuracy standard, XVAL ledger
    "oss_identity_max": 1e-6,               # kernel == OSS forward-gather stencil (float32 weight table floor)
    "symbol_identity_max": 1e-6,            # measured error == closed-form |1-S(k)| (float32 floor)
    "uniform_nil_max": 1e-14,               # no spurious body force on a uniform field
    "symmetry_equivariance_max": 1e-12,     # kernel commutes with D2Q9 symmetry group
    "momentum_edm_max": 1e-13,              # edm net total force telescopes to zero
    "mutation_min": ACCEPT_PCT,             # every planted defect must exceed the gate
    "resolved_min_cells_per_wavelength": 8.0,  # declared interface resolution floor
}


def momentum_defect(fx1, fy1, fx2, fy2) -> float:
    """Action-reaction defect: |sum(F_1+F_2)| / sum(|F_1|+|F_2|) over the domain."""
    net = abs(float((fx1 + fx2).sum())) + abs(float((fy1 + fy2).sum()))
    scale = float(np.abs(fx1).sum() + np.abs(fy1).sum() + np.abs(fx2).sum() + np.abs(fy2).sum())
    return net / scale if scale > 0 else float("nan")


def rel_l2(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    a = np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=np.float64)
    b = np.asarray(b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b, dtype=np.float64)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / den if den > 0 else float("nan")


def kernel_force(rho1: np.ndarray, rho2: np.ndarray, G12: float, sc_form: str = "standard"):
    t1 = torch.from_numpy(np.ascontiguousarray(rho1, dtype=np.float64))
    t2 = torch.from_numpy(np.ascontiguousarray(rho2, dtype=np.float64))
    fx1, fy1, fx2, fy2 = sc_two_component_force(t1, t2, G_12=G12, sc_form=sc_form)
    return (
        fx1.numpy().astype(np.float64),
        fy1.numpy().astype(np.float64),
        fx2.numpy().astype(np.float64),
        fy2.numpy().astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Arm A/B: single-mode sweep vs closed-form symbol
# ---------------------------------------------------------------------------
def arm_mode_sweep(n: int = 64) -> dict:
    rows = []
    for direction, label in ((np.array([1.0, 0.0]), "axis"), (np.array([1.0, 1.0]), "diagonal")):
        for m in range(1, n // 2 + 1):
            kvec = 2.0 * math.pi * m / n * direction
            kmag = float(np.linalg.norm(kvec))
            if kmag >= math.pi - 1e-12:
                continue  # Nyquist: the sampled mode is uniform, nothing to test
            rho2 = L.mode_field(n, (float(kvec[0]), float(kvec[1])))
            rho1 = np.full_like(rho2, 0.5)
            fx1, fy1, _, _ = kernel_force(rho1, rho2, -2.5)
            rx, ry = L.spectral_force(rho1, rho2, -2.5)
            got = float(np.hypot(fx1, fy1).sum())
            ref = float(np.hypot(rx, ry).sum())
            ratio = got / ref if ref > 0 else float("nan")
            rows.append(
                {
                    "direction": label,
                    "m": m,
                    "k": kmag,
                    "wavelength_cells": 2.0 * math.pi / kmag,
                    "ratio_measured": ratio,
                    "ratio_symbol": L.symbol_ratio((float(kvec[0]), float(kvec[1]))),
                    "err_vs_spectral": rel_l2(np.stack([fx1, fy1]), np.stack([rx, ry])),
                    "err_closed_form": abs(1.0 - L.symbol_ratio((float(kvec[0]), float(kvec[1])))),
                }
            )
    boundary = _first_breach(rows, "axis")
    return {
        "n": n,
        "rows": rows,
        "symbol_identity_max": max(abs(r["ratio_measured"] - r["ratio_symbol"]) for r in rows),
        "closed_form_identity_max": max(
            abs(r["err_vs_spectral"] - r["err_closed_form"]) for r in rows
        ),
        "worst_err_axis_resolved": max(
            r["err_vs_spectral"]
            for r in rows
            if r["direction"] == "axis"
            and r["wavelength_cells"] >= THRESHOLDS["resolved_min_cells_per_wavelength"]
        ),
        "worst_err_diagonal_resolved": max(
            r["err_vs_spectral"]
            for r in rows
            if r["direction"] == "diagonal"
            and r["wavelength_cells"] >= THRESHOLDS["resolved_min_cells_per_wavelength"]
        ),
        "boundary": boundary,
    }


def _first_breach(rows: list[dict], direction: str) -> dict:
    """Coarsest resolved wavelength that still meets the gate, and where it breaks."""
    sel = sorted(
        (r for r in rows if r["direction"] == direction), key=lambda r: r["wavelength_cells"]
    )
    passing = [r for r in sel if r["err_vs_spectral"] < ACCEPT_PCT / 100.0]
    failing = [r for r in sel if r["err_vs_spectral"] >= ACCEPT_PCT / 100.0]
    return {
        "min_passing_wavelength_cells": min((r["wavelength_cells"] for r in passing), default=None),
        "first_failing_wavelength_cells": min((r["wavelength_cells"] for r in failing), default=None),
        "first_failing_err": min((r["err_vs_spectral"] for r in failing), default=None),
        "first_failing_k": min((r["k"] for r in failing), default=None),
    }


def arm_symbol_refinement() -> dict:
    """Refinement study: same physical harmonic, resolution doubled (order check)."""
    rows = []
    for n in (32, 64, 128, 256):
        kvec = (2.0 * math.pi / n, 0.0)
        rho2 = L.mode_field(n, kvec)
        rho1 = np.full_like(rho2, 0.5)
        fx1, fy1, _, _ = kernel_force(rho1, rho2, -2.5)
        rx, ry = L.spectral_force(rho1, rho2, -2.5)
        rows.append(
            {
                "n": n,
                "k": float(kvec[0]),
                "err": rel_l2(np.stack([fx1, fy1]), np.stack([rx, ry])),
                "err_closed_form": abs(1.0 - L.symbol_ratio(kvec)),
            }
        )
    orders = []
    for a, b in zip(rows, rows[1:]):
        if a["err"] > 0 and b["err"] > 0:
            orders.append(math.log(a["err"] / b["err"]) / math.log(b["n"] / a["n"]))
    return {"rows": rows, "observed_orders": orders, "mean_order": float(np.mean(orders))}


# ---------------------------------------------------------------------------
# Arm C: real solver state (relaxed planar interface)
# ---------------------------------------------------------------------------
def arm_slab(cases: list[dict]) -> dict:
    out = []
    cache = ART / "slab_cases.jsonl"
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    done = {(c["G12"], c["n"]) for c in out}
    cases = [c for c in cases if (c["G12"], c["n"]) not in done]
    for case in cases:
        t0 = time.time()
        run = simulate_planar_slab(
            G_12=case["G12"],
            n=case["n"],
            steps=case["steps"],
            stop_tol=1e-4,
            check_every=250,
            device="cpu",
        )
        rho1 = run["rho1"].numpy().astype(np.float64)
        rho2 = run["rho2"].numpy().astype(np.float64)
        row = {
            "G12": case["G12"],
            "n": case["n"],
            "steps_run": int(run["steps_run"]),
            "converged": bool(run["converged"]),
            "u_max": float(run["u_max"]),
            "rho1_range": [float(rho1.min()), float(rho1.max())],
            "rho2_range": [float(rho2.min()), float(rho2.max())],
            "wall_time_s": round(time.time() - t0, 2),
        }
        rx, ry = L.spectral_force(rho1, rho2, case["G12"])
        for sc_form in ("standard", "edm"):
            fx1, fy1, fx2, fy2 = kernel_force(rho1, rho2, case["G12"], sc_form=sc_form)
            row[sc_form] = {
                "err_vs_spectral": rel_l2(np.stack([fx1, fy1]), np.stack([rx, ry])),
                "momentum_defect_rel": momentum_defect(fx1, fy1, fx2, fy2),
                "force_scale": float(np.abs(fx1).max() + np.abs(fy1).max()),
            }
            if sc_form == "standard":
                row["profile_y"] = rho1.mean(axis=1).tolist()
                row["force_profile_y"] = fy1.mean(axis=1).tolist()
                row["ref_profile_y"] = ry.mean(axis=1).tolist()
        row["spectral_force_scale"] = float(np.abs(rx).max() + np.abs(ry).max())
        with cache.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        out.append(row)
    return {"cases": out, "worst_err": max(max(c["standard"]["err_vs_spectral"], c["edm"]["err_vs_spectral"]) for c in out)}


# ---------------------------------------------------------------------------
# Arm D: action-reaction (momentum) defect
# ---------------------------------------------------------------------------
def arm_momentum(n: int = 64) -> dict:
    rho1, rho2 = L.sloshing_fields(n)
    rows = {}
    for sc_form in ("standard", "edm"):
        fx1, fy1, fx2, fy2 = kernel_force(rho1, rho2, -2.5, sc_form=sc_form)
        net = abs(float((fx1 + fx2).sum())) + abs(float((fy1 + fy2).sum()))
        scale = float(np.abs(fx1).sum() + np.abs(fy1).sum() + np.abs(fx2).sum() + np.abs(fy2).sum())
        rows[sc_form] = {
            "net_total_force": float(net),
            "force_sum": float(scale),
            "defect_rel": float(net / scale),
            "sum_fx_total": float((fx1 + fx2).sum()),
            "sum_fy_total": float((fy1 + fy2).sum()),
            # local (pointwise) non-cancellation of the pair: what the edm split removes
            "local_defect": float(
                (np.abs(fx1 + fx2) + np.abs(fy1 + fy2)).mean()
                / (np.abs(fx1) + np.abs(fy1) + np.abs(fx2) + np.abs(fy2)).mean()
            ),
            "local_defect_max": float(
                np.max(np.hypot(fx1 + fx2, fy1 + fy2)) / np.max(np.hypot(fx1, fy1) + np.hypot(fx2, fy2))
            ),
        }
    # discrete nil for a pure gradient field: rho1 = rho2 -> antisymmetric part vanishes
    return rows


# ---------------------------------------------------------------------------
# Arm E: invariants
# ---------------------------------------------------------------------------
def arm_invariants(n: int = 64) -> dict:
    res = {}

    # (a) uniform field: the interaction force must be exactly zero
    uni = np.full((n, n), 0.7)
    fx1, fy1, fx2, fy2 = kernel_force(uni, uni, -2.5)
    res["uniform_nil"] = float(max(np.abs(fx1).max(), np.abs(fy1).max(), np.abs(fx2).max(), np.abs(fy2).max()))

    # (b) OSS forward-gather identity with g_oss = -G_12
    rho1, rho2 = L.sloshing_fields(n)
    G12 = -2.5
    kx, ky = kernel_force(rho1, rho2, G12)[:2]
    ox, oy = L.oss_force(rho1, rho2, g_oss=-G12)
    res["oss_identity"] = max(rel_l2(kx, ox), rel_l2(ky, oy))
    # also prove the *unmapped* choice (g = +G_12) is wrong, so the map is not free
    bx, by = L.oss_force(rho1, rho2, g_oss=G12)
    res["oss_identity_bad_map"] = max(rel_l2(kx, bx), rel_l2(ky, by))

    # (c) D2Q9 symmetry equivariance: rotate the field 90 deg, force must rotate
    fx, fy = kx.copy(), ky.copy()
    f2x, f2y = kernel_force(np.rot90(rho1), np.rot90(rho2), G12)[:2]
    res["symmetry_rot90"] = max(
        rel_l2(f2x, np.rot90(fy, 1)),
        rel_l2(f2y, -np.rot90(fx, 1)),
    )
    f3x, f3y = kernel_force(rho1.T, rho2.T, G12)[:2]
    res["symmetry_transpose"] = max(rel_l2(f3x, fy.T), rel_l2(f3y, fx.T))

    # (d) spectral reference nil for a uniform field (guards the reference itself)
    spx, spy = L.spectral_force(uni, uni, G12)
    res["spectral_nil_uniform"] = float(max(np.abs(spx).max(), np.abs(spy).max()))

    # (e) symbol identity on one mode
    kv = (2.0 * math.pi * 4 / n, 0.0)
    m1 = np.full((n, n), 0.5)
    m2 = L.mode_field(n, kv)
    a = kernel_force(m1, m2, G12)[0]
    b = L.spectral_force(m1, m2, G12)[0]
    # a domain sum of a single-mode force vanishes; compare amplitudes instead
    res["symbol_identity"] = abs(float(np.linalg.norm(a) / np.linalg.norm(b)) - L.symbol_ratio(kv))
    return res


# ---------------------------------------------------------------------------
# Arm F: mutation controls (is the metric able to fail?)
# ---------------------------------------------------------------------------
def arm_mutation(n: int = 64) -> dict:
    rho1, rho2 = L.sloshing_fields(n)
    G12 = -2.5
    kv = (2.0 * math.pi * 4 / n, 0.0)
    m2 = L.mode_field(n, kv)
    m1 = np.full((n, n), 0.5)
    fx1, fy1, _, _ = kernel_force(m1, m2, G12)
    rx, ry = L.spectral_force(m1, m2, G12)
    out = {}

    # M1 sign flip (equivalent to dropping the -G12 sign convention)
    mx, my = L.oss_force(m1, m2, g_oss=G12)
    out["M1_sign_flip"] = rel_l2(np.stack([mx, my]), np.stack([rx, ry]))

    # M2 axis-only (D2Q5-like) stencil: diagonal weights dropped
    w = L.W.copy()
    w[:] = 0.0
    w[0] = 4.0 / 9.0
    w[1:5] = 1.0 / 9.0
    mx, my = L.oss_force(m1, m2, g_oss=-G12, weights=w)
    out["M2_diagonals_dropped"] = rel_l2(np.stack([mx, my]), np.stack([rx, ry]))

    # M3 missing cs^2 normalisation (force returned 3x too large)
    mx, my = L.oss_force(m1, m2, g_oss=-G12)
    out["M3_missing_cs2"] = rel_l2(np.stack([3.0 * mx, 3.0 * my]), np.stack([rx, ry]))

    # M4 reverse/forward gather confusion (field shifted the wrong way)
    acc = np.zeros(m2.shape + (2,))
    for wi, ci in zip(L.W, L.C):
        acc += wi * ci * L._shift(m2, -ci[0], -ci[1])[..., None]
    acc *= -(-G12) * m1[..., None]
    out["M4_wrong_gather_direction"] = rel_l2(np.stack([acc[..., 0], acc[..., 1]]), np.stack([rx, ry]))

    # M5 wrong pseudopotential (psi = sqrt(rho) instead of linear)
    mx, my = L.oss_force(np.sqrt(m1), np.sqrt(m2), g_oss=-G12)
    out["M5_psi_sqrt"] = rel_l2(np.stack([mx, my]), np.stack([rx, ry]))

    # M6 mis-normalised weights (1/4 for the axes, total 1.25)
    w = L.W.copy()
    w[1:5] = 1.0 / 4.0
    mx, my = L.oss_force(m1, m2, g_oss=-G12, weights=w)
    out["M6_bad_weight_table"] = rel_l2(np.stack([mx, my]), np.stack([rx, ry]))

    worst = float(
        rel_l2(np.stack([fx1, fy1]), np.stack([rx, ry]))
    )
    return {
        "baseline_err": worst,
        "mutations": out,
        "min_mutation_err": min(out.values()),
        "amplification_min": min(out.values()) / worst if worst > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    summary: dict = {
        "anchor": "XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE",
        "mission_id": int(os.environ.get("MISSION_ID", "18101")),
        "repo": "TensorLBM",
        "thresholds": THRESHOLDS,
        "reference_objects": {
            "oss": "omalaspinas/palabos src/multiPhysics/shanChenProcessor3D.hh:304 (t_i isotropic weights, PsiIsRho) / sorush-khajepor/listLBM mcmp_mod.f90 - forward gather, g_oss = -G_12",
            "symbol": "closed form S(k) = sum_i w_i (c_i.khat) sin(k.c_i)/(cs^2|k|)",
            "spectral": "exact Fourier-symbol gradient of the sampled field: F = G_12 cs^2 rho1 grad rho2",
        },
        "contract": {
            "kernel": "tensorlbm.multiphase.sc_two_component_force (sc_form='standard'|'edm')",
            "solver_call_site": "src/tensorlbm/multiphase.py:788 (collide_sc_two_component)",
        },
        "kernel_version": {
            "commit": _git_head(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "dtype": "float64",
            "device": "cpu",
        },
    }

    print("[A] mode sweep ...", flush=True)
    summary["A_mode_sweep"] = arm_mode_sweep(64)
    print("[B] symbol refinement ...", flush=True)
    summary["B_refinement"] = arm_symbol_refinement()
    print("[C] planar slab (real solver state) ...", flush=True)
    summary["C_slab"] = arm_slab(
        [
            {"G12": -2.5, "n": 64, "steps": 4000},
            {"G12": -2.5, "n": 96, "steps": 4000},
            {"G12": -3.0, "n": 96, "steps": 5000},
        ]
    )
    print("[D] momentum defect ...", flush=True)
    summary["D_momentum"] = arm_momentum(64)
    print("[E] invariants ...", flush=True)
    summary["E_invariants"] = arm_invariants(64)
    print("[F] mutation controls ...", flush=True)
    summary["F_mutation"] = arm_mutation(64)

    summary["verdicts"] = _verdicts(summary)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False))
    np.savez_compressed(
        ART / "fields.npz",
        **{
            f"case{i}_profile_y": np.asarray(c["profile_y"], dtype=np.float64)
            for i, c in enumerate(summary["C_slab"]["cases"])
        },
        **{
            f"case{i}_force_profile_y": np.asarray(c["force_profile_y"], dtype=np.float64)
            for i, c in enumerate(summary["C_slab"]["cases"])
        },
        **{
            f"case{i}_ref_profile_y": np.asarray(c["ref_profile_y"], dtype=np.float64)
            for i, c in enumerate(summary["C_slab"]["cases"])
        },
    )
    print(json.dumps(summary["verdicts"], indent=2))
    print(f"wall {summary['wall_time_s']} s -> {ART/'summary.json'}")
    return summary


def _git_head() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _verdicts(s: dict) -> dict:
    T = s["thresholds"]
    acc = T["accept_pct"] / 100.0
    slab_worst = s["C_slab"]["worst_err"]
    a = s["A_mode_sweep"]
    e = s["E_invariants"]
    d = s["D_momentum"]
    f = s["F_mutation"]
    checks = {
        "primary_slab_err_le_15pct": slab_worst <= acc,
        "secondary_resolved_band_le_15pct": max(
            a["worst_err_axis_resolved"], a["worst_err_diagonal_resolved"]
        )
        <= acc,
        "oss_stencil_identity": e["oss_identity"] <= T["oss_identity_max"],
        "symbol_closed_form_identity": a["closed_form_identity_max"] <= T["symbol_identity_max"],
        "uniform_field_nil": e["uniform_nil"] <= T["uniform_nil_max"],
        "symmetry_equivariance": max(e["symmetry_rot90"], e["symmetry_transpose"])
        <= T["symmetry_equivariance_max"],
        "momentum_edm_exact": d["edm"]["defect_rel"] <= T["momentum_edm_max"],
        "mutation_sensitivity": f["min_mutation_err"] >= T["mutation_min"] / 100.0,
    }
    return {
        "checks": checks,
        "primary_value": slab_worst,
        "all_invariants_pass": all(v for k, v in checks.items() if not k.startswith("primary") and not k.startswith("secondary")),
        "gate_pass": all(checks.values()),
        "numbers": {
            "slab_err_standard_max": max(c["standard"]["err_vs_spectral"] for c in s["C_slab"]["cases"]),
            "slab_err_edm_max": max(c["edm"]["err_vs_spectral"] for c in s["C_slab"]["cases"]),
            "mode_worst_resolved_axis": a["worst_err_axis_resolved"],
            "mode_worst_resolved_diagonal": a["worst_err_diagonal_resolved"],
            "boundary": a["boundary"],
            "refinement_order": s["B_refinement"]["mean_order"],
            "momentum_standard": d["standard"]["defect_rel"],
            "momentum_edm": d["edm"]["defect_rel"],
            "mutation_min": f["min_mutation_err"],
            "mutation_amplification": f["amplification_min"],
        },
    }


def _cli() -> None:
    if "--slab-case" in sys.argv:
        n, g12, steps = sys.argv[sys.argv.index("--slab-case") + 1].split(":")
        res = arm_slab([{"n": int(n), "G12": float(g12), "steps": int(steps)}])
        print(json.dumps(res["cases"], indent=2))
        return
    main()


if __name__ == "__main__":
    _cli()