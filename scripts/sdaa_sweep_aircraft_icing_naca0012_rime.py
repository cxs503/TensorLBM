#!/usr/bin/env python3
"""SDAA validation sweep for benchmarks/verified/aircraft_icing_naca0012_rime.

Independent runner (does **not** import the case's ``run.py``, which hard-asserts
``torch.cuda.is_available()``).  The case configs, the 15-array IC-A determinism
protocol and the gate definitions below are taken verbatim from
``benchmarks/verified/aircraft_icing_naca0012_rime/{run.py,result.json,README.md}``
so the criteria口径 is unchanged; only the compute device changes (cuda -> sdaa).

Gates (same 口径 as the case result.json::gates):

  G1  Lagrangian mass closure, worst over runs            <= 1e-12
  G2  Eulerian closure (relative), worst over runs         <= 1e-6
  G7  bitwise determinism of det_pair, 15 arrays sha256     all equal
  G9a deposited vs LWC log-log slope                       in [0.97, 1.03]
  G9b proportional-fit R^2 (about the mean)                >= 0.999
  G9c proportional-fit max |dev|                           <= 3 %
  G9d deposited vs LWC strictly increasing                 true
  G10a t_max ladder L1 < L2 < L3                           strictly increasing
  G10b 3-point Richardson error at L3                      |err| <= 3 %
  G10c same f_ext evaluated at L4                          |err| <= 3 %
  G10d Richardson meta (p, f_ext) reproduces the IC-A prior  |rel| <= 1e-9

Every gate value is computed here from the runs of this script — no gate
number is copied from the NVIDIA-side evidence.  Only the *priors* used by
G10d (and the disclosure columns) are read from the case ``result.json``.

Usage:
  TRITON_INTERPRET=1 python scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py \
      [--budget-s 1100] [--out artifacts/sdaa_aircraft_icing_naca0012_rime.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.aircraft_icing import (  # noqa: E402
    IcingConfig,
    run_rime_icing,
    save_icing_artifacts,
)

CASE_DIR = REPO / "benchmarks" / "verified" / "aircraft_icing_naca0012_rime"
DEVICE = "sdaa"

# IC-A determinism protocol (verbatim from the case run.py).
DET_ARRAYS = (
    "airfoil",
    "solid",
    "m_w",
    "impact_mass",
    "s_grid",
    "beta_grid",
    "stag",
    "hist_step",
    "hist_t",
    "hist_cd",
    "hist_cl",
    "hist_ice",
    "alpha_e",
    "impact_mass_e",
    "beta_e_grid",
)

# (name, nx, ny, steps, warmup, lwc) — verbatim from the case run.py.
RUNS = (
    ("b1_prod", 320, 160, 3000, 6000, 5.0e-4),
    ("ladder_L1", 320, 160, 3000, 6000, 5.0e-4),
    ("ladder_L2", 640, 320, 6000, 12000, 5.0e-4),
    ("ladder_L3", 890, 445, 8334, 16600, 5.0e-4),
    ("ladder_L4", 1280, 640, 13340, 26600, 5.0e-4),
    ("m20_lwc025", 320, 160, 3000, 6000, 2.5e-4),
    ("m20_lwc10", 320, 160, 3000, 6000, 1.0e-3),
    ("det_run1", 320, 160, 3000, 6000, 5.0e-4),
    ("det_run2", 320, 160, 3000, 6000, 5.0e-4),
)

# Priority order under the wall-clock cap.  ``ladder_L1`` is *identical* in
# config to ``b1_prod`` (both 320x160/3000/6000, LWC 0.5 g/m^3), so b1_prod is
# reused as L1 when the ladder cannot be afforded; that reuse is recorded.
ORDER = (
    "b1_prod",
    "m20_lwc025",
    "m20_lwc10",
    "det_run1",
    "det_run2",
    "ladder_L2",
    "ladder_L3",
    "ladder_L4",
)

CFG_TABLE = {r[0]: r for r in RUNS}


def make_cfg(name: str) -> IcingConfig:
    """Case production-point config, only ``device`` changed to sdaa."""
    _, nx, ny, steps, warmup, lwc = CFG_TABLE[name]
    return IcingConfig(
        nx=nx,
        ny=ny,
        chord_frac=0.4,
        u_in=0.05,
        aoa_deg=4.0,
        steps=steps,
        warmup_steps=warmup,
        chord_phys=0.5334,
        v_inf=67.0,
        lwc=lwc,
        mvd=20.0e-6,
        t_static_c=-10.0,
        t_exposure=360.0,
        rime_density_mode="macklin",
        droplet_phase="eulerian",
        eulerian_scheme="donor2",
        collision="cumulant",
        c_s=0.1,
        re_lu_target=2.5e6,
        beta_window_mode="clean",
        drag_law="stokes",
        device=DEVICE,
        seed=0,
        log_every=500,
    )


def npz_array_hashes(npz_path: Path) -> list:
    """IC-A determinism protocol: per-array sha256 (b_common verbatim)."""
    z = np.load(npz_path)
    out = []
    for k in DET_ARRAYS:
        a = np.ascontiguousarray(z[k])
        out.append([k, hashlib.sha256(a.tobytes()).hexdigest()])
    return out


def stag_thickness_mm(result: dict, cfg: IcingConfig) -> float:
    ice = result["ice_only"]
    x_le = int(result["metrics"]["x_le"])
    return float(ice[:, x_le].sum()) * cfg.dx_phys * 1e3


def device_evidence() -> dict:
    ev = {
        "compute_device": DEVICE,
        "torch_version": torch.__version__,
        "torch_sdaa_module": hasattr(torch, "sdaa"),
        "triton_interpret_env": os.environ.get("TRITON_INTERPRET"),
        "triton_imported": "triton" in sys.modules,
        "compile_mode": None,
        "note_compile": (
            "case run.py uses compile_mode=None (plain torch ops); the compiled "
            "variants are never built, so no triton kernel path is entered and "
            "TRITON_INTERPRET=1 is a belt-and-braces guard only"
        ),
    }
    if hasattr(torch, "sdaa"):
        ev["sdaa_is_available"] = bool(torch.sdaa.is_available())
        ev["sdaa_device_count"] = int(torch.sdaa.device_count())
        ev["sdaa_device_name"] = torch.sdaa.get_device_name(0)
    # Hard proof that tensors live on the device: allocate/matmul there.
    a = torch.randn(64, 64, device=DEVICE)
    b = (a @ a).sum().item()
    ev["sdaa_smoke_matmul_ok"] = bool(np.isfinite(b))
    ev["sdaa_memory_allocated_bytes_after_smoke"] = int(
        torch.sdaa.memory_allocated(0)
    )
    return ev


def run_one(name: str, out_root: Path, log_sink: list) -> dict:
    cfg = make_cfg(name)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{name}] {msg}", flush=True)
        log_sink.append(f"{name}: {msg}")

    if hasattr(torch, "sdaa"):
        torch.sdaa.synchronize()
    torch.sdaa.reset_peak_memory_stats(0)
    t0 = time.time()
    r = run_rime_icing(cfg, log=log)
    torch.sdaa.synchronize()
    wall = time.time() - t0

    files = save_icing_artifacts(r, str(out_dir))
    e, m, aud = r["eulerian"], r["metrics"], r["audit"]
    row = {
        "name": name,
        "grid": [cfg.nx, cfg.ny],
        "steps": cfg.steps,
        "warmup_steps": cfg.warmup_steps,
        "lwc": cfg.lwc,
        "wall_s": wall,
        "device": DEVICE,
        "device_peak_mem_bytes": int(torch.sdaa.max_memory_allocated(0)),
        "beta_e_pk": float(e["beta"]["beta"].max()) if len(e["beta"]["beta"]) else None,
        "euler_deposited_kg": float(e["audit"]["deposited"]),
        "audit_lagr": {k: aud[k] for k in ("seeded", "frozen", "exited", "trapped",
                                          "airborne", "pending", "closure_error")},
        "audit_euler": e["audit"],
        "n_ice_cells": int(m["n_ice_cells"]),
        "ice_area_pct_chord2": float(m["ice_area_pct_chord2"]),
        "t_max_mm": float(1e3 * max(m.get("upper_horn_m", 0.0), m.get("lower_horn_m", 0.0))),
        "upper_horn_mm": float(1e3 * m.get("upper_horn_m", 0.0)),
        "lower_horn_mm": float(1e3 * m.get("lower_horn_m", 0.0)),
        "t_stag_mm": stag_thickness_mm(r, cfg),
        "cd0": r["cd0"],
        "cl0": r["cl0"],
        "cd_end": r["cd_end"],
        "cl_end": r["cl_end"],
        "cd_drift_pct": r["cd_drift_pct"],
        "array_sha256": npz_array_hashes(out_dir / "result.npz"),
        "artifacts_dir": str(out_dir),
        "artifacts": files,
    }
    # prereg §4 guards (verbatim from case run.py)
    guard = {}
    ok = True
    for k in ("m_w", "impact_mass"):
        a = np.asarray(r[k], dtype=np.float64)
        f = bool(np.isfinite(a).all())
        guard[f"{k}_finite"] = f
        ok &= f
    guard["deposited_positive"] = float(e["audit"]["deposited"]) > 0.0
    guard["ice_nonzero"] = int(np.asarray(r["ice_only"]).sum()) > 0
    guard["wall_under_3600s"] = wall < 3600.0
    ok &= guard["deposited_positive"] and guard["ice_nonzero"] and guard["wall_under_3600s"]
    row["guards"] = guard
    row["guards_pass"] = bool(ok)
    print(
        f"[sdaa] {name} DONE wall={wall:.1f}s t_max={row['t_max_mm']:.6f}mm "
        f"dep={row['euler_deposited_kg']:.6e} guards={ok}",
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# Gate arithmetic (identical 口径 to the case result.json::gates)
# ---------------------------------------------------------------------------
def gate_g1_g2(rows: dict) -> tuple:
    lag = [r["audit_lagr"]["closure_error"] for r in rows.values()]
    eul = [
        r["audit_euler"].get("closure_error")
        if isinstance(r["audit_euler"], dict)
        else None
        for r in rows.values()
    ]
    eul = [float(v) for v in eul if v is not None]
    g1 = max(lag) if lag else None
    g2 = max(eul) if eul else None
    return g1, g2


def gate_g7(rows: dict) -> dict:
    h1 = dict((k, v) for k, v in rows["det_run1"]["array_sha256"])
    h2 = dict((k, v) for k, v in rows["det_run2"]["array_sha256"])
    keys = sorted(set(h1) | set(h2))
    equal = [k for k in keys if h1.get(k) == h2.get(k)]
    return {
        "value": f"{len(equal)}/{len(keys)} arrays sha256 equal",
        "n_equal": len(equal),
        "n_total": len(keys),
        "mismatched": [k for k in keys if h1.get(k) != h2.get(k)],
        "gate": "all arrays bitwise identical",
        "pass": len(equal) == len(keys) == len(DET_ARRAYS),
    }


def gate_g9(rows: dict) -> dict:
    names = ("m20_lwc025", "b1_prod", "m20_lwc10")
    x = np.array([rows[n]["lwc"] for n in names], dtype=np.float64)
    y = np.array([rows[n]["euler_deposited_kg"] for n in names], dtype=np.float64)
    slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
    k = float((x * y).sum() / (x * x).sum())  # proportional fit through origin
    yp = k * x
    r2 = float(1.0 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    dev_pct = np.abs((y - yp) / yp) * 100.0
    mono = bool(np.all(np.diff(y) > 0.0))
    return {
        "lwc": x.tolist(),
        "deposited_kg": y.tolist(),
        "g9a": {"value": slope, "gate": "[0.97,1.03]", "pass": 0.97 <= slope <= 1.03},
        "g9b": {"value": r2, "gate": ">=0.999", "pass": r2 >= 0.999},
        "g9c": {
            "value": float(dev_pct.max()),
            "per_point_pct": dev_pct.tolist(),
            "gate": "<=3",
            "pass": float(dev_pct.max()) <= 3.0,
        },
        "g9d": {"value": mono, "gate": "strictly increasing", "pass": mono},
        "prop_fit_k_kg_per_unit_lwc": k,
    }


def richardson_nonuniform(levels: list) -> dict:
    """3-point Richardson: t(dx) = f_ext - a*dx^p  (analytic, no gate literals).

    Eliminates a between the three levels to get a single scalar equation in p
    (brentq), then back-substitutes a and f_ext.
    """
    from scipy.optimize import brentq

    levels = sorted(levels, key=lambda lv: lv[0])  # by nx ascending -> dx descending
    nx = np.array([lv[0] for lv in levels], dtype=np.float64)
    t = np.array([lv[1] for lv in levels], dtype=np.float64)
    dx = 1.0 / nx  # dx ∝ 1/nx (chord_frac fixed); f_ext and p invariant to this scale
    k = (t[0] - t[2]) / (t[1] - t[2])
    g = lambda p: (dx[2] ** p - dx[0] ** p) / (dx[2] ** p - dx[1] ** p) - k  # noqa: E731
    p = brentq(g, 1.0, 30.0, xtol=1e-15, rtol=1e-15)
    a = (t[1] - t[2]) / (dx[2] ** p - dx[1] ** p)
    f_ext = float(t[2] + a * dx[2] ** p)
    return {"p": float(p), "a": float(a), "f_ext_mm": f_ext, "nx": nx.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=1150.0,
                    help="wall cap for the *simulation* runs (task cap 1200 s)")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sdaa_aircraft_icing_naca0012_rime.json"))
    ap.add_argument("--out-root", default=str(REPO / "artifacts" / "sdaa_runs" / "aircraft_icing_naca0012_rime"))
    ap.add_argument("--max-runs", type=int, default=len(ORDER))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    dev_ev = device_evidence()
    print(json.dumps(dev_ev, indent=2, default=float), flush=True)

    rows: dict = {}
    run_log: list = []
    skipped: list = []
    per_run_est = None
    for i, name in enumerate(ORDER[: args.max_runs]):
        spent = time.time() - t_start
        # budget guard, using the measured cost of the previous run of same grid
        if per_run_est is not None and spent + per_run_est > args.budget_s:
            skipped.append({"name": name, "reason": "wall budget",
                            "spent_s": spent, "est_next_s": per_run_est})
            print(f"[sdaa] SKIP {name}: budget spent={spent:.0f}s est={per_run_est:.0f}s",
                          flush=True)
            continue
        try:
            row = run_one(name, out_root, run_log)
        except Exception as ex:  # honest failure, keep sweeping the rest
            import traceback
            traceback.print_exc()
            skipped.append({"name": name, "reason": f"EXCEPTION {type(ex).__name__}: {ex}",
                            "spent_s": time.time() - t_start})
            continue
        rows[name] = row
        per_run_est = row["wall_s"]
    sweep_wall = time.time() - t_start

    # ladder_L1 reuse (identical config to b1_prod)
    l1_source = None
    if "ladder_L1" not in rows and "b1_prod" in rows:
        rows["ladder_L1"] = dict(rows["b1_prod"])
        rows["ladder_L1"]["name"] = "ladder_L1"
        rows["ladder_L1"]["reused_from"] = "b1_prod"
        l1_source = "reused b1_prod (identical config: 320x160, 3000/6000 steps, LWC 0.5 g/m^3)"

    # ---- priors from the case result.json (reference only, never gate values)
    case = json.loads((CASE_DIR / "result.json").read_text())
    prior_gate = case["gates"]

    gates: dict = {}

    def add_gate(key, value, gate, passed, note=None, prior=None):
        g = {"value": value, "gate": gate, "pass": bool(passed), "source": "this script"}
        if note:
            g["note"] = note
        if prior is not None:
            g["prior_nvidia"] = prior
        gates[key] = g

    # G1 / G2
    g1, g2 = gate_g1_g2(rows)
    if g1 is None:
        add_gate("G1_lagrangian_closure", None, "<=1e-12", False,
                 "NOT_RUN: no run completed")
    else:
        add_gate("G1_lagrangian_closure", g1, "<=1e-12", g1 <= 1e-12,
                 f"worst over {len(rows)} run(s): {sorted(rows)}",
                 prior_gate["G1_lagrangian_closure"]["prior"])
    if g2 is None:
        add_gate("G2_eulerian_closure_rel", None, "<=1e-6", False,
                 "NOT_RUN: no run completed")
    else:
        add_gate("G2_eulerian_closure_rel", g2, "<=1e-6", g2 <= 1e-6,
                 f"worst over {len(rows)} run(s): {sorted(rows)}",
                 prior_gate["G2_eulerian_closure_rel"]["prior"])

    # G7
    if "det_run1" in rows and "det_run2" in rows:
        g7 = gate_g7(rows)
        add_gate("G7_determinism_bitwise", g7["value"], g7["gate"], g7["pass"],
                 f"mismatched={g7['mismatched']}", prior_gate["G7_determinism_bitwise"]["prior"])
        gates["G7_determinism_bitwise"]["n_equal"] = g7["n_equal"]
        gates["G7_determinism_bitwise"]["n_total"] = g7["n_total"]
    else:
        add_gate("G7_determinism_bitwise", None, "all arrays bitwise identical", False,
                 "NOT_RUN: det_pair (det_run1, det_run2) not both executed",
                 prior_gate["G7_determinism_bitwise"]["prior"])

    # G9
    if all(n in rows for n in ("m20_lwc025", "b1_prod", "m20_lwc10")):
        g9 = gate_g9(rows)
        add_gate("G9a_deposited_loglog", g9["g9a"]["value"], g9["g9a"]["gate"], g9["g9a"]["pass"],
                 "deposited vs LWC (0.25/0.5/1.0 g/m^3)",
                 prior_gate["G9a_deposited_loglog"]["prior"])
        add_gate("G9b_deposited_r2", g9["g9b"]["value"], g9["g9b"]["gate"], g9["g9b"]["pass"],
                 f"prop fit k={g9['prop_fit_k_kg_per_unit_lwc']:.6e} kg per (kg/m^3)",
                 prior_gate["G9b_deposited_r2"]["prior"])
        add_gate("G9c_deposited_prop_dev_pct", g9["g9c"]["value"], g9["g9c"]["gate"],
                 g9["g9c"]["pass"], f"per point pct={g9['g9c']['per_point_pct']}",
                 prior_gate["G9c_deposited_prop_dev_pct"]["prior"])
        add_gate("G9d_deposited_monotone", g9["g9d"]["value"], g9["g9d"]["gate"], g9["g9d"]["pass"],
                 "deposited vs LWC", None)
        gates["_g9_series"] = {"lwc": g9["lwc"], "deposited_kg": g9["deposited_kg"]}
    else:
        for k in ("G9a_deposited_loglog", "G9b_deposited_r2",
                  "G9c_deposited_prop_dev_pct", "G9d_deposited_monotone"):
            add_gate(k, None, "see README", False,
                     "NOT_RUN: LWC family (m20_lwc025, b1_prod, m20_lwc10) incomplete")

    # G10
    lad = [n for n in ("ladder_L1", "ladder_L2", "ladder_L3", "ladder_L4") if n in rows]
    if all(n in rows for n in ("ladder_L1", "ladder_L2", "ladder_L3")):
        t = [rows[n]["t_max_mm"] for n in ("ladder_L1", "ladder_L2", "ladder_L3")]
        mono = bool(t[0] < t[1] < t[2])
        add_gate("G10a_monotone_L1_L2_L3", t, "t(L1)<t(L2)<t(L3) strict", mono,
                 "t_max_mm", prior_gate["G10a_monotone_L1_L2_L3"]["prior"])
        rich = richardson_nonuniform([
            (rows["ladder_L1"]["grid"][0], t[0]),
            (rows["ladder_L2"]["grid"][0], t[1]),
            (rows["ladder_L3"]["grid"][0], t[2]),
        ])
        err3 = (t[2] - rich["f_ext_mm"]) / rich["f_ext_mm"] * 100.0
        add_gate("G10b_richardson_L3_pct", err3, "|err|<=3%", abs(err3) <= 3.0,
                 f"f_ext={rich['f_ext_mm']:.9f} mm",
                 prior_gate["G10b_richardson_L3_pct"]["prior"])
        if "ladder_L4" in rows:
            t4 = rows["ladder_L4"]["t_max_mm"]
            err4 = (t4 - rich["f_ext_mm"]) / rich["f_ext_mm"] * 100.0
            add_gate("G10c_richardson_L4_pct", err4, "|err|<=3%", abs(err4) <= 3.0,
                     f"sampled f_ext from L1..L3, t_max(L4)={t4!r}",
                     prior_gate["G10c_richardson_L4_pct"]["prior"])
        else:
            add_gate("G10c_richardson_L4_pct", None, "|err|<=3%", False,
                     "NOT_RUN: ladder_L4 not executed",
                     prior_gate["G10c_richardson_L4_pct"]["prior"])
        pr = prior_gate["G10d_richardson_meta"]["prior"]
        rel_p = abs(rich["p"] - pr["p"]) / abs(pr["p"])
        rel_f = abs(rich["f_ext_mm"] - pr["f_ext_mm"]) / abs(pr["f_ext_mm"])
        okd = rel_p <= 1e-9 and rel_f <= 1e-9
        add_gate("G10d_richardson_meta", {"p": rich["p"], "f_ext_mm": rich["f_ext_mm"]},
                 "|p-prior|,|f_ext-prior| <=1e-9 rel", okd,
                 f"rel_p={rel_p:.3e} rel_f_ext={rel_f:.3e}", pr)
    else:
        for k in ("G10a_monotone_L1_L2_L3", "G10b_richardson_L3_pct",
                  "G10c_richardson_L4_pct", "G10d_richardson_meta"):
            add_gate(k, None, "see README", False,
                     f"NOT_RUN: ladder incomplete (completed: {lad})")

    # ---- disclosure: SDAA reproduction of the NVIDIA production point/pair
    repro = {}
    if "b1_prod" in rows:
        r0 = rows["b1_prod"]
        pr = case["runs"]["b1_prod"]
        repro["b1_prod"] = {
            "sdaa": {"t_max_mm": r0["t_max_mm"], "euler_deposited_kg": r0["euler_deposited_kg"],
                     "beta_e_pk": r0["beta_e_pk"]},
            "nvidia_prior": pr,
            "rel_dev": {
                "t_max": (r0["t_max_mm"] - pr["t_max_mm"]) / pr["t_max_mm"],
                "euler_deposited": (r0["euler_deposited_kg"] - pr["euler_deposited_kg"])
                / pr["euler_deposited_kg"],
                "beta_pk": (r0["beta_e_pk"] - pr["beta_pk"]) / pr["beta_pk"],
            },
        }
    for nm in ("m20_lwc025", "m20_lwc10"):
        if nm in rows:
            pr = case["runs"][nm]
            repro[nm] = {
                "sdaa": {"t_max_mm": rows[nm]["t_max_mm"],
                         "euler_deposited_kg": rows[nm]["euler_deposited_kg"]},
                "nvidia_prior": {"t_max_mm": pr["t_max_mm"],
                                 "euler_deposited_kg": pr["euler_deposited_kg"]},
            }

    gate_vals = {k: v for k, v in gates.items() if not k.startswith("_")}
    all_pass = all(g["pass"] for g in gate_vals.values()) and not skipped
    n_pass = sum(1 for g in gate_vals.values() if g["pass"])

    result = {
        "case": "aircraft_icing_naca0012_rime",
        "kind": "SDAA validation sweep (independent runner)",
        "script": "scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py",
        "case_dir": str(CASE_DIR.relative_to(REPO)),
        "criterion_口径": (
            "case README/result.json gate family (internal consistency): "
            "G1 Lagrangian closure, G2 Eulerian closure, G7 IC-A bitwise "
            "determinism, G9 LWC linearity (log-log slope / R^2 / dev / "
            "monotone), G10 mesh ladder + 3-point Richardson vs IC-A prior"
        ),
        "device_evidence": dev_ev,
        "device_evidence_line": (
            f"torch.sdaa.get_device_name(0) = {dev_ev.get('sdaa_device_name')!r} "
            f"(sdaa available={dev_ev.get('sdaa_is_available')}, "
            f"count={dev_ev.get('sdaa_device_count')}, "
            f"torch={dev_ev['torch_version']}, compute device='{DEVICE}', "
            f"cuda used=False)"
        ),
        "triton": {
            "required": False,
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET"),
            "triton_imported_at_end": "triton" in sys.modules,
            "bypass_note": (
                "entry imports no triton and compile_mode=None keeps both "
                "compile_unit variants un-built; TRITON_INTERPRET=1 set in env "
                "as a guard and recorded here, nothing was interpreted"
            ),
        },
        "wall_clock": {
            "budget_s": args.budget_s,
            "sweep_wall_s": sweep_wall,
            "per_run_s": {k: v["wall_s"] for k, v in rows.items()
                          if not v.get("reused_from")},
            "skipped": skipped,
            "note": "task cap 1200 s (20 min) for the whole sweep",
        },
        "runs": rows,
        "run_order": list(rows.keys()),
        "skipped_runs": skipped,
        "ladder_L1_source": l1_source,
        "gates": gate_vals,
        "gates_summary": {"n_pass": n_pass, "n_total": len(gate_vals),
                          "all_pass": bool(all_pass)},
        "reproduction_vs_nvidia_prior": repro,
        "verdict": {
            "overall": "PASS" if all_pass else "FAIL",
            "reason": ("all gates PASS on sdaa with the full 9-run matrix"
                       if all_pass else
                       "honest-FAIL: not every gate could be evaluated/PASS on "
                       "sdaa within the wall cap; see gates[*].note and "
                       "skipped_runs for the exact reasons"),
        },
        "log_tail": run_log[-40:],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[sdaa] wrote {args.out}")
    print(f"[sdaa] gates {n_pass}/{len(gate_vals)} PASS  overall="
          f"{result['verdict']['overall']}  wall={sweep_wall:.0f}s", flush=True)
    for k, g in gate_vals.items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {k}: {g['value']}  ({g['gate']})",
              flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())