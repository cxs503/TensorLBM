#!/usr/bin/env python3
"""XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR — staged, reproducible driver.

Anchor:  ``XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR``
Repo arm: TensorLBM ``src/tensorlbm/les_graph_unroll/`` (solver2d, reference2d,
selection)
Prereg:  ``docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR-PREREG.md``

The driver is stage-wise because a single shell call here is capped at ~100 s:

    python3 scripts/xval_r2_selection_bptt_tgv.py ref          # reference + convergence
    python3 scripts/xval_r2_selection_bptt_tgv.py arm <closure> <objective> <start>
    python3 scripts/xval_r2_selection_bptt_tgv.py window       # secondary train-window study
    python3 scripts/xval_r2_selection_bptt_tgv.py judge        # verdict table (+P1..P5)
    python3 scripts/xval_r2_selection_bptt_tgv.py figures
    python3 scripts/xval_r2_selection_bptt_tgv.py all          # everything (slow)

Every number in the report is read back from
``artifacts/xval_r2_selection_bptt_tgv/results.json`` (zero hand-copy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tensorlbm.les_graph_unroll import reference2d as ref2d  # noqa: E402
from tensorlbm.les_graph_unroll.selection import (  # noqa: E402
    OBJECTIVES,
    SelectionCase,
    autograd_gradient,
    bptt_select,
    fd_gradient,
    golden_select,
    holdout_metrics,
    objective_loss,
    sweep_select,
)
from tensorlbm.les_graph_unroll.solver2d import equilibrium  # noqa: E402

ART = _ROOT / "artifacts" / "xval_r2_selection_bptt_tgv"
RESULTS = ART / "results.json"
REF_NPZ = ART / "reference.npz"

# ---- fixed workload (prereg §3) -------------------------------------------
U0 = 0.1          # lattice reference velocity
RE = 500.0        # Reynolds number
NX_REF = 128      # fine/reference grid
NX_CONV = 256     # convergence grid
NX_LES = 32       # coarse (LES) grid
T_STEPS = 200     # unroll window
TRAIN_STEPS = (50, 100, 150)
HOLDOUT = 200
SWEEP_GRID = tuple(round(0.025 * i, 4) for i in range(25))  # 0.000 .. 0.600
CLOSURES = ("smagorinsky", "oss_proxy")
STARTS = (0.02, 0.17)


# ---- io helpers -----------------------------------------------------------
def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"meta": {}, "ref": {}, "arms": {}, "windows": {}, "verdict": {}}


def save_results(res: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2, sort_keys=True))


# ---- stage: reference -----------------------------------------------------
def stage_ref(force: bool = False) -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    if REF_NPZ.exists() and not force:
        d = np.load(REF_NPZ)
        print("reference cached:", list(d.keys()))
        return {}

    t0 = time.time()
    # fine reference on 128^2 (DNS proxy), snapshots at every judged step
    f0, _, _ = ref2d.tgv_ic(NX_REF, U0)
    tau_ref = ref2d.tau_for_re(NX_REF, U0, RE)
    _, snaps = ref2d.run_reference(f0, T_STEPS, tau_ref, snapshot_steps=(50, 100, 150, 200))
    # convergence grid 256^2 -- terminal snapshot only
    f0c, _, _ = ref2d.tgv_ic(NX_CONV, U0)
    tau_conv = ref2d.tau_for_re(NX_CONV, U0, RE)
    _, snaps_c = ref2d.run_reference(f0c, T_STEPS, tau_conv, snapshot_steps=(200,))

    out = {}
    for k, (ux, uy) in snaps.items():
        out[f"ux_{k}"] = ref2d.box_filter(ux, NX_REF // NX_LES)
        out[f"uy_{k}"] = ref2d.box_filter(uy, NX_REF // NX_LES)
    uxc, uyc = snaps_c[200]
    out["ux_200_n256"] = uxc
    out["uy_200_n256"] = uyc
    np.savez_compressed(REF_NPZ, **out)

    ke_c = ref2d.kinetic_energy(uxc, uyc)
    ke_r = ref2d.kinetic_energy(*[ref2d.box_filter(a, NX_REF // NX_LES) for a in snaps[200]])
    ke_128_full = ref2d.kinetic_energy(*snaps[200])
    res = load_results()
    res["meta"] = {
        "anchor": "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR",
        "mission": 18237,
        "u0": U0, "re": RE, "nx_ref": NX_REF, "nx_conv": NX_CONV, "nx_les": NX_LES,
        "n_steps": T_STEPS, "train_steps": list(TRAIN_STEPS), "holdout_step": HOLDOUT,
        "tau_ref": tau_ref, "tau_conv": tau_conv,
        "tau0": 0.5 + 3.0 * (U0 / RE) / (2 * math.pi / NX_LES) ** 2,
        "ma": U0 * math.sqrt(3.0),
        "nu_phys": U0 / RE,
        "dx_les": 2 * math.pi / NX_LES,
        "sweep_grid": list(SWEEP_GRID),
        "clamp": [1e-3, 1.0],
        "bptt": {"iters": 20, "lr": 0.1, "starts": list(STARTS), "param": "log10 theta"},
        "wall_seconds": time.time() - t0,
    }
    res["ref"] = {
        "ke_128_filtered_200": ke_r,
        "ke_128_raw_200": ke_128_full,
        "ke_256_raw_200": ke_c,
        "conv_ke_err_128raw_vs_256raw": abs(ke_128_full - ke_c) / ke_c,
        "conv_ke_err_128filt_vs_256raw": abs(ke_r - ke_c) / ke_c,
    }
    save_results(res)
    print(json.dumps(res["meta"], indent=2))
    print(json.dumps(res["ref"], indent=2))
    return res


def load_case(closure: str, train_steps=TRAIN_STEPS, n_steps: int = T_STEPS) -> SelectionCase:
    d = np.load(REF_NPZ)
    steps = tuple(sorted(set(train_steps) | {HOLDOUT}))
    ref = {k: (d[f"ux_{k}"], d[f"uy_{k}"]) for k in steps}

    x = np.arange(NX_LES) * (2 * math.pi / NX_LES)
    X = np.broadcast_to(x.reshape(NX_LES, 1), (NX_LES, NX_LES))
    Y = np.broadcast_to(x.reshape(1, NX_LES), (NX_LES, NX_LES))
    ux = U0 * np.sin(X) * np.cos(Y)
    uy = -U0 * np.cos(X) * np.sin(Y)
    rho = np.ones_like(ux)
    f0n = ref2d.equilibrium_np(rho, ux, uy)
    f0 = torch.as_tensor(f0n, dtype=torch.float64)

    tau0 = 0.5 + 3.0 * (U0 / RE) / (2 * math.pi / NX_LES) ** 2
    return SelectionCase(
        f0=f0, ref=ref, tau0=tau0, n_steps=n_steps, closure=closure,
        train_steps=tuple(train_steps), holdout_step=HOLDOUT,
    )


# ---- stage: one (closure, objective, start) arm ---------------------------
def stage_arm(closure: str, objective: str, start: float) -> dict:
    case = load_case(closure)
    t0 = time.time()
    with torch.no_grad():
        j0 = float(objective_loss(case, objective, start))

    bptt = bptt_select(case, objective, start, iters=20, lr=0.1)
    ga = autograd_gradient(case, objective, bptt["theta"])
    gf = fd_gradient(case, objective, bptt["theta"], h=1e-3)
    hold = holdout_metrics(case, bptt["theta"])
    hold0 = holdout_metrics(case, 0.0)

    res = load_results()
    key = f"{closure}|{objective}|{start:g}"
    res["arms"][key] = {
        "closure": closure, "objective": objective, "theta_init": start,
        "bptt": bptt, "loss_at_start": j0, "grad": {"autograd": ga, "fd": gf},
        "holdout": hold, "holdout_no_model": hold0,
        "wall_seconds": time.time() - t0,
    }
    save_results(res)
    print(json.dumps({k: res["arms"][key][k] for k in
                      ("closure", "objective", "theta_init", "loss_at_start", "grad",
                       "holdout")}, indent=2)[:2000])
    print("theta =", bptt["theta"], " loss =", bptt["loss"], " init loss =", bptt["loss_init"])
    return res


# ---- stage: derivative-free controls (sweep + golden) ---------------------
def stage_controls(closure: str, objective: str) -> dict:
    case = load_case(closure)
    t0 = time.time()
    sw = sweep_select(case, objective, SWEEP_GRID)
    gd = golden_select(case, objective, 0.0, 0.6, iters=40)
    res = load_results()
    res.setdefault("controls", {})[f"{closure}|{objective}"] = {
        "sweep": sw, "golden": gd, "wall_seconds": time.time() - t0,
    }
    save_results(res)
    print(json.dumps({"sweep_argmin": [sw["theta"], sw["loss"]],
                      "golden": [gd["theta"], gd["loss"]]}, indent=2))
    return res


# ---- stage: secondary train window (in-sample for energy) -----------------
def stage_window(closure: str, objective: str, start: float = 0.02) -> dict:
    """Train on {50,100,150,200} and judge the *same* snapshot (declared in-sample)."""
    case = load_case(closure, train_steps=(50, 100, 150, 200))
    bptt = bptt_select(case, objective, start, iters=20, lr=0.1)
    hold = holdout_metrics(case, bptt["theta"])
    res = load_results()
    res["windows"][f"{closure}|{objective}|{start:g}"] = {
        "closure": closure, "objective": objective, "theta_init": start,
        "bptt": bptt, "holdout_in_sample": hold, "note": "in-sample: t=200 trained on",
    }
    save_results(res)
    print(json.dumps({"theta": bptt["theta"], "loss": bptt["loss"],
                      "holdout_in_sample": hold}, indent=2))
    return res


# ---- stage: judge (P1..P5 verdict table) ----------------------------------
def stage_judge() -> dict:
    """Aggregate the prereg verdict table + hold-out curves on the sweep grid."""
    res = load_results()
    t0 = time.time()
    thresh = {"P1_grad_rel": 0.15, "P2_select_agreement_rel": 0.15,
              "P3_ke_err": 0.15, "P4_regret_pp": 15.0, "P5_spread_rel": 0.15}

    # hold-out curves on the sweep grid, per closure (the identifiability probe)
    curves = {}
    for c in CLOSURES:
        case = load_case(c)
        pts = []
        for th in SWEEP_GRID:
            h = holdout_metrics(case, th)
            pts.append({"theta": th, "loss_field": float(objective_loss(case, "field", th)),
                        "loss_energy": float(objective_loss(case, "energy", th)),
                        "loss_enstrophy": float(objective_loss(case, "enstrophy", th)),
                        "ke_err": h["ke_err"], "z_err": h["z_err"],
                        "field_err": h["field_err"], "spec_err": h["spec_err"]})
        # the OSS-as-shipped point (no training at all)
        h_oss = holdout_metrics(case, 0.17)
        curves[c] = {"grid": pts, "theta_oss_default": 0.17, "holdout_oss_default": h_oss}
    res["curves"] = curves

    # per-combo aggregation over the two starts
    combos = {}
    for c in CLOSURES:
        for o in OBJECTIVES:
            arms = {k: v for k, v in res["arms"].items()
                    if v["closure"] == c and v["objective"] == o}
            if not arms:
                continue
            best_key = min(arms, key=lambda k: arms[k]["bptt"]["loss"])
            best = arms[best_key]
            thetas = sorted(v["bptt"]["theta"] for v in arms.values())
            spread = (thetas[-1] - thetas[0]) / max(abs(thetas[0]), 1e-12)
            ctrl = res.get("controls", {}).get(f"{c}|{o}")
            combos[f"{c}|{o}"] = {
                "closure": c, "objective": o,
                "theta_selected": best["bptt"]["theta"],
                "loss_selected": best["bptt"]["loss"],
                "theta_by_start": {k.split("|")[-1]: v["bptt"]["theta"] for k, v in arms.items()},
                "loss_by_start": {k.split("|")[-1]: v["bptt"]["loss"] for k, v in arms.items()},
                "spread_rel": spread,
                "holdout": best["holdout"],
                "sweep_argmin": ctrl["sweep"]["theta"] if ctrl else None,
                "sweep_argmin_loss": ctrl["sweep"]["loss"] if ctrl else None,
                "golden": ctrl["golden"] if ctrl else None,
                "grad_rel_err": max(
                    abs(v["grad"]["autograd"] - v["grad"]["fd"])
                    / max(abs(v["grad"]["fd"]), 1e-300) for v in arms.values()),
                "grad_rel_err": max(
                    abs(v["grad"]["autograd"] - v["grad"]["fd"])
                    / max(abs(v["grad"]["fd"]), 1e-300) for v in arms.values()),
            }

    # selection by the *training* loss only (prereg rule)
    sel_key = min(combos, key=lambda k: combos[k]["loss_selected"])
    sel = combos[sel_key]

    # best achievable hold-out KE error over the whole candidate space
    ke_grid = [(p["ke_err"], c, p["theta"]) for c in CLOSURES for p in curves[c]["grid"]]
    ke_min, ke_min_closure, ke_min_theta = min(ke_grid)
    no_model = curves["smagorinsky"]["grid"][0]  # theta = 0.0
    oss_ship = curves["oss_proxy"]["holdout_oss_default"]

    verdict = {
        "thresholds": thresh,
        "combos": combos,
        "selected_by_training_loss": sel_key,
        "selected_theta": sel["theta_selected"],
        "selected_ke_err": sel["holdout"]["ke_err"],
        "ke_err_min_over_space": ke_min,
        "ke_err_min_at": {"closure": ke_min_closure, "theta": ke_min_theta},
        "no_model": {"theta": 0.0, "ke_err": no_model["ke_err"],
                     "field_err": no_model["field_err"], "z_err": no_model["z_err"]},
        "oss_as_shipped": {"closure": "oss_proxy", "theta": 0.17,
                           "ke_err": oss_ship["ke_err"], "z_err": oss_ship["z_err"],
                           "field_err": oss_ship["field_err"]},
        "regret_pp": (sel["holdout"]["ke_err"] - ke_min) * 100.0,
        "regret_rel": (sel["holdout"]["ke_err"] - ke_min) / max(ke_min, 1e-300),
        "P1": {
            "max_grad_rel_err": max(v["grad_rel_err"] for v in combos.values()),
            "pass": max(v["grad_rel_err"] for v in combos.values()) < thresh["P1_grad_rel"],
        },
        "P2": {
            "agreement": {
                k: {
                    "theta_bptt": v["theta_selected"],
                    "theta_sweep": v["sweep_argmin"],
                    "rel": abs(v["theta_selected"] - v["sweep_argmin"])
                    / max(abs(v["sweep_argmin"]), 1e-12),
                    "loss_gap_rel": abs(v["loss_selected"] - v["sweep_argmin_loss"])
                    / max(abs(v["sweep_argmin_loss"]), 1e-300),
                }
                for k, v in combos.items()
            },
        },
        "P3": {"selected_ke_err": sel["holdout"]["ke_err"],
               "pass": sel["holdout"]["ke_err"] < thresh["P3_ke_err"]},
        "P4": {"regret_pp": (sel["holdout"]["ke_err"] - ke_min) * 100.0,
               "pass": (sel["holdout"]["ke_err"] - ke_min) * 100.0 <= thresh["P4_regret_pp"]},
        "P5": {k: {"spread_rel": v["spread_rel"],
                   "pass": v["spread_rel"] < thresh["P5_spread_rel"]}
               for k, v in combos.items()},
        "wall_seconds": time.time() - t0,
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "cwd": str(_ROOT), "dtype": "float64"},
    }
    verdict["P2"]["note"] = ("agreement judged on theta when the derivative-free argmin "
                             "is > 0.02; otherwise the bound-hit case is reported via "
                             "loss_gap_rel (and the hold-out KE error).")
    verdict["P2"]["pass_theta"] = all(
        (a["rel"] < thresh["P2_select_agreement_rel"]) or (a["theta_sweep"] <= 0.02)
        for a in verdict["P2"]["agreement"].values())
    verdict["P2"]["pass_loss"] = all(
        a["loss_gap_rel"] < thresh["P2_select_agreement_rel"]
        for a in verdict["P2"]["agreement"].values())
    res["verdict"] = verdict
    save_results(res)
    print(json.dumps({k: verdict[k] for k in
                      ("selected_by_training_loss", "selected_theta", "selected_ke_err",
                       "ke_err_min_over_space", "ke_err_min_at", "no_model",
                       "oss_as_shipped", "regret_pp", "regret_rel")}, indent=2))
    print("P1", verdict["P1"]["pass"], "P3", verdict["P3"]["pass"],
          "P4", verdict["P4"]["pass"], "P2theta", verdict["P2"]["pass_theta"],
          "P2loss", verdict["P2"]["pass_loss"],
          "P5", {k: v["pass"] for k, v in verdict["P5"].items()})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ref", "arm", "controls", "window", "judge", "all"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.stage == "ref":
        stage_ref(force=a.force)
    elif a.stage == "arm":
        stage_arm(a.args[0], a.args[1], float(a.args[2]))
    elif a.stage == "controls":
        stage_controls(a.args[0], a.args[1])
    elif a.stage == "window":
        stage_window(*a.args[:3])
    elif a.stage == "judge":
        stage_judge()
    elif a.stage == "all":
        stage_ref(force=a.force)
        for c in CLOSURES:
            for o in OBJECTIVES:
                for s in STARTS:
                    stage_arm(c, o, s)
                stage_controls(c, o)


if __name__ == "__main__":
    main()