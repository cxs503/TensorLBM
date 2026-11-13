#!/usr/bin/env python3
"""XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS — TensorLBM D2Q9 free surface vs the OSS
``Maarten-vd-Sande/lbm`` minimal D2Q9 free-surface oracle.

Design (pre-declared before any result was read; frozen in ``DESIGN``)
---------------------------------------------------------------------
decision      Is TensorLBM's D2Q9 Körner free surface interchangeable with
              the OSS minimal mass-tracking oracle on a dam-break collapse,
              i.e. do both land inside the owner's 15 % band against the
              Martin & Moyce (1952) experiment?
alternatives  A ``tensorlbm_koerner``: repo ``free_surface_step_2d``
                 (Körner 2005 / Thürey 2007, independent mass ledger, ABB,
                  half-way walls), BGK τ = 1.0.
              B ``oss_port``: ``xval/oracle_minimal_d2q9_fs/oss_oracle_port``
                 — a port of Maarten-vd-Sande/lbm verified against the
                 unmodified upstream code (``oss_original``) to float32
                 round-off; BGK ω = 1.0.
hypothesis    Both arms agree with M&M within 15 % on the front-position
              curve, and with each other within 15 %; any larger gap is a
              model/practice difference (mass-exchange rule, wall law,
              interface initialisation), not numerical noise.
metric        Z(T) = x_front/a on T = t sqrt(g/h0), sampled every 0.25 T over
              the pre-declared window; per-arm ``max |rel err|`` vs the
              digitised M&M curve (piecewise-linear interpolation, no
              extrapolation beyond the digitised window), arm-to-arm
              ``max |ΔZ|/Z``, and liquid-inventory drift.
threshold     15 % (owner precision standard) on both the experiment error
              and the arm-to-arm gap; inventory drift < 1 %.
control      hydrostatic column (g on, initial block), spurious |u|_max.
variables    arm, lattice resolution a (16/32/48, h0 = 2a fixed),
              relaxation τ = 0.6/1.0/1.4, gravity (Mach number), OSS quirk
              switches (rho_prev alias, interface init density).
not varied   domain aspect (nx = 5a, headroom = a), walls on four sides,
              no surface tension, BGK collision only.
stopping rule fixed step count = ceil(T_max/sqrt(g/h0)); no early stopping,
              no threshold tuning after the fact (thresholds are constants
              in this file, checked into the tree before the run).
environment  CPU-only, torch CPU single thread per worker, one process per
              arm via multiprocessing; floating point; lattice units.

Usage
    python xval/oracle_minimal_d2q9_fs/run_xval.py --matrix     # full sweep
    python xval/oracle_minimal_d2q9_fs/run_xval.py --arm oss_port --a 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

MM = json.loads((REPO / "src" / "artifacts" / "martin_moyce_1952_dam_break.json").read_text())
MM_T = [d["t_star"] for d in MM["data"]]
MM_Z = [d["X_star"] for d in MM["data"]]

DESIGN = {
    "anchor": "XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS",
    "oss_reference": {
        "repo": "Maarten-vd-Sande/lbm",
        "snapshot": "/root/oss-cache/Maarten-vd-Sande_lbm",
        "files": ["lbm/util.py", "lbm/init.py", "lbm/streamcollide.py",
                  "lbm/update_types.py", "lbm/time_evolve.py",
                  "examples/dambreak.py"],
        "shipped_case": {"nx": 170, "ny": 170, "omega": 1.0,
                         "gravity": [0.0, -0.1], "steps": 750,
                         "block": "x in [0,70), y in [40,170)"},
    },
    "experiment_reference": {
        "source": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312-324, Fig. 4",
        "doi": "10.1098/rsta.1952.0006",
        "artifact": "src/artifacts/martin_moyce_1952_dam_break.json",
        "sha256": MM.get("sha256"),
    },
    "nondim": {"Z": "x_front / a", "T": "t * sqrt(g / h0)"},
    "thresholds_pct": {"z_max_rel_err": 15.0, "arm_arm_max_rel_diff": 15.0,
                       "inventory_drift": 1.0},
    "window_T": [0.5, 4.0],
    "sample_dT": 0.25,
    "case": {"aspect_h0_over_a": 2.0, "width_in_a": 5.0, "headroom_in_a": 1.0,
             "walls": "solid on all four sides", "g_default": 5e-5,
             "ma_target": 0.1},
    "arms": {
        "tensorlbm_koerner": {
            "module": "tensorlbm.free_surface_lbm_2d",
            "entry": "free_surface_step_2d",
            "relaxation": "tau", "default_tau": 1.0,
        },
        "oss_port": {
            "module": "xval/oracle_minimal_d2q9_fs/oss_oracle_port.py",
            "entry": "step", "relaxation": "omega (omega = 1/tau)",
            "default_tau": 1.0,
        },
    },
    "sensitivity": {
        "resolution": [16, 32, 48],
        "tau": [0.6, 1.0, 1.4],
        "oss_quirk_switches": ["rho_prev_alias", "interface_rho"],
        "mach_probe": {"g": 0.1, "note": "OSS shipped gravity, Ma ~ 6 with h0=2a"},
    },
    "stopping_rule": "steps = ceil(T_max / sqrt(g/h0))",
}


# ---------------------------------------------------------------------------
def mm_interp(t_star: float) -> float | None:
    if t_star < MM_T[0] or t_star > MM_T[-1]:
        return None
    return float(np.interp(t_star, MM_T, MM_Z))


def front_index_port(st) -> int:
    return st.front_cells()


# ---------------------------------------------------------------------------
# arm A: TensorLBM Körner free surface
# ---------------------------------------------------------------------------
def run_tensorlbm(a: int, tau: float, g: float, t_max: float, sample_dt: float):
    import torch
    torch.set_num_threads(1)
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.free_surface_lbm_2d import (
        INTERFACE, LIQUID, free_surface_step_2d, init_fill_rectangular_2d,
        init_flags_from_fill_2d, total_liquid_inventory_2d,
    )

    h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
    ny = h0 + a                                   # headroom = a
    fill, solid = init_fill_rectangular_2d(ny, nx, float(a + 1), float(h0 + 1), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    rho0 = torch.where(flags == LIQUID, torch.ones_like(fill), torch.ones_like(fill))
    f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill.clone()
    inv0 = total_liquid_inventory_2d(f, mass, flags)
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(t_max / dtT))
    sample_every = max(1, int(round(sample_dt / dtT)))
    series, inv_tr = [], []
    t0 = time.time()
    u_max = 0.0
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            # same convention as OSS front_cells(): rightmost LIQUID or
            # INTERFACE column (a leading INTERFACE cell is part of the front)
            fluid = (flags == LIQUID) | (flags == INTERFACE)
            cols = torch.nonzero(fluid.any(dim=0)).flatten()
            front = int(cols.max().item()) if cols.numel() else 0  # 0-based, wall at 0
            series.append({"step": step, "T": step * dtT, "Z": front / a})
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (total_liquid_inventory_2d(f, mass, flags)
                                                 - inv0) / inv0})
        if step == n_steps:
            break
        # epsilon needed by the step (mass ledger / rho_liquid)
        fill = torch.where(flags == LIQUID, torch.ones_like(fill),
                           torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0),
                                       torch.zeros_like(fill)))
        f, mass, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gy=-g, rho_liquid=1.0, rho_gas=1.0,
            mass=mass, return_mass=True)
        rho = f.sum(0).clamp(min=1e-9)
        ux = f[[1, 5, 8]].sum(0) / rho
        uy = f[[2, 6, 7]].sum(0) / rho
        fluid = (flags == LIQUID) | (flags == INTERFACE)
        u_max = max(u_max, float(((ux ** 2 + uy ** 2).sqrt() * fluid).max()))
        if not torch.isfinite(f).all():
            return {"arm": "tensorlbm_koerner", "a": a, "tau": tau, "g": g,
                    "status": "nan", "nan_at_step": step, "series": series,
                    "inventory": inv_tr, "wall_seconds": round(time.time() - t0, 2)}
    return {"arm": "tensorlbm_koerner", "a": a, "tau": tau, "g": g, "status": "ok",
            "h0": h0, "nx": nx, "ny": ny, "steps": n_steps,
            "nu": (tau - 0.5) / 3.0, "u_max": u_max,
            "series": series, "inventory": inv_tr,
            "final_fields": _tensorlbm_snapshot(f, mass, flags, a, g),
            "wall_seconds": round(time.time() - t0, 2)}


def _tensorlbm_snapshot(f, mass, flags, a, g):
    """Velocity + fill snapshot for the field figure (downsampled to .npy-free json)."""
    import torch
    from tensorlbm.free_surface_lbm_2d import INTERFACE, LIQUID
    rho = f.sum(0).clamp(min=1e-9)
    ux = (f[[1, 5, 8]].sum(0) / rho)
    uy = (f[[2, 6, 7]].sum(0) / rho)
    fluid = (flags == LIQUID) | (flags == INTERFACE)
    umag = ((ux ** 2 + uy ** 2).sqrt() * fluid).cpu().numpy()
    fill = torch.where(flags == LIQUID, torch.ones_like(mass),
                       torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0),
                                   torch.zeros_like(mass)))
    return {"umag": np.round(umag, 5).tolist(), "fill": np.round(fill.cpu().numpy(), 4).tolist()}


# ---------------------------------------------------------------------------
# arm B: OSS oracle port
# ---------------------------------------------------------------------------
def run_oss_port(a: int, tau: float, g: float, t_max: float, sample_dt: float,
                 rho_prev_alias: bool = True, interface_rho: str = "oss"):
    import oss_oracle_port as P

    h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
    ny = h0 + a
    st = P.make_dambreak(nx=nx, ny=ny, a=a, h0=h0, interface_rho=interface_rho)
    omega = 1.0 / tau
    inv0 = st.liquid_inventory()
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(t_max / dtT))
    sample_every = max(1, int(round(sample_dt / dtT)))
    series, inv_tr = [], []
    t0 = time.time()
    u_max = 0.0
    snapshot = None
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            series.append({"step": step, "T": step * dtT,
                           "Z": st.front_cells() / a})
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (st.liquid_inventory() - inv0) / inv0})
        if step == n_steps:
            break
        P.step(st, omega=omega, gravity=(0.0, -g), rho_prev_alias=rho_prev_alias)
        um = st.velocity_magnitude()
        if um.size:
            u_max = max(u_max, float(um.max()))
        if not np.isfinite(P.I(st.fin)).all():
            return {"arm": "oss_port", "a": a, "tau": tau, "g": g, "status": "nan",
                    "nan_at_step": step, "series": series, "inventory": inv_tr,
                    "wall_seconds": round(time.time() - t0, 2)}
    snapshot = {"umag": np.round(st.velocity_magnitude(), 5).tolist(),
                "fill": np.round(np.where(st.flags() == P.FLUID, 1.0,
                                          np.where(st.flags() == P.INTERFACE,
                                                   P.I(st.mass), 0.0)), 4).tolist()}
    return {"arm": "oss_port", "a": a, "tau": tau, "g": g, "status": "ok",
            "h0": h0, "nx": nx, "ny": ny, "steps": n_steps, "omega": omega,
            "nu": (tau - 0.5) / 3.0, "u_max": u_max,
            "rho_prev_alias": rho_prev_alias, "interface_rho": interface_rho,
            "series": series, "inventory": inv_tr, "final_fields": snapshot,
            "wall_seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
def run_task(task: dict) -> dict:
    arm = task.pop("arm")
    fn = run_tensorlbm if arm == "tensorlbm_koerner" else run_oss_port
    try:
        out = fn(**task)
    except Exception as exc:  # keep the sweep alive, record honestly
        return {"arm": arm, **task, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    out["label"] = task_label(arm, task)
    return out


def task_label(arm: str, task: dict) -> str:
    bits = [arm, f"a{task['a']}", f"tau{task['tau']:g}"]
    if task.get("g", 5e-5) != 5e-5:
        bits.append(f"g{task['g']:g}")
    if arm == "oss_port":
        if not task.get("rho_prev_alias", True):
            bits.append("rho_prev_fixed")
        if task.get("interface_rho", "oss") != "oss":
            bits.append("if_rho_liquid")
    return "_".join(bits)


def evaluate(res: dict) -> dict:
    """Attach the pre-declared metrics to a run."""
    lo, hi = DESIGN["window_T"]
    pts = []
    for s in res.get("series", []):
        if not (lo - 1e-9 <= s["T"] <= hi + 1e-9):
            continue
        ref = mm_interp(s["T"])
        if ref is None:
            continue
        pts.append({"T": round(s["T"], 4), "Z": round(s["Z"], 4),
                    "Z_ref": round(ref, 4),
                    "rel_err_pct": round(100.0 * (s["Z"] - ref) / ref, 3)})
    if pts:
        errs = np.array([abs(p["rel_err_pct"]) for p in pts])
        res["vs_mm"] = {"points": pts,
                        "max_abs_rel_err_pct": round(float(errs.max()), 3),
                        "mean_abs_rel_err_pct": round(float(errs.mean()), 3),
                        "rmse_rel_pct": round(float(np.sqrt((errs ** 2).mean())), 3),
                        "pass_15pct": bool(errs.max() <= DESIGN["thresholds_pct"]["z_max_rel_err"])}
    inv = res.get("inventory") or []
    if inv:
        drift = max(abs(p["drift_pct"]) for p in inv)
        res["inventory_drift_max_pct"] = round(float(drift), 4)
        res["inventory_pass"] = bool(drift <= DESIGN["thresholds_pct"]["inventory_drift"])
    return res


def compare_arms(res_a: dict, res_b: dict) -> dict:
    """Arm-to-arm gap on the common T grid."""
    da = {round(s["T"], 4): s["Z"] for s in res_a.get("series", [])}
    db = {round(s["T"], 4): s["Z"] for s in res_b.get("series", [])}
    lo, hi = DESIGN["window_T"]
    pts = []
    for T in sorted(set(da) & set(db)):
        if not (lo - 1e-9 <= T <= hi + 1e-9):
            continue
        za, zb = da[T], db[T]
        denom = max(abs(za), abs(zb), 1e-12)
        pts.append({"T": T, "Z_tensorlbm": za, "Z_oss": zb,
                    "rel_diff_pct": round(100.0 * (za - zb) / denom, 3)})
    if not pts:
        return {"points": [], "max_abs_rel_diff_pct": None, "pass_15pct": None}
    diffs = np.array([abs(p["rel_diff_pct"]) for p in pts])
    return {"points": pts,
            "max_abs_rel_diff_pct": round(float(diffs.max()), 3),
            "mean_abs_rel_diff_pct": round(float(diffs.mean()), 3),
            "pass_15pct": bool(diffs.max() <= DESIGN["thresholds_pct"]["arm_arm_max_rel_diff"])}


def hydrostatic_control(arm: str, a: int = 32, tau: float = 1.0,
                        g: float = 5e-5, steps: int = 400) -> dict:
    """Control: column at rest under gravity -> spurious velocity scale."""
    if arm == "tensorlbm_koerner":
        import torch
        torch.set_num_threads(1)
        from tensorlbm.d2q9 import equilibrium
        from tensorlbm.free_surface_lbm_2d import (
            INTERFACE, LIQUID, free_surface_step_2d, init_fill_rectangular_2d,
            init_flags_from_fill_2d,
        )
        h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
        ny = h0 + a
        fill, solid = init_fill_rectangular_2d(ny, nx, float(a + 1), float(h0 + 1), "cpu")
        flags = init_flags_from_fill_2d(fill, solid)
        f = equilibrium(torch.ones_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
        mass = fill.clone()
        u_max = 0.0
        for _ in range(steps):
            fill = torch.where(flags == LIQUID, torch.ones_like(fill),
                               torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0),
                                           torch.zeros_like(fill)))
            f, mass, flags = free_surface_step_2d(f, fill, flags, solid, tau=tau, gy=-g,
                                                  mass=mass, return_mass=True)
            rho = f.sum(0).clamp(min=1e-9)
            ux = f[[1, 5, 8]].sum(0) / rho
            uy = f[[2, 6, 7]].sum(0) / rho
            fluid = (flags == LIQUID) | (flags == INTERFACE)
            u_max = max(u_max, float(((ux ** 2 + uy ** 2).sqrt() * fluid).max()))
        return {"control": "hydrostatic", "arm": arm, "steps": steps,
                "u_max_lattice": u_max}
    import oss_oracle_port as P
    h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
    ny = h0 + a
    st = P.make_dambreak(nx=nx, ny=ny, a=a, h0=h0)
    u_max = 0.0
    for _ in range(steps):
        P.step(st, omega=1.0 / tau, gravity=(0.0, -g))
        um = st.velocity_magnitude()
        u_max = max(u_max, float(um.max()))
    return {"control": "hydrostatic", "arm": arm, "steps": steps,
            "u_max_lattice": u_max}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--arm", default="tensorlbm_koerner")
    ap.add_argument("--a", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=DESIGN["case"]["g_default"])
    ap.add_argument("--tmax", type=float, default=DESIGN["window_T"][1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="xval_results.json")
    args = ap.parse_args()

    common = {"g": args.g, "t_max": args.tmax,
              "sample_dt": DESIGN["sample_dT"]}
    if not args.matrix:
        tasks = [{"arm": args.arm, "a": args.a, "tau": args.tau, **common}]
    else:
        tasks = []
        for a in DESIGN["sensitivity"]["resolution"]:
            for arm in DESIGN["arms"]:
                tasks.append({"arm": arm, "a": a, "tau": 1.0, **common})
        for tau in (0.6, 1.4):
            for arm in DESIGN["arms"]:
                tasks.append({"arm": arm, "a": 32, "tau": tau, **common})
        # OSS quirk ablations at the reference resolution
        tasks.append({"arm": "oss_port", "a": 32, "tau": 1.0,
                      "rho_prev_alias": False, **common})
        tasks.append({"arm": "oss_port", "a": 32, "tau": 1.0,
                      "interface_rho": "liquid", **common})
        # Mach (compressibility) probe at the OSS shipped gravity
        tasks.append({"arm": "oss_port", "a": 16, "tau": 1.0, "g": 0.1, **common})
        tasks.append({"arm": "tensorlbm_koerner", "a": 16, "tau": 1.0, "g": 0.1, **common})

    print(f"[xval] {len(tasks)} runs, up to {args.workers} in parallel", flush=True)
    t0 = time.time()
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        results = pool.map(run_task, tasks)
    results = [evaluate(r) for r in results]
    for r in results:
        if r.get("status") != "ok":
            print(f"  {r.get('label', r['arm'])}: {r['status']} {r.get('error', '')}", flush=True)
            continue
        m = r.get("vs_mm", {})
        print(f"  {r['label']:<44} Z_err max {m.get('max_abs_rel_err_pct'):>6.2f}%  "
              f"drift {r.get('inventory_drift_max_pct'):>7.3f}%  "
              f"u_max {r['u_max']:.4f}  {r['wall_seconds']}s", flush=True)

    # controls
    with Pool(processes=2) as pool:
        controls = pool.starmap(hydrostatic_control, [("tensorlbm_koerner",), ("oss_port",)])
    for c in controls:
        print(f"  control {c['arm']}: u_max={c['u_max_lattice']:.3e}", flush=True)

    # arm-to-arm comparisons at matched (a, tau)
    gaps = []
    index = {(r["arm"], r.get("a"), r.get("tau"), r.get("g"), r.get("label")): r
             for r in results if r.get("status") == "ok"}
    base = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        key = (r["a"], r["tau"], r["g"])
        if r.get("rho_prev_alias", True) is False or r.get("interface_rho", "oss") != "oss":
            continue
        base.setdefault(key, {})[r["arm"]] = r
    for key, arms in sorted(base.items(), key=lambda kv: str(kv[0])):
        if len(arms) == 2:
            gap = compare_arms(arms["tensorlbm_koerner"], arms["oss_port"])
            gap.update({"a": key[0], "tau": key[1], "g": key[2]})
            gaps.append(gap)
    for g in gaps:
        print(f"  arm-vs-arm a={g['a']} tau={g['tau']:g}: "
              f"max {g['max_abs_rel_diff_pct']:.2f}% mean {g['mean_abs_rel_diff_pct']:.2f}%",
              flush=True)

    payload = {"design": DESIGN, "env": {"python": sys.version.split()[0],
                                         "numpy": np.__version__,
                                         "workers": args.workers},
               "runs": results, "controls": controls, "arm_vs_arm": gaps,
               "wall_seconds": round(time.time() - t0, 1)}
    (ART / args.tag).write_text(json.dumps(payload, indent=1))
    print(f"[xval] wrote {ART / args.tag} in {payload['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())