#!/usr/bin/env python3
"""XVAL-LBM-MASS-EXCHANGE-SELECTOR — TensorLBM vs the OSS
``Maarten-vd-Sande/lbm`` mass-exchange selector.

Pre-declared design (``DESIGN`` below) is frozen in this file before any result
was read.  Three experiments, all on the same lattice-unit dam-break case:

E1  port fidelity      the selector extracted in ``oss_selector.py`` must
                       reproduce the *fidelity-checked* full port
                       (``xval/oracle_minimal_d2q9_fs/oss_oracle_port.py``)
                       exactly on randomized states — otherwise the
                       differential test below measures my transcription, not
                       the OSS rule.
E2  selector differential (paired, 2x2 factorial)
                       the identical frozen state is fed to both selectors; the
                       factors are (a) fill tolerance kappa in {3e-3 = OSS
                       ``fill_offset``, 1e-2 = repo default} and (b) the
                       unconditional ``NO_IFACE_NEIGH | NO_FLUID_NEIGH``
                       demotion guard (on = OSS, off = repo).
E3  downstream consequence
                       arms A (repo default), C (repo + OSS kappa), B (OSS port)
                       on the dam-break, scored against the digitised Martin &
                       Moyce (1952) curve, plus a kappa / tau / resolution scan.

Usage
    python xval/mass_exchange_selector/run_xval.py --fidelity
    python xval/mass_exchange_selector/run_xval.py --matrix
    python xval/mass_exchange_selector/run_xval.py --analyse
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
sys.path.insert(0, str(REPO / "xval" / "oracle_minimal_d2q9_fs"))

import oss_selector as S            # noqa: E402
import repo_selector as R           # noqa: E402
import oss_oracle_port as P         # noqa: E402

ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
OSS_CACHE = Path("/root/oss-cache/Maarten-vd-Sande_lbm")

MM = json.loads((REPO / "src" / "artifacts" / "martin_moyce_1952_dam_break.json").read_text())
MM_T = [d["t_star"] for d in MM["data"]]
MM_Z = [d["X_star"] for d in MM["data"]]

# ---------------------------------------------------------------------------
# pre-declared design — constants, not tunables
# ---------------------------------------------------------------------------
DESIGN = {
    "anchor": "XVAL-LBM-MASS-EXCHANGE-SELECTOR",
    "oss_reference": {
        "repo": "Maarten-vd-Sande/lbm",
        "snapshot": str(OSS_CACHE),
        "files": ["lbm/update_types.py", "lbm/util.py", "lbm/streamcollide.py"],
        "selector_lines": {
            "fill_offset": "lbm/update_types.py:12  (0.003)",
            "lonely_tresh": "lbm/update_types.py:13  (0.1)",
            "marking": "lbm/update_types.py:18-27",
            "envelope": "lbm/update_types.py:34-70",
            "excess_distribution": "lbm/update_types.py:72-116",
            "neigh_flags": "lbm/update_types.py:118-136",
        },
        "carrier_port": "xval/oracle_minimal_d2q9_fs/oss_oracle_port.py "
                        "(fidelity-checked against the unmodified upstream code)",
    },
    "repo_reference": {
        "file": "src/tensorlbm/free_surface_lbm_2d.py",
        "marking": "604-614 (cell_conversion_threshold / force_threshold)",
        "excess_distribution": "617-626 (uniform 1/n_recv split)",
        "mass_exchange": "345-370 (Koerner eq.9 / Thuerey table 4.1)",
    },
    "decision": "Transplant the OSS selector (fill_offset = 3e-3 + the "
                "NO_IFACE_NEIGH / NO_FLUID_NEIGH demotion guard + the "
                "normal-weighted excess split) into TensorLBM's D2Q9 "
                "free-surface step, or keep the repo's Koerner/Thuerey "
                "selector?",
    "alternatives": {
        "A_repo_selector": "kappa = 1e-2, kappa_f = 0.1, uniform 1/n_recv "
                           "excess split (repo status quo)",
        "B_oss_selector": "fill_offset = 3e-3, lonely = 0.1, unconditional "
                          "no-iface/no-fluid demotion, eta-weighted excess split",
        "C_hybrid": "repo excess split, OSS conversion tolerances "
                    "(kappa -> 3e-3).  Reachable through the existing public "
                    "API (cell_conversion_threshold), so it is the only "
                    "transplant that needs no library change.",
    },
    "hypothesis": [
        "H1 the selector is second order for the collapse front: the A-vs-C "
            "front-curve gap |Z_A - Z_C| stays inside the 15 % owner band at "
            "every sampled T over the whole T in [0.5, 4] window, at a = 16, "
            "32 and 48",
        "H2 the selector is first order for mass book-keeping: the OSS guard "
            "set changes the liquid-inventory drift by more than 1 % (relative)",
        "H3 the decision disagreement on identical states is dominated by the "
            "unconditional no-iface/no-fluid demotion, not by the 3.3x "
            "tolerance ratio (2x2 factorial attribution)",
    ],
    "thresholds": {
        "owner_precision_pct": 15.0,     # accuracy vs Martin & Moyce
        "arm_arm_pct": 15.0,             # A vs C front-curve interchangeability
        "inventory_drift_pct": 1.0,      # mass-closure acceptance
        "selector_disagreement_pct": 5.0,  # H3: interchangeability of decisions
        "zero_tol_mass": 1e-12,          # E1 fidelity, float64
    },
    "case": {
        "name": "2D dam-break, walled box",
        "h0_over_a": 2.0, "width_in_a": 5.0, "headroom_in_a": 1.0,
        "gravity": 5e-5, "tau": 1.0, "nu": 1.0 / 6.0, "Ma_target": 0.1,
        "T_max": 4.0, "sample_dT": 0.25, "window_T": [0.5, 4.0],
        "nondim": {"Z": "x_front / a", "T": "t sqrt(g / h0)"},
    },
    "experiment_reference": {
        "source": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312-324, Fig. 4",
        "doi": "10.1098/rsta.1952.0006",
        "artifact": "src/artifacts/martin_moyce_1952_dam_break.json",
    },
    "arms": {
        "A_repo": {"kappa": 1e-2, "kappa_f": 0.1},
        "C_hybrid": {"kappa": 3e-3, "kappa_f": 0.1},
        "B_oss": {"rho_prev_alias": False,
                  "note": "OSS port with its shipped quirk Q1 disabled; the "
                          "shipped alias is measured separately because it "
                          "drifts 200 % inventory (XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS) "
                          "and would swamp the selector signal"},
    },
    "deviation_predeclared": "B runs with rho_prev_alias=False (quirk Q1 "
                             "disabled).  Q1 is an OSS aliasing bug, not part "
                             "of the selector; the shipped configuration is "
                             "still measured once at a = 16 for the record.",
    "sensitivity": {"resolution": [16, 32, 48], "kappa": [3e-3, 1e-2, 3e-2],
                    "tau": [0.6, 1.0, 1.4]},
    "repetitions": "n = 1 per arm; determinism is verified instead by a "
                   "bitwise repeat of arm A at a = 16 (CPU, single thread, "
                   "float32) — if the repeat is bitwise identical, replica "
                   "counts add no information.",
    "stopping_rule": "steps = ceil(T_max / sqrt(g/h0)); no early stopping; no "
                     "threshold edited after inspection.",
    "environment": "CPU-only torch 2.7.1, 1 thread per worker, one process per "
                   "arm, lattice units, no cluster needed (all arms < 10 min).",
}


# ---------------------------------------------------------------------------
def mm_at(t_star):
    lo, hi = DESIGN["case"]["window_T"]
    if t_star < MM_T[0] or t_star > MM_T[-1] or t_star < lo or t_star > hi:
        return None
    return float(np.interp(t_star, MM_T, MM_Z))


def score_series(series):
    """max / mean |delta Z| / Z_ref over the pre-declared window."""
    errs = []
    for pt in series:
        ref = mm_at(pt["T"])
        if ref is None or ref <= 0:
            continue
        errs.append((pt["T"], 100.0 * abs(pt["Z"] - ref) / ref))
    if not errs:
        return {"n": 0, "max_pct": None, "mean_pct": None}
    arr = np.array([e[1] for e in errs])
    return {"n": len(errs), "max_pct": round(float(arr.max()), 3),
            "mean_pct": round(float(arr.mean()), 3),
            "worst_T": errs[int(arr.argmax())][0]}


# ---------------------------------------------------------------------------
# E1 — port fidelity of the extracted selector
# ---------------------------------------------------------------------------
def random_state(rng, nx=48, ny=32, dtype=np.float64):
    """A plausible free-surface state: liquid below a random height field,
    one interface layer, liquid mass perturbed around fill ~ 1."""
    st = P.State(nx, ny, dtype=dtype)
    height = 4 + rng.integers(0, 10, size=nx)
    cls = np.full((nx, ny), P.GAS, dtype=np.int32)
    for i in range(nx):
        cls[i, 1:1 + height[i]] = P.FLUID
        if 1 + height[i] < ny - 1:
            cls[i, 1 + height[i]] = P.INTERFACE
    cls = np.where(rng.random((nx, ny)) < 0.08,
                   np.where(cls == P.FLUID, P.INTERFACE, cls), cls)
    rho = 1.0 + 0.05 * rng.standard_normal((nx, ny))
    fill = np.where(cls == P.FLUID, 1.0,
                    np.where(cls == P.INTERFACE,
                             rng.uniform(-0.2, 1.3, (nx, ny)), 0.0))
    mass = fill * rho
    st.ct[1:-1, 1:-1] = cls
    st.rho[1:-1, 1:-1] = rho
    st.mass[1:-1, 1:-1] = mass
    # upstream holds neighbour bits from the previous sweep
    st.ct[1:-1, 1:-1] = S.neigh_flags(st.ct[1:-1, 1:-1])
    return st


def run_fidelity(n_states=40, seed=20261112, dtype=np.float64):
    """The port's carrier dtype must be float64 here: the pre-declared 1e-12
    tolerance is a float64 round-off statement.  A float32 pass is recorded as
    context only (its floor is float32 eps, ~1e-7)."""
    rng = np.random.default_rng(seed)
    rec = []
    worst_mask, worst_mass = 0, 0.0
    for trial in range(n_states):
        st = random_state(rng, dtype=dtype)
        pre = P.I(st.ct).copy()
        rho_prev = P.I(st.rho).copy()
        mass_prev = P.I(st.mass).copy()
        st2 = P.State(st.nx, st.ny, dtype=dtype)
        st2.ct[:] = st.ct
        st2.rho[:] = st.rho
        st2.mass[:] = st.mass
        st2.u[:] = st.u
        # the port's update_types takes *padded* previous fields (it applies
        # I() internally), exactly as ``step`` passes them
        P.update_types(st2, st.rho.copy(), st.mass.copy(), st.u.copy(),
                       st.ct.copy())
        post = P.I(st2.ct).copy()
        tf_ref = ((post & P.FLUID) != 0) & ((pre & P.INTERFACE) != 0)
        tg_ref = ((post & P.GAS) != 0) & ((pre & P.INTERFACE) != 0)

        out = S.select(pre, rho_prev, mass_prev)
        tf = out["to_fluid"]
        tg = out["to_gas"]
        dmask = int(np.sum(tf != tf_ref) + np.sum(tg != tg_ref))
        # mass fidelity on cells the envelope did not touch
        touched = out["promote"] | out["demote"]
        m_ref = P.I(st2.mass)
        dmass = np.abs(out["mass_new"] - m_ref)[~(touched | out["promote"])]
        worst_mask = max(worst_mask, dmask)
        if dmass.size:
            worst_mass = max(worst_mass, float(dmass.max()))
        rec.append({"trial": trial, "n_iface": int(np.sum((pre & P.INTERFACE) != 0)),
                    "mask_mismatch": dmask,
                    "mass_max_abs_dev": float(dmass.max()) if dmass.size else 0.0})
    verdict = {
        "dtype": str(np.dtype(dtype)),
        "states": n_states,
        "mass_max_abs_dev_float32": float(run_fidelity.__dict__.get("_f32", 0.0)),
        "mask_mismatch_total": int(sum(r["mask_mismatch"] for r in rec)),
        "mass_max_abs_dev": worst_mass,
        "tolerance": DESIGN["thresholds"]["zero_tol_mass"],
        "pass": bool(sum(r["mask_mismatch"] for r in rec) == 0
                     and worst_mass <= DESIGN["thresholds"]["zero_tol_mass"]),
        "per_state": rec,
    }
    (ART / "fidelity.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_state"}, indent=1))
    return verdict


# ---------------------------------------------------------------------------
# E2 — paired factorial selector differential on frozen states
# ---------------------------------------------------------------------------
def to_oss_bits(flags):
    """repo (ny, nx) GAS/LIQUID/INTERFACE/SOLID -> OSS (nx, ny) CT bits."""
    fl = np.asarray(flags).T
    ct = np.full(fl.shape, P.GAS, dtype=np.int32)
    ct = np.where(fl == 1, P.FLUID, ct)
    ct = np.where(fl == 2, P.INTERFACE, ct)
    ct = np.where(fl == 3, P.OBSTACLE, ct)
    return S.neigh_flags(ct)


def differential(flags, mass, rho, kappa=R.DEFAULT_KAPPA,
                 kappa_f=R.DEFAULT_KAPPA_F):
    """All selector variants on one frozen state.

    Layout discipline: repo rules are evaluated in ``(ny, nx)``, OSS rules in
    ``(nx, ny)`` (the port's convention); the two are never mixed without an
    explicit transpose, because a silent transpose on a square patch is
    invisible in the shapes.
    """
    fl = np.asarray(flags)                                  # (ny, nx)
    rho = np.asarray(rho, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    fill = mass / np.where(rho > 0, rho, 1.0)
    liq, gas, _ = R.mark(fl, fill, kappa, kappa_f)           # (ny, nx)

    ct = to_oss_bits(fl)                                    # (nx, ny)
    variants = {}
    for name, fo, guard in (("oss_native", S.FILL_OFFSET, True),
                            ("oss_noGuard", S.FILL_OFFSET, False),
                            ("oss_kappaRepo", 1e-2, True),
                            ("oss_kappaRepo_noGuard", 1e-2, False)):
        tf, tg = S.mark(ct, rho.T, mass.T, fo, S.LONELY_TRESH, guard)
        variants[name] = (tf, tg)                           # (nx, ny)

    n_if = int(np.sum(fl == 2))
    rep = {"n_iface": n_if, "variants": {}}
    for name, (tf, tg) in variants.items():
        tf_r, tg_r = tf.T, tg.T                             # (ny, nx)
        d_liq = (tf_r != liq) & (fl == 2)
        d_gas = (tg_r != gas) & (fl == 2)
        rep["variants"][name] = {
            "disagree_liq": int(d_liq.sum()),
            "disagree_gas": int(d_gas.sum()),
            "disagree_pct": round(100.0 * (d_liq.sum() + d_gas.sum())
                                  / max(n_if, 1), 3),
            "oss_only_gas": int((tg_r & ~gas).sum()),
            "repo_only_gas": int((gas & ~tg_r).sum()),
            "oss_only_liq": int((tf_r & ~liq).sum()),
            "repo_only_liq": int((liq & ~tf_r).sum()),
        }
    rep["decisions"] = {"repo_native": {"n_liq": int(liq.sum()),
                                        "n_gas": int(gas.sum())}}
    for name, (tf, tg) in variants.items():
        rep["decisions"][name] = {"n_liq": int(tf.sum()), "n_gas": int(tg.sum())}
    rep["excess"] = compare_excess(fl, ct, rho, mass, liq, gas,
                                   variants["oss_native"])
    return rep


def compare_excess(fl, ct, rho, mass, liq, gas, oss_native):
    """Same frozen state, both excess-placement rules (does the split itself
    change, or only which cell converts?)."""
    inc_repo, excess, recv_mask, n_recv = R.redistribute(fl, mass, rho, liq, gas)
    tf, tg = oss_native                                      # (nx, ny)
    ct_now, _, _ = S.envelope(ct, tf, tg)
    _, placed, mex = S.redistribute(ct, ct_now, rho.T, mass.T, tf, tg,
                                    mass.T.copy(), rho.T.copy())
    inc_oss = placed.sum(axis=0).T                           # (ny, nx)
    tot_ex = float(np.sum(np.abs(excess)))
    tot_oss = float(np.sum(np.abs(mex)))
    l1 = float(np.sum(np.abs(inc_repo - inc_oss)))
    return {
        "excess_repo": round(tot_ex, 8),
        "excess_oss": round(tot_oss, 8),
        "excess_rel_diff_pct": round(100.0 * abs(tot_ex - tot_oss)
                                     / max(abs(tot_oss), 1e-30), 3),
        "placement_l1": round(l1, 8),
        "placement_l1_over_excess_pct": round(100.0 * l1 / max(tot_oss, 1e-30), 3),
        "increment_sum_repo": round(float(inc_repo.sum()), 8),
        "increment_sum_oss": round(float(inc_oss.sum()), 8),
    }


def synthetic_configs():
    """Five neighbourhood configurations a single INTERFACE cell can be in.

    Only the neighbour *classes* matter to both selectors, so a 7x7 patch per
    configuration is enough to expose every branch.
    """
    cfgs = {}
    fl = np.zeros((7, 7), dtype=np.int64)
    fl[:3, :] = 1; fl[3, :] = 2                      # liquid / iface / gas
    cfgs["layer"] = fl
    fl = np.zeros((7, 7), dtype=np.int64)
    fl[3, :] = 2                                     # interface / gas / gas
    cfgs["no_liq_nbr"] = fl
    fl = np.ones((7, 7), dtype=np.int64)
    fl[3, 3] = 2                                     # single iface in liquid
    cfgs["no_iface_nbr_liq"] = fl
    fl = np.ones((7, 7), dtype=np.int64)
    fl[3, :] = 2                                     # iface strip, no gas
    cfgs["no_gas_nbr"] = fl
    fl = np.zeros((7, 7), dtype=np.int64)
    fl[2, :] = 2; fl[4, :] = 2                       # two interface rows, gas
    cfgs["two_iface"] = fl
    return cfgs


def run_differential():
    """E2 on (a) exported real states from the runs and (b) a synthetic sweep."""
    out = {"states": [], "synthetic": []}
    for npz in sorted(ART.glob("states_*.npz")):
        z = np.load(npz)
        rows = []
        for i in range(len(z["steps"])):
            rep = differential(z[f"flags_{i}"], z[f"mass_{i}"], z[f"rho_{i}"])
            rep["step"] = int(z["steps"][i])
            rep["T"] = float(z["T"][i])
            rows.append(rep)
        out["states"].append({"file": npz.name, "rows": rows})

    fills = np.concatenate([np.round(np.arange(-0.03, 0.031, 0.001), 4),
                            np.round(np.arange(0.85, 1.061, 0.005), 4),
                            np.array([0.5])])
    for name, fl in synthetic_configs().items():
        for fv in fills:
            if fl[3, 3] == 2:
                mass = np.where(fl == 2, fv, np.where(fl == 1, 1.0, 0.0))
            else:
                mass = np.where(fl == 2, fv, np.where(fl == 1, 1.0, 0.0))
            rho = np.ones_like(mass)
            rep = differential(fl, mass, rho)
            rep["config"] = name
            rep["fill"] = float(fv)
            out["synthetic"].append(rep)
    (ART / "differential.json").write_text(json.dumps(out, indent=1))
    print(f"differential: {len(out['synthetic'])} synthetic rows, "
          f"{sum(len(s['rows']) for s in out['states'])} frozen-state rows")
    return out


# ---------------------------------------------------------------------------
# E3 — downstream arms
# ---------------------------------------------------------------------------
def case_geometry(a):
    h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
    ny = h0 + a
    return h0, nx, ny


def run_repo(task):
    import torch
    torch.set_num_threads(1)
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.free_surface_lbm_2d import (
        GAS, INTERFACE, LIQUID, free_surface_step_2d,
        init_fill_rectangular_2d, init_flags_from_fill_2d,
        total_liquid_inventory_2d,
    )

    a, tau, g = task["a"], task["tau"], DESIGN["case"]["gravity"]
    kappa, kappa_f = task["kappa"], task["kappa_f"]
    h0, nx, ny = case_geometry(a)
    fill, solid = init_fill_rectangular_2d(ny, nx, float(a + 1), float(h0 + 1), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    rho0 = torch.ones_like(fill)
    f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill.clone()
    inv0 = total_liquid_inventory_2d(f, mass, flags)
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(DESIGN["case"]["T_max"] / dtT))
    sample_every = max(1, int(round(DESIGN["case"]["sample_dT"] / dtT)))
    series, inv_tr, conv = [], [], {"gas2if": 0, "if2gas": 0, "if2liq": 0, "liq2if": 0}
    exported = []
    t0 = time.time()
    u_max = 0.0
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            fluid = (flags == LIQUID) | (flags == INTERFACE)
            cols = torch.nonzero(fluid.any(dim=0)).flatten()
            front = int(cols.max().item()) if cols.numel() else 0
            series.append({"step": step, "T": step * dtT, "Z": front / a})
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (total_liquid_inventory_2d(f, mass, flags) - inv0) / inv0})
            if task.get("export"):
                rho_s = f.sum(0).clamp(min=1e-9)
                exported.append((step, step * dtT, flags.cpu().numpy().copy(),
                                 mass.cpu().numpy().copy(), rho_s.cpu().numpy().copy()))
        if step == n_steps:
            break
        prev = flags.clone()
        fill = torch.where(flags == LIQUID, torch.ones_like(fill),
                           torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0),
                                       torch.zeros_like(fill)))
        f, mass, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gy=-g, rho_liquid=1.0, rho_gas=1.0,
            mass=mass, return_mass=True,
            cell_conversion_threshold=kappa,
            cell_conversion_force_threshold=kappa_f)
        conv["gas2if"] += int(((prev == GAS) & (flags == INTERFACE)).sum())
        conv["if2gas"] += int(((prev == INTERFACE) & (flags == GAS)).sum())
        conv["if2liq"] += int(((prev == INTERFACE) & (flags == LIQUID)).sum())
        conv["liq2if"] += int(((prev == LIQUID) & (flags == INTERFACE)).sum())
        rho = f.sum(0).clamp(min=1e-9)
        ux = f[[1, 5, 8]].sum(0) / rho
        uy = f[[2, 6, 7]].sum(0) / rho
        fluid = (flags == LIQUID) | (flags == INTERFACE)
        u_max = max(u_max, float(((ux ** 2 + uy ** 2).sqrt() * fluid).max()))
        if not torch.isfinite(f).all():
            return {"label": task["label"], "status": "nan", "nan_step": step}
    if task.get("export"):
        payload = {"steps": np.array([e[0] for e in exported]),
                   "T": np.array([e[1] for e in exported])}
        for i, (_, _, fl, m, r) in enumerate(exported):
            payload[f"flags_{i}"] = fl
            payload[f"mass_{i}"] = m
            payload[f"rho_{i}"] = r
        np.savez_compressed(ART / f"states_{task['label']}.npz", **payload)
    return {"label": task["label"], "status": "ok", "a": a, "tau": tau,
            "kappa": kappa, "kappa_f": kappa_f, "h0": h0, "nx": nx, "ny": ny,
            "steps": n_steps, "nu": (tau - 0.5) / 3.0, "u_max": u_max,
            "series": series, "inventory": inv_tr, "conversions": conv,
            "wall_seconds": round(time.time() - t0, 2)}


def _oss_classes(st):
    ct = st.flags()
    out = np.zeros(ct.shape, dtype=np.int8)          # 0 gas
    out[(ct & P.FLUID) != 0] = 1
    out[(ct & P.INTERFACE) != 0] = 2
    out[(ct & P.OBSTACLE) != 0] = 3
    return out


def run_oss(task):
    a, tau, g = task["a"], task["tau"], DESIGN["case"]["gravity"]
    h0, nx, ny = case_geometry(a)
    st = P.make_dambreak(nx=nx, ny=ny, a=a, h0=h0, interface_rho="oss")
    omega = 1.0 / tau
    inv0 = st.liquid_inventory()
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(DESIGN["case"]["T_max"] / dtT))
    sample_every = max(1, int(round(DESIGN["case"]["sample_dT"] / dtT)))
    series, inv_tr, conv = [], [], {"gas2if": 0, "if2gas": 0, "if2liq": 0, "liq2if": 0}
    exported = []
    t0 = time.time()
    u_max = 0.0
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            series.append({"step": step, "T": step * dtT, "Z": st.front_cells() / a})
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (st.liquid_inventory() - inv0) / inv0})
            if task.get("export"):
                exported.append((step, step * dtT, _oss_classes(st).T.copy(),
                                 P.I(st.mass).T.copy(), P.I(st.rho).T.copy()))
        if step == n_steps:
            break
        prev = _oss_classes(st)
        P.step(st, omega=omega, gravity=(0.0, -g),
               rho_prev_alias=task["rho_prev_alias"])
        cur = _oss_classes(st)
        conv["gas2if"] += int(np.sum((prev == 0) & (cur == 2)))
        conv["if2gas"] += int(np.sum((prev == 2) & (cur == 0)))
        conv["if2liq"] += int(np.sum((prev == 2) & (cur == 1)))
        conv["liq2if"] += int(np.sum((prev == 1) & (cur == 2)))
        um = st.velocity_magnitude()
        if um.size:
            u_max = max(u_max, float(um.max()))
        if not np.isfinite(P.I(st.fin)).all():
            return {"label": task["label"], "status": "nan", "nan_step": step}
    if task.get("export"):
        payload = {"steps": np.array([e[0] for e in exported]),
                   "T": np.array([e[1] for e in exported])}
        for i, (_, _, fl, m, r) in enumerate(exported):
            payload[f"flags_{i}"] = fl
            payload[f"mass_{i}"] = m
            payload[f"rho_{i}"] = r
        np.savez_compressed(ART / f"states_{task['label']}.npz", **payload)
    return {"label": task["label"], "status": "ok", "a": a, "tau": tau,
            "rho_prev_alias": task["rho_prev_alias"], "h0": h0, "nx": nx, "ny": ny,
            "steps": n_steps, "omega": omega, "nu": (tau - 0.5) / 3.0,
            "u_max": u_max, "series": series, "inventory": inv_tr,
            "conversions": conv, "wall_seconds": round(time.time() - t0, 2)}


def run_task(task):
    try:
        if task["kind"] == "repo":
            return run_repo(task)
        return run_oss(task)
    except Exception as exc:  # pragma: no cover - diagnostics only
        import traceback
        return {"label": task["label"], "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


G = DESIGN["case"]["gravity"]
TASKS = [
    {"kind": "repo", "label": "A_repo_a16", "a": 16, "tau": 1.0,
     "kappa": 1e-2, "kappa_f": 0.1, "export": True},
    {"kind": "repo", "label": "A_repo_a16_rep", "a": 16, "tau": 1.0,
     "kappa": 1e-2, "kappa_f": 0.1},
    {"kind": "repo", "label": "C_hybrid_a16", "a": 16, "tau": 1.0,
     "kappa": 3e-3, "kappa_f": 0.1},
    {"kind": "repo", "label": "D_kappa3e2_a16", "a": 16, "tau": 1.0,
     "kappa": 3e-2, "kappa_f": 0.1},
    {"kind": "repo", "label": "E_tau0.6_a16", "a": 16, "tau": 0.6,
     "kappa": 1e-2, "kappa_f": 0.1},
    {"kind": "repo", "label": "F_tau1.4_a16", "a": 16, "tau": 1.4,
     "kappa": 1e-2, "kappa_f": 0.1},
    {"kind": "oss", "label": "B_ossFixed_a16", "a": 16, "tau": 1.0,
     "rho_prev_alias": False, "export": True},
    {"kind": "oss", "label": "B_ossShipped_a16", "a": 16, "tau": 1.0,
     "rho_prev_alias": True},
    {"kind": "repo", "label": "A_repo_a32", "a": 32, "tau": 1.0,
     "kappa": 1e-2, "kappa_f": 0.1},
    {"kind": "repo", "label": "C_hybrid_a32", "a": 32, "tau": 1.0,
     "kappa": 3e-3, "kappa_f": 0.1},
    {"kind": "oss", "label": "B_ossFixed_a32", "a": 32, "tau": 1.0,
     "rho_prev_alias": False},
    {"kind": "repo", "label": "A_repo_a48", "a": 48, "tau": 1.0,
     "kappa": 1e-2, "kappa_f": 0.1},
    {"kind": "repo", "label": "C_hybrid_a48", "a": 48, "tau": 1.0,
     "kappa": 3e-3, "kappa_f": 0.1},
]


def run_matrix(workers=8):
    t0 = time.time()
    with Pool(workers) as pool:
        runs = pool.map(run_task, TASKS)
    for r in runs:
        if r.get("status") == "ok":
            r["score"] = score_series(r["series"])
            inv = [p["drift_pct"] for p in r["inventory"]]
            r["inventory_drift_pct"] = round(float(max(abs(v) for v in inv)), 4)
    payload = {"design": DESIGN, "runs": runs,
               "env": {"torch": __import__("torch").__version__,
                       "numpy": np.__version__, "python": sys.version.split()[0],
                       "host": os.uname().nodename, "cpus": os.cpu_count()},
               "wall_seconds": round(time.time() - t0, 1)}
    (ART / "runs.json").write_text(json.dumps(payload, indent=1))
    print(f"matrix done in {payload['wall_seconds']} s -> artifacts/runs.json")
    return payload


# ---------------------------------------------------------------------------
def analyse():
    runs = json.loads((ART / "runs.json").read_text())["runs"]
    by = {r["label"]: r for r in runs if r.get("status") == "ok"}

    def arm_arm(l1, l2):
        a, b = by[l1], by[l2]
        diffs = []
        for p, q in zip(a["series"], b["series"]):
            if p["T"] < DESIGN["case"]["window_T"][0] or p["T"] > DESIGN["case"]["window_T"][1]:
                continue
            denom = max(abs(q["Z"]), 1e-9)
            diffs.append(100.0 * abs(p["Z"] - q["Z"]) / denom)
        return {"max_pct": round(max(diffs), 3) if diffs else None,
                "mean_pct": round(float(np.mean(diffs)), 3) if diffs else None}

    pairs = [("A_repo_a16", "C_hybrid_a16"), ("A_repo_a32", "C_hybrid_a32"),
             ("A_repo_a48", "C_hybrid_a48"), ("A_repo_a16", "B_ossFixed_a16"),
             ("A_repo_a32", "B_ossFixed_a32"), ("C_hybrid_a32", "B_ossFixed_a32")]
    arm_arm_tab = {f"{l1}_vs_{l2}": arm_arm(l1, l2) for l1, l2 in pairs
                   if l1 in by and l2 in by}
    det = None
    if "A_repo_a16" in by and "A_repo_a16_rep" in by:
        s1 = [p["Z"] for p in by["A_repo_a16"]["series"]]
        s2 = [p["Z"] for p in by["A_repo_a16_rep"]["series"]]
        det = {"bitwise_identical": s1 == s2, "n": len(s1)}

    out = {"arm_arm": arm_arm_tab, "determinism": det,
           "table": [{k: r.get(k) for k in
                      ("label", "a", "tau", "kappa", "kappa_f", "rho_prev_alias",
                       "steps", "u_max", "inventory_drift_pct", "wall_seconds",
                       "conversions")} | {"score": r.get("score")}
                     for r in runs if r.get("status") == "ok"],
           "failed": [r["label"] for r in runs if r.get("status") != "ok"]}
    (ART / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"arm_arm": arm_arm_tab, "determinism": det}, indent=1))
    return out


def run_chunk(labels, workers=4):
    """Run a subset of the matrix and merge into the accumulated runs.json.

    Needed because a single local tool call cannot exceed ~2 minutes: the
    short arms are run here in chunks, the long ones go to the cluster queue.
    """
    tasks = [t for t in TASKS if t["label"] in labels]
    missing = set(labels) - {t["label"] for t in tasks}
    if missing:
        raise SystemExit(f"unknown arm labels: {sorted(missing)}")
    t0 = time.time()
    with Pool(min(workers, len(tasks))) as pool:
        runs = pool.map(run_task, tasks)
    for r in runs:
        if r.get("status") == "ok":
            r["score"] = score_series(r["series"])
            inv = [p["drift_pct"] for p in r["inventory"]]
            r["inventory_drift_pct"] = round(float(max(abs(v) for v in inv)), 4)
    path = ART / "runs.json"
    payload = (json.loads(path.read_text()) if path.exists()
               else {"design": DESIGN, "runs": [], "env": {}, "wall_seconds": 0.0})
    kept = [r for r in payload["runs"] if r["label"] not in {x["label"] for x in runs}]
    payload["runs"] = kept + runs
    payload["env"] = {"torch": __import__("torch").__version__,
                      "numpy": np.__version__, "python": sys.version.split()[0],
                      "host": os.uname().nodename, "cpus": os.cpu_count(),
                      "mode": "local chunk + cluster queue"}
    payload["wall_seconds"] = round(payload.get("wall_seconds", 0.0)
                                    + time.time() - t0, 1)
    path.write_text(json.dumps(payload, indent=1))
    for r in runs:
        print(r["label"], r.get("status"), r.get("score"), r.get("wall_seconds"))
    print(f"chunk done in {round(time.time() - t0, 1)} s -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--differential", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=str, default="",
                    help="comma-separated arm labels; runs only those and "
                         "merges the result into artifacts/runs.json")
    args = ap.parse_args()
    if args.fidelity:
        v64 = run_fidelity(40, 20261112, np.float64)
        v32 = run_fidelity(40, 20261112, np.float32)
        (ART / "fidelity_both.json").write_text(json.dumps(
            {"float64": {k: v for k, v in v64.items() if k != "per_state"},
             "float32": {k: v for k, v in v32.items() if k != "per_state"}}, indent=1))
        print(json.dumps({"float64": {k: v for k, v in v64.items() if k != "per_state"},
                          "float32": {k: v for k, v in v32.items() if k != "per_state"}},
                         indent=1))
    if args.matrix:
        run_matrix(workers=args.workers)
    if args.differential:
        run_differential()
    if args.chunk:
        run_chunk([c.strip() for c in args.chunk.split(",") if c.strip()],
                  workers=args.workers)
    if args.analyse:
        analyse()