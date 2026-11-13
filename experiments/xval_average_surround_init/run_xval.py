#!/usr/bin/env python
"""XVAL-LBM-AVERAGE-SURROUND-INIT — TensorLBM birth init vs OSS ``average_surround``.

Decision
--------
Does TensorLBM's birth state for a newly created free-surface INTERFACE cell
agree with the OSS ``Maarten-vd-Sande/lbm`` ``average_surround``
(``lbm/update_types.py``, commit 573a01c) once TensorLBM exposes an explicit
``init_mode``, and where does the historical ``fixed_rho`` birth deviate?

Alternatives
------------
* ``fixed_rho``        — equilibrium at ``rho_liquid`` + neighbour-averaged u
                         (TensorLBM historical conversion-site birth);
* ``average_surround`` — equilibrium at neighbour-averaged rho *and* u
                         (OSS semantics, this work);
* OSS literal          — per-cell ``float64`` transcription of the OSS body,
                         the ``ref`` side of every comparison.

Metrics
-------
``rel_rho``, ``rel_u``, ``rel_f`` = relative error of the born-cell density,
velocity and PDF vector against the OSS literal reference.  System level:
liquid-inventory drift (reference value 0) and front position / state
divergence between modes.  Threshold: 0.15 (owner precision standard).

Repro
-----
    PYTHONPATH=src python experiments/xval_average_surround_init/run_xval.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm import free_surface_init_modes as im
from tensorlbm.d2q9 import C as C2, W as W2, equilibrium
from tensorlbm.free_surface_lbm_2d import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)

OUT = Path(__file__).parent / "artifacts"
THRESHOLD = 0.15
OSS_SOURCE = "Maarten-vd-Sande/lbm @573a01c lbm/update_types.py::average_surround"

W64 = np.asarray(W2, dtype=np.float64)
C64 = np.asarray(C2, dtype=np.float64)


# ---------------------------------------------------------------------------
# REF: literal per-cell transcription of the OSS body (float64)
# ---------------------------------------------------------------------------


def oss_set_equi(rho: float, ux: float, uy: float) -> np.ndarray:
    """D2Q9 equilibrium, written independently of the torch kernel."""
    cu = C64[:, 0] * ux + C64[:, 1] * uy
    usq = ux * ux + uy * uy
    return W64 * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def oss_average_surround(ix, iy, rho_prev, u_prev, active, shifts=im.D2Q9_SHIFTS):
    """``mass=0; rho=0; u=0;`` then average over FLUID|INTERFACE neighbours."""
    rho = 0.0
    ux = 0.0
    uy = 0.0
    c = 0
    ny, nx = rho_prev.shape
    for dx, dy in shifts:
        jx, jy = ix + dx, iy + dy  # OSS probes ``ix + v[n, 0]``
        if not (0 <= jx < nx and 0 <= jy < ny):
            continue
        if active[jy, jx]:
            c += 1
            rho += rho_prev[jy, jx]
            ux += u_prev[0][jy, jx]
            uy += u_prev[1][jy, jx]
    if c > 0:
        rho /= c
        ux /= c
        uy /= c
    return rho, ux, uy, c


# ---------------------------------------------------------------------------
# reference birth state on a controlled lattice patch
# ---------------------------------------------------------------------------


def build_patch(n=13, rho_left=1.0, rho_slope=0.0, u_amp=0.02, interface_col=None):
    """Liquid block (x < split) | one INTERFACE column | GAS, density ramp."""
    split = n // 2 if interface_col is None else interface_col
    flags = np.full((n, n), GAS, dtype=np.int8)
    flags[:, :split] = LIQUID
    flags[:, split] = INTERFACE
    active = flags != GAS
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    # The ramp varies along y *and* x so the 3 active neighbours of an
    # interface-adjacent GAS cell carry genuinely different densities: this is
    # what separates an averaging birth from a fixed-density birth.
    rho = np.where(
        active,
        rho_left + rho_slope * (ys[:, None] - split + 0.5 * (xs[None, :] - split)),
        0.0,
    )
    ux = np.where(active, u_amp * np.cos(2.0 * math.pi * xs[None, :] / n), 0.0) * np.ones((n, 1))
    uy = np.where(active, -u_amp * np.sin(2.0 * math.pi * xs[None, :] / n), 0.0) * np.ones((n, 1))
    return flags, active, rho, ux, uy, split


def got_birth_state(flags, rho, ux, uy, ix, iy, mode, rho_ref):
    """TensorLBM torch kernel evaluated on the same patch."""
    mask = torch.zeros(flags.shape, dtype=torch.bool)
    mask[iy, ix] = True
    f = equilibrium(
        torch.from_numpy(rho).float(),
        torch.from_numpy(ux).float(),
        torch.from_numpy(uy).float(),
    )
    out = im.birth_population(
        f,
        mask,
        rho_ref,
        f.new_tensor(ux).float(),
        f.new_tensor(uy).float(),
        None,
        mode=mode,
        equilibrium_fn=equilibrium,
        active=torch.from_numpy(flags != GAS),
        liquid_flag=LIQUID,
        interface_flag=INTERFACE,
        shifts=im.D2Q9_SHIFTS,
    )
    pop = out[:, iy, ix].numpy()
    rho_got = float(pop.sum())
    ux_got = float((pop * np.asarray(C2)[:, 0]).sum() / rho_got) if rho_got > 0 else 0.0
    uy_got = float((pop * np.asarray(C2)[:, 1]).sum() / rho_got) if rho_got > 0 else 0.0
    return rho_got, ux_got, uy_got, pop


def rel(a, b):
    scale = max(abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / scale


def operator_case(name, rho_left, rho_slope, u_amp, ix, iy, rho_ref=1.0, n=13):
    flags, active, rho, ux, uy, split = build_patch(
        n=n, rho_left=rho_left, rho_slope=rho_slope, u_amp=u_amp
    )
    ref_rho, ref_ux, ref_uy, c = oss_average_surround(ix, iy, rho, (ux, uy), active)
    ref_f = oss_set_equi(ref_rho, ref_ux, ref_uy)

    row = {
        "case": name,
        "birth_cell": [int(iy), int(ix)],
        "active_neighbours": int(c),
        "ref": {"rho": ref_rho, "ux": ref_ux, "uy": ref_uy},
        "modes": {},
    }
    for mode in ("fixed_rho", "average_surround"):
        g_rho, g_ux, g_uy, g_f = got_birth_state(flags, rho, ux, uy, ix, iy, mode, rho_ref)
        f_scale = max(float(np.abs(ref_f).max()), 1e-12)
        rel_f = float(np.abs(g_f - ref_f).max() / f_scale)
        row["modes"][mode] = {
            "rho": g_rho,
            "ux": g_ux,
            "uy": g_uy,
            "rel_rho": rel(g_rho, ref_rho) if ref_rho != 0 else None,
            "rel_u": rel(g_ux, ref_ux) if ref_ux != 0 else abs(g_ux - ref_ux),
            "rel_f_linf": rel_f,
            "passes_15pct": all(
                (v is None) or v < THRESHOLD for v in (rel_f,)
            ),
        }
    return row


# ---------------------------------------------------------------------------
# system level: dam break, both modes, two grids
# ---------------------------------------------------------------------------


def front_position(flags, mass):
    filled = (flags == LIQUID) | ((flags == INTERFACE) & (mass > 0.5))
    xs = torch.nonzero(filled.any(dim=0)).flatten()
    return float(xs.max().item()) if xs.numel() else 0.0


def _pair_divergence(grid, d, other, out):
    a, b = d["legacy"], d[other]
    diff = float(np.abs(a["_f"] - b["_f"]).max())
    scale = max(float(np.abs(a["_f"]).max()), 1e-12)
    out.append({
        "grid": grid,
        "mode": other,
        "linf_f_state_diff": diff,
        "rel_linf_f_state_diff": diff / scale,
        "front_legacy": a["front"],
        "front_mode": b["front"],
        "front_rel_diff": abs(a["front"] - b["front"]) / max(a["front"], 1e-12),
        "inventory_rel_diff": abs(a["inventory"] - b["inventory"]) / a["inventory"],
        "under_15pct": abs(a["front"] - b["front"]) / max(a["front"], 1e-12) < THRESHOLD,
    })


def run_dam_break(ny, nx, steps, mode, tau=0.8, gy=-1e-4):
    fill, solid = init_fill_rectangular_2d(
        ny, nx, column_width=nx // 4, column_height=(ny * 2) // 3, device="cpu"
    )
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags)
    rho0 = torch.where(flags == GAS, torch.zeros_like(fill), torch.ones_like(fill))
    f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
    inv0 = total_liquid_inventory_2d(f, mass, flags)
    led = {}
    t0 = time.time()
    history = []
    births = {"calls": 0, "cells": 0}

    import tensorlbm.free_surface_lbm_2d as fs2

    original = fs2._init_new_2d

    def counting_init_new_2d(*args, **kwargs):
        out = original(*args, **kwargs)
        mask = args[2]
        n = int(mask.sum().item())
        if n:
            births["calls"] += 1
            births["cells"] += n
        return out

    fs2._init_new_2d = counting_init_new_2d
    try:
        for step in range(steps):
            f, mass, flags = koerner_step_2d(
                f, mass, flags, solid, tau=tau, gy=gy, rho_liquid=1.0, init_mode=mode, ledger=led
            )
            if step % max(1, steps // 8) == 0 or step == steps - 1:
                history.append(
                    {"step": step + 1, "front": front_position(flags, mass),
                     "inventory": total_liquid_inventory_2d(f, mass, flags)}
                )
    finally:
        fs2._init_new_2d = original
    inv = total_liquid_inventory_2d(f, mass, flags)
    return {
        "grid": f"{ny}x{nx}",
        "steps": steps,
        "mode": mode,
        "runtime_s": round(time.time() - t0, 2),
        "inventory": inv,
        "inventory_drift_rel": abs(inv - inv0) / inv0,
        "mass_drift": led.get("mass_drift"),
        "front": front_position(flags, mass),
        "n_interface": int((flags == INTERFACE).sum()),
        "n_liquid": int((flags == LIQUID).sum()),
        "finite": bool(torch.isfinite(f).all()),
        "births": births,
        "history": history,
        "_f": f.contiguous().numpy().copy(),
        "_inv0": inv0,
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t_start = time.time()
    results: dict = {
        "experiment": "xval_average_surround_init",
        "oss_reference": OSS_SOURCE,
        "tensorlbm_impl": "src/tensorlbm/free_surface_init_modes.py",
        "threshold": THRESHOLD,
        "repro_cmd": "PYTHONPATH=src python experiments/xval_average_surround_init/run_xval.py",
    }

    # ---------------- 1. operator-level: ref (OSS literal) vs got -----------
    print("=" * 78)
    print("1. Operator XVAL: born-cell state, OSS literal ref vs TensorLBM got")
    print("=" * 78)
    cases = [
        operator_case("C1_homogeneous", 1.0, 0.0, 0.02, ix=7, iy=8),
        operator_case("C2_pressure_gradient", 1.0, 1e-3, 0.02, ix=7, iy=8),
        operator_case("C3_density_contrast", 1.0, 0.15, 0.02, ix=7, iy=8),
        operator_case("C4_multiphase_like", 0.3, 0.10, 0.05, ix=7, iy=8),
    ]
    for row in cases:
        print(f"\n  {row['case']}  (birth cell {row['birth_cell']}, "
              f"active neighbours {row['active_neighbours']})")
        print(f"    ref  OSS literal : rho={row['ref']['rho']:.6f} "
              f"u=({row['ref']['ux']:.6e}, {row['ref']['uy']:.6e})")
        for mode, m in row["modes"].items():
            print(f"    got  {mode:16s}: rho={m['rho']:.6f} u=({m['ux']:.6e}, {m['uy']:.6e})"
                  f"  rel_rho={m['rel_rho'] if m['rel_rho'] is None else round(m['rel_rho'], 3)}"
                  f"  rel_f={m['rel_f_linf']:.3e}  {'OK' if m['rel_f_linf'] < THRESHOLD else 'FAIL'}")
    results["operator_cases"] = cases

    # isolated-cell (documented deviation): OSS leaves rho = 0
    flags, active, rho, ux, uy, _ = build_patch(n=13, rho_left=1.0, rho_slope=0.0, u_amp=0.02)
    iso_rho, iso_ux, iso_uy, iso_c = oss_average_surround(11, 11, rho, (ux, uy), active)
    iso = {"active_neighbours": int(iso_c), "ref_oss_rho": iso_rho, "modes": {}}
    for mode in ("fixed_rho", "average_surround"):
        g_rho, g_ux, g_uy, g_f = got_birth_state(flags, rho, ux, uy, 11, 11, mode, 1.0)
        iso["modes"][mode] = {"rho": g_rho, "population": float(g_f.sum())}
    # historical zero fallback (``rho_init=None``): averaged density, no
    # isolated-cell rescue -> exactly the OSS zero-population state.
    g_rho, g_ux, g_uy, g_f = got_birth_state(
        flags, rho, ux, uy, 11, 11, "average_surround", 0.0
    )
    iso["modes"]["legacy_zero_fallback"] = {"rho": g_rho, "population": float(g_f.sum())}
    print(f"\n  C5_isolated_cell: OSS ref rho={iso['ref_oss_rho']:.6f} "
          f"(empty population -> fill = m/rho divides by the clamp); "
          f"TensorLBM average_surround rho={iso['modes']['average_surround']['rho']:.6f} "
          f"(documented deviation, deliberate)")
    results["isolated_cell_deviation"] = iso

    # ---------------- 2. sensitivity of the operator error ------------------
    print("\n" + "=" * 78)
    print("2. Sensitivity: velocity amplitude and density contrast")
    print("=" * 78)
    sens_u, sens_rho = [], []
    for amp in (0.005, 0.02, 0.08):
        row = operator_case(f"u_amp_{amp}", 1.0, 1e-3, amp, ix=7, iy=8)
        sens_u.append({"u_amp": amp,
                       "avg_rel_f": row["modes"]["average_surround"]["rel_f_linf"],
                       "fixed_rel_f": row["modes"]["fixed_rho"]["rel_f_linf"]})
        print(f"  u_amp={amp:<6}: avg rel_f={sens_u[-1]['avg_rel_f']:.3e}  "
              f"fixed rel_f={sens_u[-1]['fixed_rel_f']:.3e}")
    for slope in (0.0, 0.02, 0.05, 0.15):
        row = operator_case(f"slope_{slope}", 1.0, slope, 0.02, ix=7, iy=8)
        sens_rho.append({"rho_slope_per_cell": slope,
                         "avg_rel_rho": row["modes"]["average_surround"]["rel_rho"],
                         "fixed_rel_rho": row["modes"]["fixed_rho"]["rel_rho"]})
        print(f"  rho slope={slope:<6}: avg rel_rho={sens_rho[-1]['avg_rel_rho']:.3e}  "
              f"fixed rel_rho={sens_rho[-1]['fixed_rel_rho']:.3f}")
    results["sensitivity_velocity"] = sens_u
    results["sensitivity_density_contrast"] = sens_rho

    # ---------------- 3. system level: dam break ----------------------------
    print("\n" + "=" * 78)
    print("3. System level: 2D dam break, init_mode sweep (2 grids)")
    print("=" * 78)
    runs = []
    for (ny, nx, steps), modes in (((32, 64, 160), ("legacy", "fixed_rho", "average_surround")),
                                   ((64, 128, 80), ("legacy", "fixed_rho"))):
        for mode in modes:
            r = run_dam_break(ny, nx, steps, mode)
            runs.append(r)
            print(f"  {r['grid']:8s} {mode:16s} {r['runtime_s']:6.1f}s "
                  f"inv={r['inventory']:.6f} drift={r['inventory_drift_rel']:.2e} "
                  f"front={r['front']:.0f} iface={r['n_interface']} liq={r['n_liquid']} "
                  f"births={r['births']['cells']} finite={r['finite']}")

    # mode-to-mode divergence on the same grid
    by_grid: dict[str, dict] = {}
    for r in runs:
        by_grid.setdefault(r["grid"], {})[r["mode"]] = r
    divergence = []
    for grid, d in by_grid.items():
        for other in ("fixed_rho", "average_surround"):
            if "legacy" not in d or other not in d:
                continue
            _pair_divergence(grid, d, other, divergence)
    results["solver_mode_divergence"] = divergence
    for row in divergence:
        print(f"  mode divergence {row['grid']} legacy vs {row['mode']}: "
              f"|df|inf={row['rel_linf_f_state_diff']:.2e} "
              f"front {row['front_legacy']:.0f} vs {row['front_mode']:.0f} "
              f"({row['front_rel_diff']*100:.2f}%) "
              f"inventory {row['inventory_rel_diff']*100:.3f}%")
    if False:
        a, b = {}, {}
    results["solver_runs"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]

    # ---------------- 4. figures -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    labels, refs, avgs, fixeds = [], [], [], []
    for row in cases:
        labels.append(row["case"].split("_")[0])
        refs.append(row["ref"]["rho"])
        avgs.append(row["modes"]["average_surround"]["rho"])
        fixeds.append(row["modes"]["fixed_rho"]["rho"])
    idx = np.arange(len(labels))
    ax.bar(idx - 0.25, refs, 0.25, label="OSS literal ref", color="0.35")
    ax.bar(idx, avgs, 0.25, label="TensorLBM average_surround", color="steelblue")
    ax.bar(idx + 0.25, fixeds, 0.25, label="TensorLBM fixed_rho", color="coral")
    ax.set_xticks(idx, labels)
    ax.set_ylabel(r"born-cell density $\rho$")
    ax.set_title("(a) Born-cell density vs OSS reference")
    ax.legend(fontsize=7)

    ax = axes[1]
    slopes = [s["rho_slope_per_cell"] for s in sens_rho]
    ax.semilogy(slopes, [max(s["avg_rel_rho"], 1e-9) for s in sens_rho], "o-",
                label="average_surround")
    ax.semilogy(slopes, [max(s["fixed_rel_rho"], 1e-9) for s in sens_rho], "s-",
                label="fixed_rho")
    ax.axhline(THRESHOLD, color="k", ls="--", lw=1, label="15 % threshold")
    ax.set_xlabel(r"density slope per cell $\partial\rho/\partial x$")
    ax.set_ylabel(r"relative error of born-cell $\rho$")
    ax.set_title("(b) Density-contrast sensitivity")
    ax.legend(fontsize=7)

    ax = axes[2]
    for mode, style in (("legacy", "k-"), ("fixed_rho", "r--"), ("average_surround", "b:")):
        d = by_grid.get("32x64", {}).get(mode)
        if not d:
            continue
        h = np.array([[p["step"], p["front"]] for p in d["history"]], dtype=float)
        ax.plot(h[:, 0], h[:, 1], style, lw=1.4, label=mode)
    ax.set_xlabel("step")
    ax.set_ylabel("front position [cells]")
    ax.set_title("(c) Dam-break front, 32x64")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig_path = OUT / "xval_ref_vs_got.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---------------- 5. verdict -------------------------------------------
    op_avg = [row["modes"]["average_surround"]["rel_f_linf"] for row in cases]
    op_fixed = [row["modes"]["fixed_rho"]["rel_f_linf"] for row in cases]
    verdict = {
        "operator_average_surround_max_rel_f": max(op_avg),
        "operator_average_surround_pass": max(op_avg) < THRESHOLD,
        "operator_fixed_rho_max_rel_f": max(op_fixed),
        "operator_fixed_rho_pass": max(op_fixed) < THRESHOLD,
        "system_front_max_rel_diff": max((d["front_rel_diff"] for d in divergence), default=0.0),
        "system_inventory_max_drift_rel": max(r["inventory_drift_rel"] for r in runs),
        "all_finite": all(r["finite"] for r in runs),
        "overall_pass": bool(max(op_avg) < THRESHOLD and all(r["finite"] for r in runs)),
        "runtime_s": round(time.time() - t_start, 1),
        "figure": str(fig_path.relative_to(OUT.parent.parent.parent)),
    }
    results["verdict"] = verdict

    with open(OUT / "xval_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"VERDICT: operator max rel_f (average_surround) = {verdict['operator_average_surround_max_rel_f']:.3e}"
          f"  [{'PASS' if verdict['operator_average_surround_pass'] else 'FAIL'} <{THRESHOLD}]")
    print(f"         operator max rel_f (fixed_rho)        = {verdict['operator_fixed_rho_max_rel_f']:.3e}"
          f"  [{'PASS' if verdict['operator_fixed_rho_pass'] else 'FAIL'} <{THRESHOLD}]")
    print(f"         system front max rel diff (modes)       = {verdict['system_front_max_rel_diff']*100:.2f}%")
    print(f"         system inventory max drift              = {verdict['system_inventory_max_drift_rel']:.2e}")
    print(f"         runtime {verdict['runtime_s']}s -> {OUT/'xval_results.json'}")
    print("=" * 78)
    return results


if __name__ == "__main__":
    main()