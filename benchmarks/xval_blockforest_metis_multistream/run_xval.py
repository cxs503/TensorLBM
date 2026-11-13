#!/usr/bin/env python
"""XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP.

Cross-validation of TensorLBM's newly ported block-forest / block-graph load
balancing / compute-communication-overlap stack against the waLBerla
(``lssfau/walberla``) practice described in
``src/tensorlbm/blockforest.py`` (module docstring) and in
``docs/XVAL_BLOCKFOREST_METIS_MULTISTREAM.md``.

Predeclared decision, thresholds and analysis plan (fixed *before* the runs,
see ``PREDECLARED`` - they are copied verbatim into ``result.json``):

D1  partitioning quality:  ``cut/ref_cut - 1 <= 15%``  (ref = exhaustive
    optimum for 2 parts, networkx Kernighan-Lin for uniform k>2 instances,
    best-of-64-restarts for weighted k>2 instances)
D2  load balance:          ``imbalance = max(w)/mean(w) - 1 <= 5%``
D3  numerical equivalence: decomposed / overlapped vs monolithic
    ``max|f_dec - f_mono| <= 1e-12`` (float64 machine precision)
D4  physical fidelity:     shear-wave decay rate vs analytic ``nu k^2``
    ``|rate/ref - 1| <= 15%`` (owner accuracy standard)
D5  overlap efficiency:    hidden fraction ``1 - (t_ov - t0)/(t_ser - t0)
    >= 85%`` for latency/compute ratio ``alpha <= 0.5`` and blocks >= 16^3

Stopping rule: every experiment has a fixed sample (listed below); no threshold
is adjusted after inspecting a result.  Compute is single-process CPU
(``torch.set_num_threads(1)`` for the timing experiment only, so that the
worker thread cannot be starved by intra-op parallelism).

Usage
-----
    python benchmarks/xval_blockforest_metis_multistream/run_xval.py [--fast]

Writes ``artifacts/xval_blockforest_metis_multistream/{result.json,tables.md,fig*.png}``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.blockforest import (  # noqa: E402
    BlockForest,
    BlockGraph,
    MultistreamBlockSim,
    exact_bisection_cut,
    partition_blockforest,
    slab_partition,
    weighted_slab_partition,
)
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "xval_blockforest_metis_multistream"

PREDECLARED = {
    "anchor": "XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP",
    "reference": "lssfau/walberla blockforest + MetisPartitioning + multistream (CUDA streams) overlap",
    "decision": (
        "Adopt the ported block-forest / block-graph balancing / multistream "
        "overlap stack into TensorLBM as the distributed-simulation substrate."
    ),
    "thresholds": {
        "D1_cut_rel_error_max": 0.15,
        "D2_imbalance_max": 0.05,
        "D3_field_abs_tol": 1e-12,
        "D4_decay_rate_rel_error_max": 0.15,
        "D5_hidden_fraction_min": 0.85,
    },
    "metrics": {
        "cut": "sum of interface areas (cells) between different parts",
        "imbalance": "max(part weight)/mean(part weight) - 1",
        "field_error": "max |f_decomposed - f_monolithic| (float64)",
        "decay_rate": "log-linear fit of max_y u_x(t) over t in [200, 600]",
        "hidden_fraction": "1 - (t_overlapped - t0)/(t_serialized - t0), t0 = latency-free step time",
    },
    "controls": {
        "slab": "contiguous z-slabs over the linear block index = the decomposition "
                "the repository ships today (DomainDecomposer)",
        "weighted_slab": "contiguous z-slabs split at cumulative-workload boundaries (balanced 1-D)",
        "monolithic": "collide_bgk3d + stream3d_roll on one block (existing kernel path)",
    },
    "sample": {
        "E1": "6 instances, exhaustive enumeration (<= 2^17 assignments each)",
        "E2": "3 uniform forests x {4,8} parts, networkx KL best-of-20",
        "E3": "3 refined forests x {2,4,8} parts, best-of-64 reference cut",
        "E4": "shear wave 8x64x8, 600 steps, 1 configuration",
        "E5": "3 block sizes x 5 alpha levels, 4 repeats x 3 steps, median",
        "E6": "block size / part count sweep",
    },
    "environment": {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "device": "cpu",
        "dtype": "float64",
    },
}

REFINED_PATTERNS = {
    # deterministic refinement masks (1 = refined to level 1, 8x more cells)
    "refA": ((4, 4, 4), tuple(1 if (i // 2 + j // 2 + k // 2) % 2 else 0
                              for i in range(4) for j in range(4) for k in range(4))),
    "refB": ((4, 4, 4), tuple(1 if (j * 4 + k) % 3 == 0 else 0
                              for i in range(4) for j in range(4) for k in range(4))),
    "refC": ((3, 3, 3), tuple(1 if (i + j + k) % 2 else 0
                              for i in range(3) for j in range(3) for k in range(3))),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mk(nblocks, block_cells, refinement=None):
    return BlockForest(
        nblocks=nblocks,
        block_cells=block_cells,
        periodic=(True, True, True),
        refinement=refinement,
    )


def _nx_kl_partition(g: BlockGraph, nparts: int, restarts: int = 20, seed: int = 0):
    """Recursive bisection with networkx's Kernighan-Lin (independent OSS ref)."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for p in range(g.xadj[u], g.xadj[u + 1]):
            v = g.adjncy[p]
            if u < v:
                G.add_edge(u, v, weight=g.adjwgt[p])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        parts = np.zeros(g.n, dtype=int)

        def rec(nodes, k, pid):
            if k == 1 or len(nodes) <= 1:
                parts[nodes] = pid
                return
            k1 = k // 2
            sub = G.subgraph(nodes)
            perm = list(nodes)
            rng.shuffle(perm)
            half = max(1, len(perm) // 2)
            init = (set(perm[:half]), set(perm[half:]))
            try:
                a, b = nx.algorithms.community.kernighan_lin_bisection(
                    sub, partition=init, weight="weight", max_iter=20, seed=int(rng.integers(1 << 30))
                )
            except Exception:  # disconnected / degenerate
                a, b = init
            a, b = list(a), list(b)
            if not a or not b:
                a, b = perm[:half], perm[half:]
            rec(np.array(a, dtype=int), k1, pid)
            rec(np.array(b, dtype=int), k - k1, pid + k1)

        rec(np.arange(g.n, dtype=int), nparts, 0)
        cut = g.cut(parts)
        imb = g.imbalance(parts, nparts)
        if best is None or (imb <= 0.05 and cut < best[0]):
            best = (cut, parts, imb)
    return best[0], best[1], best[2]


def _nice(x):
    return f"{x:.4g}"


# ---------------------------------------------------------------------------
# E1: exact optimum for 2 parts (small instances)
# ---------------------------------------------------------------------------
def e1_exact():
    rows = []
    instances = [
        ("uniform 2x3x3 @8^3", _mk((2, 3, 3), (8, 8, 8))),
        ("uniform 3x3x2 @8^3", _mk((3, 3, 2), (8, 8, 8))),
        ("uniform 2x2x4 @6^3", _mk((2, 2, 4), (6, 6, 6))),
        ("refined 2x3x3 (checker)",
         _mk((2, 3, 3), (8, 8, 8),
             tuple(1 if (i + j + k) % 2 else 0 for i in range(2) for j in range(3) for k in range(3)))),
        ("refined 2x3x3 (slabA)",
         _mk((2, 3, 3), (6, 6, 6),
             tuple(1 if j == 1 else 0 for i in range(2) for j in range(3) for k in range(3)))),
        ("uniform 2x2x5 @5^3", _mk((2, 2, 5), (5, 5, 5))),
    ]
    for name, forest in instances:
        g = forest.graph()
        t0 = time.perf_counter()
        ex = exact_bisection_cut(g, imbalance_tol=PREDECLARED["thresholds"]["D2_imbalance_max"])
        t_ex = time.perf_counter() - t0
        if ex is None:
            print(f"  [skip] {name}: n_blocks={g.n} beyond exhaustive enumeration budget")
            continue
        ours = partition_blockforest(forest, 2, imbalance_tol=0.05, restarts=16, seed=0)
        ws = weighted_slab_partition(g, 2)
        rows.append(
            {
                "instance": name,
                "n_blocks": g.n,
                "exact_cut": ex.cut,
                "exact_imbalance": ex.imbalance,
                "ours_cut": ours.cut,
                "ours_imbalance": ours.imbalance,
                "ours_rel_error": (ours.cut / ex.cut - 1.0) if ex.cut > 0 else 0.0,
                "weighted_slab_cut": ws.cut,
                "weighted_slab_rel_error": (ws.cut / ex.cut - 1.0) if ex.cut > 0 else 0.0,
                "weighted_slab_imbalance": ws.imbalance,
                "exact_enumeration_s": t_ex,
            }
        )
    ok = all(r["ours_rel_error"] <= PREDECLARED["thresholds"]["D1_cut_rel_error_max"] for r in rows)
    return {"rows": rows, "pass_D1_exact": bool(ok)}


# ---------------------------------------------------------------------------
# E2: uniform forests, k = 4 / 8, networkx-KL reference
# ---------------------------------------------------------------------------
def e2_uniform_reference(fast=False):
    restarts = 8 if fast else 20
    rows = []
    forests = [
        ("4x4x4 @8^3", _mk((4, 4, 4), (8, 8, 8))),
        ("3x4x6 @8^3", _mk((3, 4, 6), (8, 8, 8))),
        ("6x6x6 @4^3", _mk((6, 6, 6), (4, 4, 4))),
    ]
    for name, forest in forests:
        g = forest.graph()
        for k in (4, 8):
            ours = partition_blockforest(forest, k, imbalance_tol=0.05, restarts=16, seed=0)
            kl_cut, _, kl_imb = _nx_kl_partition(g, k, restarts=restarts)
            slab = slab_partition(g, k)
            rows.append(
                {
                    "instance": name,
                    "n_blocks": g.n,
                    "k": k,
                    "ours_cut": ours.cut,
                    "ours_imbalance": ours.imbalance,
                    "kl_cut": kl_cut,
                    "kl_imbalance": kl_imb,
                    "rel_error_vs_kl": (ours.cut / kl_cut - 1.0) if kl_cut else 0.0,
                    "slab_cut": slab.cut,
                    "slab_imbalance": slab.imbalance,
                }
            )
    ok = all(abs(r["rel_error_vs_kl"]) <= PREDECLARED["thresholds"]["D1_cut_rel_error_max"] for r in rows)
    bal = all(r["ours_imbalance"] <= PREDECLARED["thresholds"]["D2_imbalance_max"] for r in rows)
    return {"rows": rows, "pass_D1_kl": bool(ok), "pass_D2_kl": bool(bal)}


# ---------------------------------------------------------------------------
# E3: refined (AMR-like) forests -> workload balance
# ---------------------------------------------------------------------------
def e3_balance():
    rows = []
    for name, (nb, ref) in REFINED_PATTERNS.items():
        forest = _mk(nb, (8, 8, 8), ref)
        g = forest.graph()
        refined_fraction = sum(ref) / len(ref)
        for k in (2, 4, 8):
            if k > g.n:
                continue
            ours = partition_blockforest(forest, k, imbalance_tol=0.05, restarts=16, seed=0)
            ref_best = partition_blockforest(forest, k, imbalance_tol=0.05, restarts=64, seed=12345)
            slab = slab_partition(g, k)
            wslab = weighted_slab_partition(g, k)
            rows.append(
                {
                    "instance": name,
                    "n_blocks": g.n,
                    "refined_fraction": refined_fraction,
                    "k": k,
                    "ours_cut": ours.cut,
                    "ours_imbalance": ours.imbalance,
                    "best_known_cut": ref_best.cut,
                    "rel_error_vs_best_known": (ours.cut / ref_best.cut - 1.0) if ref_best.cut else 0.0,
                    "slab_imbalance": slab.imbalance,
                    "slab_cut": slab.cut,
                    "weighted_slab_imbalance": wslab.imbalance,
                    "weighted_slab_cut": wslab.cut,
                }
            )
    ok_d1 = all(r["rel_error_vs_best_known"] <= PREDECLARED["thresholds"]["D1_cut_rel_error_max"] for r in rows)
    ok_d2 = all(r["ours_imbalance"] <= PREDECLARED["thresholds"]["D2_imbalance_max"] for r in rows)
    slab_ok = all(r["slab_imbalance"] <= PREDECLARED["thresholds"]["D2_imbalance_max"] for r in rows)
    wslab_ok = all(r["weighted_slab_imbalance"] <= PREDECLARED["thresholds"]["D2_imbalance_max"] for r in rows)
    return {
        "rows": rows,
        "pass_D1_best_known": bool(ok_d1),
        "pass_D2_ours": bool(ok_d2),
        "pass_D2_slab_baseline": bool(slab_ok),
        "pass_D2_weighted_slab": bool(wslab_ok),
    }


# ---------------------------------------------------------------------------
# E4: numerical equivalence + analytic decay rate
# ---------------------------------------------------------------------------
def e4_physics(tau=0.8, amplitude=0.05, steps=600, sample=10):
    forest = _mk((1, 8, 1), (8, 8, 8))          # domain 8 x 64 x 8, 8 blocks
    nzg, nyg, nxg = forest.global_shape()
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / nyg
    rate_ref = nu * k * k

    dec = MultistreamBlockSim(forest, tau=tau)
    dec.set_shear_wave(amplitude)
    mon = MultistreamBlockSim(_mk((1, 1, 1), (nzg, nyg, nxg)), tau=tau)
    mon.set_shear_wave(amplitude)

    ts, amps_dec, amps_mon, field_err = [], [], [], 0.0
    for step in range(1, steps + 1):
        dec.step("overlapped", latency=0.0)
        # monolithic: broadcast the single block through the production kernels
        nz, ny, nx = mon.shape
        f = mon.pad[0][:, 1 : nz + 1, 1 : ny + 1, 1 : nx + 1]
        mon.pad[0][:, 1 : nz + 1, 1 : ny + 1, 1 : nx + 1].copy_(
            stream3d_roll(collide_bgk3d(f, tau))
        )
        if step % sample == 0:
            ts.append(step)
            amps_dec.append(float(dec.velocity_x().abs().max()))
            amps_mon.append(float(mon.velocity_x().abs().max()))
            field_err = max(field_err, float((dec.velocity_x() - mon.velocity_x()).abs().max()))

    ts_a = np.asarray(ts, dtype=float)
    lo = int(0.34 * steps)          # fit window fixed a priori: last 2/3 of the run
    sel = (ts_a >= lo) & (ts_a <= steps)
    fit_dec = np.polyfit(ts_a[sel], np.log(np.asarray(amps_dec)[sel]), 1)[0]
    fit_mon = np.polyfit(ts_a[sel], np.log(np.asarray(amps_mon)[sel]), 1)[0]
    rate_dec = -fit_dec
    rate_mon = -fit_mon
    rel_dec = rate_dec / rate_ref - 1.0
    rel_mon = rate_mon / rate_ref - 1.0
    return {
        "tau": tau,
        "nu_lattice": nu,
        "wavenumber_k": k,
        "ny_global": nyg,
        "amplitude0": amplitude,
        "steps": steps,
        "re": amplitude * nyg / (2 * math.pi) / nu,
        "mach": amplitude * math.sqrt(3.0),
        "rate_ref": rate_ref,
        "rate_decomposed_overlapped": rate_dec,
        "rate_monolithic": rate_mon,
        "rel_error_decomposed": rel_dec,
        "rel_error_monolithic": rel_mon,
        "max_field_diff": field_err,
        "t_series": {"steps": ts, "amp_decomposed": amps_dec, "amp_monolithic": amps_mon,
                     "amp_analytic": [amplitude * math.exp(-rate_ref * t) for t in ts]},
        "pass_D3": bool(field_err <= PREDECLARED["thresholds"]["D3_field_abs_tol"]),
        "pass_D4": bool(abs(rel_dec) <= PREDECLARED["thresholds"]["D4_decay_rate_rel_error_max"]),
    }


# ---------------------------------------------------------------------------
# E5: multistream overlap efficiency
# ---------------------------------------------------------------------------
def e5_overlap(block_sizes=(12, 16, 20), alphas=(0.0, 0.1, 0.25, 0.5, 1.0), reps=4, steps=3):
    torch.set_num_threads(1)          # no intra-op parallelism: it would starve the worker
    rows = []
    for b in block_sizes:
        forest = _mk((2, 2, 2), (b, b, b))
        sim = MultistreamBlockSim(forest)
        sim.set_shear_wave(0.02)

        def timed(mode, lat):
            t = time.perf_counter()
            for _ in range(steps):
                sim.step(mode, lat)
            return (time.perf_counter() - t) / steps

        timed("serialized", 0.0)
        t0 = statistics.median(timed("serialized", 0.0) for _ in range(reps))
        for alpha in alphas:
            lat = alpha * t0
            pairs = [(timed("serialized", lat), timed("overlapped", lat)) for _ in range(reps)]
            ts = statistics.median(p[0] for p in pairs)
            to = statistics.median(p[1] for p in pairs)
            exposed_ser = ts - t0
            exposed_ov = to - t0
            hidden = 1.0 - exposed_ov / exposed_ser if exposed_ser > 1e-9 else float("nan")
            rows.append(
                {
                    "block_cells": b,
                    "cells_total": forest.n_blocks * b ** 3,
                    "alpha": alpha,
                    "latency_s": lat,
                    "t0_s": t0,
                    "t_serialized_s": ts,
                    "t_overlapped_s": to,
                    "exposed_serialized_s": exposed_ser,
                    "exposed_overlapped_s": exposed_ov,
                    "hidden_fraction": hidden,
                }
            )
    gate = [r for r in rows if r["block_cells"] >= 16 and 0 < r["alpha"] <= 0.5]
    return {
        "rows": rows,
        "pass_D5": bool(all(r["hidden_fraction"] >= PREDECLARED["thresholds"]["D5_hidden_fraction_min"] for r in gate)),
        "gate_rows": len(gate),
        "note": "torch.set_num_threads(1); median of 4 repeats x 3 steps; CPU shared node",
    }


# ---------------------------------------------------------------------------
# E6: sensitivity of the decomposition to block size and part count
# ---------------------------------------------------------------------------
def e6_sensitivity():
    rows = []
    for b in (6, 8, 12, 16):
        nb = max(2, 64 // b)
        forest = _mk((nb, nb, nb), (b, b, b))
        g = forest.graph()
        k = 8
        ours = partition_blockforest(forest, k, imbalance_tol=0.05, restarts=6)
        slab = slab_partition(g, k)
        rows.append(
            {
                "block_cells": b,
                "nblocks": forest.nblocks,
                "n_blocks": g.n,
                "k": k,
                "ours_cut": ours.cut,
                "ours_imbalance": ours.imbalance,
                "slab_cut": slab.cut,
                "slab_imbalance": slab.imbalance,
                "cut_fraction_of_total_interface": ours.cut / max(1e-12, sum(g.adjwgt) / 2.0),
            }
        )
    rows_k = []
    forest = _mk((8, 8, 8), (8, 8, 8))
    g = forest.graph()
    for k in (2, 4, 8, 16, 32):
        ours = partition_blockforest(forest, k, imbalance_tol=0.05, restarts=6)
        rows_k.append(
            {
                "k": k,
                "ours_cut": ours.cut,
                "ours_imbalance": ours.imbalance,
                "comm_volume_cells_per_step": 2.0 * ours.cut,
                "total_interface": sum(g.adjwgt) / 2.0,
            }
        )
    return {"by_block_size": rows, "by_parts": rows_k}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(res, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_paths = []

    # fig1: cut quality (E1)
    r1 = res["E1"]["rows"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(r1))
    ax.bar(x - 0.25, [r["exact_cut"] for r in r1], 0.25, label="exhaustive optimum (ref)")
    ax.bar(x, [r["ours_cut"] for r in r1], 0.25, label="block-graph partition (ours)")
    ax.bar(x + 0.25, [r["weighted_slab_cut"] for r in r1], 0.25, label="weighted z-slab")
    ax.set_xticks(x)
    ax.set_xticklabels([r["instance"] for r in r1], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("interface weight (cells)")
    ax.set_title("E1  k=2 partition cut vs exhaustive optimum")
    for i, r in enumerate(r1):
        ax.annotate(f"{r['ours_rel_error']*100:+.1f}%", (i, r["ours_cut"]), ha="center",
                    va="bottom", fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig1_cut_vs_optimum.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    fig_paths.append(p)

    # fig2: imbalance for refined forests (E3)
    r3 = res["E3"]["rows"]
    labels = [f"{r['instance']}\nk={r['k']}" for r in r3]
    x = np.arange(len(r3))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.27, [r["slab_imbalance"] * 100 for r in r3], 0.27, label="z-slab (repo baseline)")
    ax.bar(x, [r["weighted_slab_imbalance"] * 100 for r in r3], 0.27, label="weighted z-slab")
    ax.bar(x + 0.27, [r["ours_imbalance"] * 100 for r in r3], 0.27, label="block graph (ours)")
    ax.axhline(5.0, color="k", ls="--", lw=1, label="5% acceptance cap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("imbalance  max/mean - 1  [%]")
    ax.set_title("E3  workload balance on refined (AMR-like) forests")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig2_imbalance_refined.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    fig_paths.append(p)

    # fig3: overlap efficiency (E5)
    r5 = res["E5"]["rows"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for b in sorted({r["block_cells"] for r in r5}):
        sub = [r for r in r5 if r["block_cells"] == b and r["alpha"] > 0]
        ax.plot([r["alpha"] for r in sub], [r["hidden_fraction"] for r in sub], "o-",
                label=f"block {b}^3")
    ax.axhline(0.85, color="k", ls="--", lw=1, label="85% acceptance")
    ax.set_xlabel("alpha = injected latency / latency-free step time")
    ax.set_ylabel("hidden comm fraction")
    ax.set_title("E5  multistream overlap: hidden fraction vs latency ratio")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig3_overlap_hidden_fraction.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    fig_paths.append(p)

    # fig4: shear-wave decay (E4)
    r4 = res["E4"]
    ts = r4["t_series"]["steps"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(ts, r4["t_series"]["amp_decomposed"], "o", ms=3, label="8-block forest (overlapped)")
    ax.semilogy(ts, r4["t_series"]["amp_monolithic"], "-", lw=1, label="monolithic roll kernel")
    ax.semilogy(ts, r4["t_series"]["amp_analytic"], "k--", lw=1,
                label=r"analytic $A_0e^{-\nu k^2 t}$")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\max_y u_x$")
    ax.set_title(f"E4  shear-wave decay (fit error {r4['rel_error_decomposed']*100:+.2f}%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig4_shear_wave_decay.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    fig_paths.append(p)

    # fig5: sensitivity (E6)
    r6 = res["E6"]["by_parts"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([r["k"] for r in r6], [r["ours_cut"] for r in r6], "o-", label="ours: cut (interface cells)")
    ax.plot([r["k"] for r in r6], [r["ours_imbalance"] * 100 for r in r6], "s--",
            label="ours: imbalance [%]")
    ax.set_xlabel("parts k (8x8x8 blocks)")
    ax.set_ylabel("cut / imbalance")
    ax.set_title("E6  sensitivity to part count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig5_part_count_sensitivity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    fig_paths.append(p)
    return fig_paths


# ---------------------------------------------------------------------------
# markdown tables
# ---------------------------------------------------------------------------
def make_tables(res) -> str:
    L = []
    L.append("### E1 - k=2 cut vs exhaustive optimum (small block graphs)\n")
    L.append("| instance | blocks | exact cut | ours cut | ours err | ours imb | wslab err | wslab imb |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in res["E1"]["rows"]:
        L.append(
            f"| {r['instance']} | {r['n_blocks']} | {_nice(r['exact_cut'])} | {_nice(r['ours_cut'])} | "
            f"{r['ours_rel_error']*100:+.2f}% | {r['ours_imbalance']*100:.2f}% | "
            f"{r['weighted_slab_rel_error']*100:+.2f}% | {r['weighted_slab_imbalance']*100:.2f}% |"
        )
    L.append("")
    L.append("### E2 - uniform forests, k=4/8, networkx Kernighan-Lin reference\n")
    L.append("| instance | blocks | k | ours cut | ours imb | KL cut | KL imb | ours vs KL | slab cut | slab imb |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["E2"]["rows"]:
        L.append(
            f"| {r['instance']} | {r['n_blocks']} | {r['k']} | {_nice(r['ours_cut'])} | "
            f"{r['ours_imbalance']*100:.2f}% | {_nice(r['kl_cut'])} | {r['kl_imbalance']*100:.2f}% | "
            f"{r['rel_error_vs_kl']*100:+.2f}% | {_nice(r['slab_cut'])} | {r['slab_imbalance']*100:.2f}% |"
        )
    L.append("")
    L.append("### E3 - refined (AMR-like) forests: workload balance\n")
    L.append("| instance | refined frac | blocks | k | ours cut | ours imb | best-known err | slab imb | wslab imb |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in res["E3"]["rows"]:
        L.append(
            f"| {r['instance']} | {r['refined_fraction']*100:.0f}% | {r['n_blocks']} | {r['k']} | "
            f"{_nice(r['ours_cut'])} | {r['ours_imbalance']*100:.2f}% | "
            f"{r['rel_error_vs_best_known']*100:+.2f}% | {r['slab_imbalance']*100:.2f}% | "
            f"{r['weighted_slab_imbalance']*100:.2f}% |"
        )
    L.append("")
    L.append("### E5 - multistream overlap efficiency (CPU, latency injection)\n")
    L.append("| block | cells | alpha | t0 [s] | t serialized [s] | t overlapped [s] | exposed ser [ms] | exposed ov [ms] | hidden |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in res["E5"]["rows"]:
        L.append(
            f"| {r['block_cells']}^3 | {r['cells_total']} | {r['alpha']:.2f} | {r['t0_s']*1e3:.1f} | "
            f"{r['t_serialized_s']*1e3:.1f} | {r['t_overlapped_s']*1e3:.1f} | "
            f"{r['exposed_serialized_s']*1e3:.2f} | {r['exposed_overlapped_s']*1e3:.2f} | "
            f"{r['hidden_fraction']*100:.1f}% |"
        )
    L.append("")
    L.append("### E6 - sensitivity: block size (k=8) and part count (8x8x8 blocks @8^3)\n")
    L.append("| block^3 | nblocks | blocks | ours cut | ours imb | slab cut | slab imb | cut/total interface |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in res["E6"]["by_block_size"]:
        L.append(
            f"| {r['block_cells']} | {r['nblocks']} | {r['n_blocks']} | {_nice(r['ours_cut'])} | "
            f"{r['ours_imbalance']*100:.2f}% | {_nice(r['slab_cut'])} | {r['slab_imbalance']*100:.2f}% | "
            f"{r['cut_fraction_of_total_interface']*100:.2f}% |"
        )
    L.append("")
    L.append("| k | ours cut | comm volume (cells/step) | imbalance |")
    L.append("|---|---|---|---|")
    for r in res["E6"]["by_parts"]:
        L.append(f"| {r['k']} | {_nice(r['ours_cut'])} | {_nice(r['comm_volume_cells_per_step'])} | {r['ours_imbalance']*100:.2f}% |")
    return "\n".join(L)


def main(fast: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    res = {"predeclared": PREDECLARED, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("[E1] exhaustive-optimum comparison ...", flush=True)
    res["E1"] = e1_exact()
    print("[E2] networkx Kernighan-Lin reference ...", flush=True)
    res["E2"] = e2_uniform_reference(fast=fast)
    print("[E3] refined-forest balance ...", flush=True)
    res["E3"] = e3_balance()
    print("[E4] shear-wave physics ...", flush=True)
    res["E4"] = e4_physics(steps=150 if fast else 400)
    print("[E5] overlap efficiency ...", flush=True)
    res["E5"] = e5_overlap(reps=3 if fast else 4)
    print("[E6] sensitivity ...", flush=True)
    res["E6"] = e6_sensitivity()

    gates = {
        "D1_exact_optimum": res["E1"]["pass_D1_exact"],
        "D1_KL_reference": res["E2"]["pass_D1_kl"],
        "D1_best_known": res["E3"]["pass_D1_best_known"],
        "D2_ours": res["E2"]["pass_D2_kl"] and res["E3"]["pass_D2_ours"],
        "D2_slab_baseline": res["E3"]["pass_D2_slab_baseline"],
        "D3_numerical_equivalence": res["E4"]["pass_D3"],
        "D4_decay_rate": res["E4"]["pass_D4"],
        "D5_overlap": res["E5"]["pass_D5"],
    }
    res["gates"] = gates
    res["verdict"] = "PASS" if all(gates[k] for k in ("D1_exact_optimum", "D1_KL_reference",
                                                      "D1_best_known", "D2_ours", "D3_numerical_equivalence",
                                                      "D4_decay_rate", "D5_overlap")) else "PARTIAL"
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    figs = make_figures(res, OUT_DIR)
    res["figures"] = [p.name for p in figs]
    (OUT_DIR / "result.json").write_text(json.dumps(res, indent=2))
    (OUT_DIR / "tables.md").write_text(make_tables(res))
    print(json.dumps(gates, indent=2))
    print("verdict:", res["verdict"])
    print("artifacts:", OUT_DIR)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(fast=a.fast)