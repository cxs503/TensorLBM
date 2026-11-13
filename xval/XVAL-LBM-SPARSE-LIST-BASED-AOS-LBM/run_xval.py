#!/usr/bin/env python3
"""XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM -- cross-validation driver.

Cross-validates TensorLBM's port of the ``lssfau/lbmpy`` *sparse list-based AoS*
LBM storage scheme (cell list + per-cell/per-direction neighbour index array,
no-slip by index aliasing) against

  A  TensorLBM dense traversal (independent implementation, same discrete scheme),
  B  TensorLBM list-based traversal (+ TensorLBM-built index array),
  C  TensorLBM list-based traversal driven by the **OSS** index array
     (``lbmpy.sparse.SparseLbMapper.create_index_array``, executed in-process),
  R  analytic reference (plane Poiseuille profile / Taylor-Green decay).

Predeclared protocol: ``predeclared.json`` next to this file.

Usage
-----
    python run_xval.py [--out DIR] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm.sparse_list_aos import (  # noqa: E402
    C_D2Q9,
    C_D2Q9_LBMPY,
    FLUID,
    GHOST,
    WALL,
    SparseListLBM,
    build_index_array,
    dense_collide_stream,
    equilibrium,
    poiseuille_analytic,
    taylor_green_analytic,
    velocity_from,
    weights_for,
)

MAX_STEPS = 120_000
CHECK_EVERY = 200
STEADY_TOL = 1e-12


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def channel_flags(length: int, height: int) -> np.ndarray:
    """Channel: ``length`` periodic columns, ``height`` fluid rows, 1-cell walls.

    Following the OSS convention, the outer rows/columns of the flag grid are
    ghost/dummy cells (flag 0) -- ``SparseLbMapper.create_index_array`` requires
    them to be non-fluid so that the periodic wrap has somewhere to fold into.
    """
    flags = np.full((length + 2, height + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1] = FLUID
    flags[:, 0] = WALL
    flags[:, -1] = WALL
    return flags


def box_flags(length: int, height: int) -> np.ndarray:
    """Fully periodic box (ghost ring only)."""
    flags = np.full((length + 2, height + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1] = FLUID
    return flags


def channel_profile(sp: SparseListLBM, height: int) -> np.ndarray:
    """Mean u_x per fluid row (rows are uniform in a fully developed channel)."""
    _, ux, _ = sp.macroscopic()
    ys = sp.coords[:, 1].astype(int)
    return np.array([ux[ys == y].mean() for y in range(1, height + 1)])


def dense_profile(f: np.ndarray, force=(0.0, 0.0)) -> np.ndarray:
    _, ux, _ = velocity_from(f.transpose(1, 2, 0), force)
    return ux.mean(axis=0)


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def run_sparse_arm(flags, tau, force, *, arm="B", height=None, max_steps=MAX_STEPS):
    """Iterate the list-based arm to steady state; return profile + diagnostics."""
    index_array = None
    stencil = C_D2Q9
    if arm == "C":
        index_array = oss_index_array(flags)
        stencil = C_D2Q9_LBMPY
    sp = SparseListLBM(flags, tau=tau, force=force, index_array=index_array, stencil=stencil)
    n_cells = sp.n_cells
    mass0 = float(sp.populations().sum())
    mass_err = 0.0
    prev = None
    it = 0
    for it in range(1, max_steps + 1):
        sp.step()
        mass = float(sp.populations().sum())
        mass_err = max(mass_err, abs(mass - mass0) / abs(mass0))
        if it % CHECK_EVERY == 0:
            _, ux, _ = sp.macroscopic()
            if prev is not None and np.max(np.abs(ux - prev)) < STEADY_TOL:
                break
            prev = ux.copy()
    return {
        "arm": arm,
        "n_cells": int(n_cells),
        "steps": it,
        "converged": it < max_steps,
        "mass_error": mass_err,
        "state": sp,
        "profile": channel_profile(sp, height) if height else None,
        "profile_raw": None,
    }


def run_dense_arm(length, height, tau, force, *, max_steps=MAX_STEPS):
    """Iterate the dense arm to steady state (same discrete scheme)."""
    w = weights_for(C_D2Q9)
    f = np.repeat(w[:, None, None], length, axis=1).repeat(height, axis=2).astype(np.float64)
    mass0 = float(f.sum())
    mass_err = 0.0
    prev = None
    it = 0
    for it in range(1, max_steps + 1):
        f = dense_collide_stream(f, tau=tau, force=force, periodic=(True, False))
        mass_err = max(mass_err, abs(float(f.sum()) - mass0) / mass0)
        if it % CHECK_EVERY == 0:
            ux = dense_profile(f, force)
            if prev is not None and np.max(np.abs(ux - prev)) < STEADY_TOL:
                break
            prev = ux.copy()
    return {
        "arm": "A",
        "n_cells": int(length * height),
        "steps": it,
        "converged": it < max_steps,
        "mass_error": mass_err,
        "state": f,
        "profile": dense_profile(f, force),
    }


def oss_index_array(flags):
    """Index array produced by the OSS implementation itself (requires lbmpy)."""
    import lbmpy
    from lbmpy.sparse import SparseLbMapper
    from lbmpy.stencils import LBStencil

    mapper = SparseLbMapper(
        LBStencil(lbmpy.enums.Stencil.D2Q9),
        flags.copy(),
        fluid_flag=FLUID,
        no_slip_flag=WALL,
        other_boundary_mask=0,
    )
    return mapper.create_index_array(ghost_layers=1)


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
def case_poiseuille(height, tau=0.8, umax_target=0.03, length=4, quick=False):
    """Plane Poiseuille flow: body force, halfway walls, periodic streamwise."""
    nu = (tau - 0.5) / 3.0
    force_x = 8.0 * nu * umax_target / height**2
    flags = channel_flags(length, height)
    max_steps = 20_000 if quick else MAX_STEPS

    arms = {}
    arms["A"] = run_dense_arm(length, height, tau, (force_x, 0.0), max_steps=max_steps)
    for arm in ("B", "C"):
        arms[arm] = run_sparse_arm(flags, tau, (force_x, 0.0), arm=arm,
                                   height=height, max_steps=max_steps)

    y = np.arange(height)
    ref = poiseuille_analytic(y, height, force_x, nu)
    rows = {}
    for key, res in arms.items():
        prof = np.asarray(res["profile"], dtype=float)
        rows[key] = {
            "u_max": float(prof.max()),
            "linf_rel": float(np.abs(prof - ref).max() / ref.max()),
            "l2_rel": float(np.sqrt(np.mean((prof - ref) ** 2)) / np.sqrt(np.mean(ref**2))),
            "steps": int(res["steps"]),
            "converged": bool(res["converged"]),
            "mass_error": float(res["mass_error"]),
            "cells": int(res["n_cells"]),
        }
    scale = float(ref.max())
    pa = np.asarray(arms["A"]["profile"])
    pb = np.asarray(arms["B"]["profile"])
    pc = np.asarray(arms["C"]["profile"])
    diffs = {
        "A_vs_B": float(np.abs(pa - pb).max() / scale),
        "A_vs_C": float(np.abs(pa - pc).max() / scale),
        "B_vs_C": float(np.abs(pb - pc).max() / scale),
    }
    ma = float(arms["A"]["mass_error"])
    mb = max(float(arms["B"]["mass_error"]), float(arms["C"]["mass_error"]))
    return {
        "case": "poiseuille",
        "params": {
            "length": length,
            "height": height,
            "tau": tau,
            "nu": nu,
            "force_x": force_x,
            "u_max_target": umax_target,
            "u_max_analytic": float(ref.max()),
            "Re": umax_target * height / nu,
            "Ma": umax_target / np.sqrt(1.0 / 3.0),
            "cells_fluid_B": int(arms["B"]["n_cells"]),
            "cells_fluid_A": int(arms["A"]["n_cells"]),
            "cells_bounding_box": int(length * height),
        },
        "arms": rows,
        "diffs": diffs,
        "mass_error": {"dense": ma, "sparse": mb},
        "reference": ref.tolist(),
        "profile_dense": pa.tolist(),
        "profile_sparse": pb.tolist(),
        "profile_oss_index": pc.tolist(),
    }


def case_taylor_green(size, tau=0.8, u0=0.02, frac=2.0, quick=False):
    """2-D Taylor-Green vortex: fully periodic, exact exponential decay.

    Measurement protocol (identical for all arms): the fundamental mode
    amplitude ``a(t) = <u_x, sin(kx)cos(ky)> / <sin,cos>^2`` is sampled every
    10 steps and a log-linear fit is taken over ``t >= skip`` inside a window of
    ``frac`` e-foldings, ``T = frac / (2 nu k^2)``.  Projection (rather than the
    rms speed) removes the acoustic background that the feq(rho=1,u)
    initialisation leaves behind; the window stops long before the amplitude
    reaches round-off.
    """
    nu = (tau - 0.5) / 3.0
    k = 2.0 * np.pi / size
    omega_analytic = 2.0 * nu * k * k
    steps = int(max(min(frac / omega_analytic, 4000.0), 40.0))
    if quick:
        steps = max(steps // 2, 40)
    skip = min(200, steps // 2)
    flags = box_flags(size, size)
    x = np.arange(size) + 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    ux0 = u0 * np.sin(k * X) * np.cos(k * Y)
    uy0 = -u0 * np.cos(k * X) * np.sin(k * Y)
    proj = np.outer(np.sin(k * x), np.cos(k * x))
    proj_norm = float((proj * proj).sum())

    results = {}
    for arm in ("B", "C"):
        index_array = oss_index_array(flags) if arm == "C" else None
        stencil = C_D2Q9_LBMPY if arm == "C" else C_D2Q9
        sp = SparseListLBM(flags, tau=tau, force=(0.0, 0.0), index_array=index_array,
                           stencil=stencil)
        ix = sp.coords[:, 0].astype(int) - 1
        iy = sp.coords[:, 1].astype(int) - 1
        sp.reset(rho=1.0, ux=ux0[ix, iy], uy=uy0[ix, iy])
        mass0 = float(sp.populations().sum())
        samples, mass_err = [], 0.0
        for it in range(1, steps + 1):
            sp.step()
            mass_err = max(mass_err, abs(float(sp.populations().sum()) - mass0) / mass0)
            if it % 10 == 0:
                _, vx, _ = sp.macroscopic()
                field = np.zeros((size, size))
                field[ix, iy] = vx
                samples.append((it, float((field * proj).sum() / proj_norm)))
        _, vx, _ = sp.macroscopic()
        field = np.zeros((size, size))
        field[ix, iy] = vx
        results[arm] = {"samples": samples, "field": field, "mass_error": mass_err,
                        "cells": int(sp.n_cells)}

    w = weights_for(C_D2Q9)
    f = np.ascontiguousarray(equilibrium(np.ones((size, size)), ux0, uy0).transpose(2, 0, 1))
    mass0 = float(f.sum())
    samples, mass_err = [], 0.0
    for it in range(1, steps + 1):
        f = dense_collide_stream(f, tau=tau, force=(0.0, 0.0), periodic=(True, True))
        mass_err = max(mass_err, abs(float(f.sum()) - mass0) / mass0)
        if it % 10 == 0:
            _, vx, _ = velocity_from(f.transpose(1, 2, 0), (0.0, 0.0))
            samples.append((it, float((vx * proj).sum() / proj_norm)))
    _, vx, _ = velocity_from(f.transpose(1, 2, 0), (0.0, 0.0))
    results["A"] = {"samples": samples, "field": vx, "mass_error": mass_err,
                    "cells": int(size * size)}

    def decay_rate(samples):
        t = np.array([s[0] for s in samples], dtype=float)
        m = np.array([s[1] for s in samples], dtype=float)
        sel = t >= skip
        return float(-np.polyfit(t[sel], np.log(m[sel]), 1)[0])

    amp_end = u0 * np.exp(-omega_analytic * steps)
    field_ref = amp_end * np.sin(k * X) * np.cos(k * Y)
    rows = {}
    for key, res in results.items():
        omega = decay_rate(res["samples"])
        rows[key] = {
            "decay_rate": omega,
            "decay_rel_err": float(abs(omega - omega_analytic) / omega_analytic),
            "field_linf_rel": float(np.abs(res["field"] - field_ref).max() / amp_end),
            "mass_error": float(res["mass_error"]),
            "cells": res["cells"],
        }
    fa = results["A"]["field"]
    fb = results["B"]["field"]
    fc = results["C"]["field"]
    scale = float(np.abs(field_ref).max())
    return {
        "case": "taylor_green",
        "params": {
            "size": size,
            "tau": tau,
            "nu": nu,
            "u0": u0,
            "k": k,
            "steps": steps,
            "skip": int(skip),
            "omega_analytic": float(omega_analytic),
            "e_foldings": float(steps * omega_analytic),
            "Re": u0 * size / nu,
            "Ma": u0 / np.sqrt(1.0 / 3.0),
            "cells_fluid": int(size * size),
        },
        "arms": rows,
        "diffs": {
            "A_vs_B_field": float(np.abs(fa - fb).max() / scale),
            "A_vs_C_field": float(np.abs(fa - fc).max() / scale),
            "B_vs_C_field": float(np.abs(fb - fc).max() / scale),
        },
        "decay_curve": {key: res["samples"] for key, res in results.items()},
        "final_amplitude_analytic": float(amp_end),
    }


def case_stride_mismatch(height=16, tau=0.8, steps=2000):
    """H4: feed OSS index array (SoA offsets) to a kernel that assumes AoS offsets."""
    nu = (tau - 0.5) / 3.0
    force_x = 8.0 * nu * 0.03 / height**2
    flags = channel_flags(4, height)
    good = SparseListLBM(flags, tau=tau, force=(force_x, 0.0))
    # 'aos' solver: offsets cell*Q+d, but the injected index array carries the
    # OSS (cell + d*N) offsets -> addresses the wrong populations.
    bad = SparseListLBM(flags, tau=tau, force=(force_x, 0.0), layout="aos")
    good_idx = good.index_array.copy()
    bad = SparseListLBM(flags, tau=tau, force=(force_x, 0.0), layout="aos",
                        index_array=good_idx[: bad.n_cells])
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            good.step()
            bad.step()
    pg = channel_profile(good, height)
    pb = channel_profile(bad, height)
    ref = poiseuille_analytic(np.arange(height), height, force_x, nu)
    with np.errstate(invalid="ignore"):
        pass
    return {
        "case": "stride_mismatch",
        "steps": steps,
        "u_max_consistent": float(pg.max()),
        "u_max_mismatched": float(pb.max()),
        "linf_rel_consistent": float(np.abs(pg - ref).max() / ref.max()),
        "linf_rel_mismatched": float(np.abs(pb - ref).max() / ref.max()),
        "diff_vs_consistent": float(np.abs(pb - pg).max() / ref.max())
        if np.all(np.isfinite(pb)) else float("inf"),
        "mismatched_finite": bool(np.all(np.isfinite(pb))),
        "mass_mismatched": float(abs(bad.populations().sum() - good.populations().sum())),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def mark(passed: bool) -> str:
    return "OK" if passed else "FAIL"


def print_tables(results):
    print("\n=== parameter table (lattice units, dimensionless) ===")
    print(f"{'case':<22}{'H/L':>6}{'tau':>6}{'nu':>8}{'G':>12}{'u_max':>9}{'Re':>8}{'Ma':>8}{'cells':>8}")
    for r in results["poiseuille"]:
        p = r["params"]
        print(f"{'poiseuille':<22}{p['height']:>6}{p['tau']:>6.2f}{p['nu']:>8.4f}"
              f"{p['force_x']:>12.3e}{p['u_max_analytic']:>9.5f}{p['Re']:>8.2f}{p['Ma']:>8.4f}"
              f"{p['cells_fluid_B']:>8d}")
    for r in results["taylor_green"]:
        p = r["params"]
        print(f"{'taylor_green':<22}{p['size']:>6}{p['tau']:>6.2f}{p['nu']:>8.4f}"
              f"{0.0:>12.3e}{p['u0']:>9.5f}{p['Re']:>8.2f}{p['Ma']:>8.4f}"
              f"{p['cells_fluid']:>8d}")

    print("\n=== Poiseuille: ref vs got (Linf relative to u_max_analytic) ===")
    print(f"{'H':>4} {'arm':>4} {'u_max':>10} {'ref':>10} {'Linf_rel':>10} {'L2_rel':>10} "
          f"{'steps':>7} {'conv':>5}  pass(<15%)")
    for r in results["poiseuille"]:
        h = r["params"]["height"]
        ref = r["params"]["u_max_analytic"]
        for arm in ("A", "B", "C"):
            a = r["arms"][arm]
            print(f"{h:>4} {arm:>4} {a['u_max']:>10.6f} {ref:>10.6f} {a['linf_rel']:>10.3e} "
                  f"{a['l2_rel']:>10.3e} {a['steps']:>7d} {str(a['converged']):>5}  "
                  f"{mark(a['linf_rel'] <= 0.15)}")
    print("\n=== Poiseuille: arm-to-arm differences (max|du|/u_max) ===")
    print(f"{'H':>4} {'A_vs_B':>12} {'A_vs_C':>12} {'B_vs_C':>12}  pass(<=1e-12)")
    for r in results["poiseuille"]:
        d = r["diffs"]
        print(f"{r['params']['height']:>4} {d['A_vs_B']:>12.3e} {d['A_vs_C']:>12.3e} "
              f"{d['B_vs_C']:>12.3e}  {mark(max(d.values()) <= 1e-12)}")

    print("\n=== Taylor-Green: measured vs analytic decay rate (2 nu k^2) ===")
    print(f"{'L':>4} {'T':>6} {'arm':>4} {'omega_got':>12} {'omega_ref':>12} {'rel':>10} "
          f"{'field_Linf':>11}  pass(<15%)")
    for r in results["taylor_green"]:
        ref = r["params"]["omega_analytic"]
        for arm in ("A", "B", "C"):
            a = r["arms"][arm]
            print(f"{r['params']['size']:>4} {r['params']['steps']:>6} {arm:>4} "
                  f"{a['decay_rate']:>12.5e} {ref:>12.5e} {a['decay_rel_err']:>10.3e} "
                  f"{a['field_linf_rel']:>11.3e}  "
                  f"{mark(a['decay_rel_err'] <= 0.15 and a['field_linf_rel'] <= 0.15)}")
    print("\n=== Taylor-Green: arm-to-arm field difference (max|du|/u_max) ===")
    for r in results["taylor_green"]:
        d = r["diffs"]
        print(f"L={r['params']['size']:>4} A_vs_B={d['A_vs_B_field']:.3e} "
              f"A_vs_C={d['A_vs_C_field']:.3e} B_vs_C={d['B_vs_C_field']:.3e}")

    print("\n=== stride-convention mismatch (H4) ===")
    s = results["stride_mismatch"]
    print(f"consistent SoA  : u_max={s['u_max_consistent']:.6f} Linf_rel={s['linf_rel_consistent']:.3e}")
    print(f"injected into AoS: u_max={s['u_max_mismatched']:.6f} Linf_rel={s['linf_rel_mismatched']:.3e}")
    print(f"max|du|/u_max between them = {s['diff_vs_consistent']:.3e} "
          f"-> {mark(s['diff_vs_consistent'] > 0.01)}")


def observed_order(errors, heights):
    """Observed order from an error sequence on successively refined grids."""
    out = []
    for i in range(len(errors) - 1):
        e1, e2 = errors[i], errors[i + 1]
        if e1 <= 0 or e2 <= 0:
            out.append(None)
            continue
        out.append(float(np.log(e1 / e2) / np.log(heights[i + 1] / heights[i])))
    return out


def make_figures(results, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    paths = []

    # 1: Poiseuille profile for the finest grid + error of every arm
    main = max(results["poiseuille"], key=lambda r: r["params"]["height"])
    h = main["params"]["height"]
    y = np.arange(h) + 0.5
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(main["reference"], y, "k-", lw=2, label="analytic")
    ax[0].plot(main["profile_dense"], y, "o", ms=3.5, mfc="none", label="A dense")
    ax[0].plot(main["profile_sparse"], y, "s", ms=3.5, mfc="none", label="B list-based (TL index)")
    ax[0].plot(main["profile_oss_index"], y, "x", ms=4, label="C OSS index array")
    ax[0].set_xlabel("$u_x$ [lattice]")
    ax[0].set_ylabel("y [cells]")
    ax[0].set_title(f"Plane Poiseuille, H={h}, $\\tau$={main['params']['tau']}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for key, style in (("profile_dense", "o-"), ("profile_sparse", "s--"), ("profile_oss_index", "x:")):
        err = (np.array(main[key]) - np.array(main["reference"])) / main["params"]["u_max_analytic"]
        ax[1].plot(err * 100.0, y, style, ms=3.5, label=key.replace("profile_", ""))
    ax[1].axvline(15, color="r", ls=":", lw=1, label="15% criterion")
    ax[1].axvline(-15, color="r", ls=":", lw=1)
    ax[1].set_xlabel("relative error to analytic [%]")
    ax[1].set_ylabel("y [cells]")
    ax[1].set_title("profile error (all arms)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figdir, "poiseuille_profile.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 2: grid convergence (log-log)
    hs = [r["params"]["height"] for r in results["poiseuille"]
          if r["case_meta"] == "convergence"]
    conv = [r for r in results["poiseuille"] if r["case_meta"] == "convergence"]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for arm, style in (("A", "o-"), ("B", "s--"), ("C", "x:")):
        errs = [max(r["arms"][arm]["linf_rel"], 1e-16) for r in conv]
        ax.loglog(hs, errs, style, label=f"arm {arm}")
    order = observed_order([max(r["arms"]["B"]["linf_rel"], 1e-16) for r in conv], hs)
    ax.set_xlabel("channel height H [cells]")
    ax.set_ylabel(r"$L_\infty$ error / $u_{max}$")
    ax.set_title("grid convergence (B: order " + ", ".join(
        "%.2f" % o if o is not None else "n/a" for o in order) + ")")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "poiseuille_convergence.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 3: Taylor-Green decay
    tg_case = max(results["taylor_green"], key=lambda r: r["params"]["size"])
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for key, style in (("A", "o"), ("B", "s--"), ("C", "x:")):
        s = np.array(tg_case["decay_curve"][key], dtype=float)
        ax.semilogy(s[:, 0], s[:, 1], style, ms=3, label=f"arm {key}")
    tg = tg_case["params"]
    t = np.linspace(0, tg["steps"], 50)
    u0 = tg["u0"]
    ax.semilogy(t, u0 * np.exp(-tg["omega_analytic"] * t), "k-", lw=1.5, label="analytic")
    ax.set_xlabel("steps")
    ax.set_ylabel(r"fundamental mode amplitude $a(t)$")
    ax.set_title(f"Taylor-Green L={tg['size']}: decay rate "
                 f"{tg['omega_analytic']:.3e} (analytic)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "taylor_green_decay.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    results = {"anchor": "XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM", "poiseuille": [],
               "taylor_green": []}

    # ---- H1: index-array integer equality with the OSS implementation ----- #
    flags = channel_flags(4, 16)
    coords, mine = build_index_array(flags, stencil=C_D2Q9_LBMPY)
    oss = oss_index_array(flags)
    h1 = {
        "equal": bool(np.array_equal(mine, oss)),
        "shape": list(mine.shape),
        "dtype": str(mine.dtype),
        "n_cells": int(mine.shape[0]),
        "max_abs_diff": int(np.abs(mine.astype(np.int64) - oss.astype(np.int64)).max()),
        "oss_version": __import__("lbmpy").__version__,
    }
    print(f"[H1] TensorLBM index array == OSS index array: {h1['equal']} "
          f"(shape {h1['shape']}, dtype {h1['dtype']}, max|diff| {h1['max_abs_diff']})")
    results["h1_index_equality"] = h1

    # ---- main + grid convergence sweep ------------------------------------ #
    heights = [8, 16, 32] if args.quick else [8, 16, 32, 64]
    for height in heights:
        r = case_poiseuille(height, quick=args.quick)
        r["case_meta"] = "convergence"
        print(f"[Poiseuille] H={height:<3d} Linf_rel A/B/C = "
              f"{r['arms']['A']['linf_rel']:.3e} / {r['arms']['B']['linf_rel']:.3e} / "
              f"{r['arms']['C']['linf_rel']:.3e}   B vs C diff "
              f"{r['diffs']['B_vs_C']:.2e}   steps={r['arms']['B']['steps']}")
        results["poiseuille"].append(r)

    # ---- parameter sensitivity: tau and force magnitude ------------------- #
    for tau in (0.7, 1.0, 1.2):
        r = case_poiseuille(32, tau=tau, quick=args.quick)
        r["case_meta"] = "tau_sensitivity"
        print(f"[Poiseuille tau={tau}] Linf_rel B = {r['arms']['B']['linf_rel']:.3e}  "
              f"B vs C = {r['diffs']['B_vs_C']:.2e}")
        results["poiseuille"].append(r)
    for scale, tag in ((0.3, "0.3x"), (3.0, "3x")):
        r = case_poiseuille(32, umax_target=0.03 * scale, quick=args.quick)
        r["case_meta"] = f"force_sensitivity_{tag}"
        print(f"[Poiseuille force {tag}] Linf_rel B = {r['arms']['B']['linf_rel']:.3e}  "
              f"Ma={r['params']['Ma']:.3f}  B vs C = {r['diffs']['B_vs_C']:.2e}")
        results["poiseuille"].append(r)

    # ---- Taylor-Green ----------------------------------------------------- #
    for size in ((16, 32) if args.quick else (16, 32, 64)):
        r = case_taylor_green(size, quick=args.quick)
        print(f"[TG L={size}] window {r['params']['steps']} steps "
              f"({r['params']['e_foldings']:.2f} e-foldings)  decay err A/B/C = "
              f"{r['arms']['A']['decay_rel_err']:.3e} / {r['arms']['B']['decay_rel_err']:.3e} / "
              f"{r['arms']['C']['decay_rel_err']:.3e}   field Linf A/C = "
              f"{r['arms']['A']['field_linf_rel']:.3e} / {r['arms']['C']['field_linf_rel']:.3e}")
        results["taylor_green"].append(r)

    # ---- H4: stride mismatch ---------------------------------------------- #
    results["stride_mismatch"] = case_stride_mismatch()
    print(f"[H4] stride mismatch: {results['stride_mismatch']}")

    # ---- verdicts --------------------------------------------------------- #
    linf_b = [r["arms"]["B"]["linf_rel"] for r in results["poiseuille"]
              if r["case_meta"] == "convergence" and r["params"]["height"] >= 16]
    linf_a = [r["arms"]["A"]["linf_rel"] for r in results["poiseuille"]
              if r["case_meta"] == "convergence" and r["params"]["height"] >= 16]
    diffs = [max(r["diffs"].values()) for r in results["poiseuille"]]
    tg_err = [max(r["arms"][a]["decay_rel_err"] for a in "ABC") for r in results["taylor_green"]]
    tg_field = [max(r["arms"][a]["field_linf_rel"] for a in "ABC") for r in results["taylor_green"]]
    tg_diff = [max(r["diffs"].values()) for r in results["taylor_green"]]
    diffs = diffs + tg_diff
    mass = max(max(r["mass_error"].values()) for r in results["poiseuille"])
    verdict = {
        "T1_index_equality": {"got": h1["equal"], "pass": bool(h1["equal"])},
        "T2_sparse_vs_dense": {"got": max(diffs), "threshold": 1e-12,
                               "pass": bool(max(diffs) <= 1e-12)},
        "T3_poiseuille_vs_analytic": {"got": max(max(linf_a), max(linf_b)), "threshold": 0.15,
                                      "pass": bool(max(max(linf_a), max(linf_b)) <= 0.15)},
        "T4_taylor_green_decay": {"got": max(tg_err), "threshold": 0.15,
                                  "pass": bool(max(tg_err) <= 0.15)},
        "T7_taylor_green_field": {"got": max(tg_field), "threshold": 0.15,
                                  "pass": bool(max(tg_field) <= 0.15)},
        "T5_mass_conservation": {"got": mass, "threshold": 1e-12,
                                 "pass": bool(mass <= 1e-12)},
        "T6_aos_stride_mismatch": {
            "got": results["stride_mismatch"]["diff_vs_consistent"], "threshold": 0.01,
            "pass": bool(results["stride_mismatch"]["diff_vs_consistent"] > 0.01)},
    }
    results["verdict"] = verdict
    results["runtime_s"] = round(time.time() - t0, 1)

    print_tables(results)
    print("\n=== verdict vs predeclared thresholds ===")
    for key, v in verdict.items():
        print(f"  {key:<26} got={v['got']!r:<24} {mark(v['pass'])}")

    figs = make_figures(results, os.path.join(out, "figures"))
    results["figures"] = [os.path.relpath(p, REPO) for p in figs]

    # JSON-friendly copy (drop solver state objects)
    slim = json.loads(json.dumps(results, default=lambda o: None))
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(slim, fh, indent=1, sort_keys=True)

    with open(os.path.join(out, "tables.md"), "w") as fh:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_tables(results)
        fh.write("# XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM -- generated tables\n\n")
        fh.write("```\n" + buf.getvalue() + "```\n")

    all_pass = all(v["pass"] for v in verdict.values())
    print(f"\noverall: {'PASS' if all_pass else 'FAIL'}  ({results['runtime_s']} s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())