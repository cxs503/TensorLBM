#!/usr/bin/env python3
"""B14-ellipse: DIGITAL-GEOMETRY Stokes reference + error-budget decomposition.

Why this file exists
--------------------
`benchmarks/pending/poiseuille_3d_ellipse/` is an open verification debt: the
`s^Q` (area-equivalent single scale of the *continuous* ellipse) comparison gives
5.107% > 3% at a=20 (2.432% at a=40).  The pending README attributes the miss to
a *reference-geometry* mismatch (the staircase wall is anisotropic), not to a
solver defect.  That attribution was never tested against an independent
reference.  This module supplies one:

    the Stokes/Poiseuille solution of the *digital* cross-section that the LBM
    actually integrates, i.e. the discrete problem

        sum_{nbrs} (u_nb - u_c) = -G/nu ,     u_wall = 0 at the mid-point
                                              between a fluid cell and a solid
                                              neighbour (half-way bounce-back
                                              no-slip locus, 0.5 lattice units).

That closure is exactly the momentum-flux balance implied by half-way bounce-back
each solid direction contributes -2 u_c instead of +u_nb).  The solve is a sparse
direct solve to machine precision, so it carries **no discretisation error of its
own** (only the geometry statement is approximate).

Two facts make this a *reference* rather than a fit:
  * For fully developed flow in a straight duct the convective term vanishes
    identically (v=w=0, du/dx=0), so the Poisson problem is the exact solution
    for **every** Reynolds number, not only Re -> 0.  Re = 5.19 / 10.38 here.
  * The solver is *validated against the analytic ellipse*: the digital solution
    converges to the continuous-ellipse Poiseuille solution with O(1/a) and its
    peak velocity error falls 3.0% -> 1.25% when a doubles (see
    `docs/POISEUILLE_3D_ELLIPSE_REFERENCE_GEOMETRY_AUDIT.md`).

Metrics produced (all programmatic, nothing hand-copied)
-------------------------------------------------------
  theta_num   : LBM binned profile vs digital-geometry reference (Q-matched),
                max relative error over the central bins (|u_ana| > 0.2 U_max,
                the repository's primary-metric convention).
  theta_geom  : continuous-ellipse analytic (s=1) vs digital-geometry reference,
                same bins -> the part of the 5.107% that is *reference geometry*.
  theta_total : LBM vs continuous-ellipse analytic (this reproduces the recorded
                primary metric 5.107% / 2.432% bit-for-bit, a pipeline gate).
  delta_eff   : effective isotropic wall offset, the single physical parameter
                that best maps the observed profile onto the ellipse family
                u ~ 1 - (y/(a+d))^2 - (z/(b+d))^2 (Q-matched least squares).
                Reported for LBM and for the digital reference; the analytic
                ellipse sits at delta = 0 by construction.

Usage
-----
    python digital_reference.py [--out digital_reference_result.json] [--figs]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = Path(__file__).resolve().parent

#: run.py's conventions (must not drift: the mask is bit-compared in the tests)
U_IN = 0.02
NU = 0.1
CASES = ((20, 10, "case_a20_b10.json"), (40, 20, "case_a40_b20.json"))


# --------------------------------------------------------------------------- #
# geometry (bit-identical to run.py::ellipse_setup)
# --------------------------------------------------------------------------- #
def ellipse_mask(a: int, b: int):
    """Return (lam, fluid) exactly as run.py builds them (float32, lam <= 1)."""
    ny, nz = 2 * a + 3, 2 * b + 3
    yc, zc = a + 1, b + 1
    iy = np.arange(ny, dtype=np.float32).reshape(1, -1)
    iz = np.arange(nz, dtype=np.float32).reshape(-1, 1)
    lam = ((iy - yc) / np.float32(a)) ** 2 + ((iz - zc) / np.float32(b)) ** 2
    return lam, lam <= np.float32(1.0)


# --------------------------------------------------------------------------- #
# digital-geometry Stokes reference
# --------------------------------------------------------------------------- #
def solve_stokes_digital(fluid: np.ndarray, nu: float = NU, G: float = 1.0):
    """Exact (machine-precision) Poisson solve on the digital cross-section.

    ``sum_nb (u_nb - u_c) = -G/nu`` with the half-way no-slip closure
    ``u_nb = -u_c`` for every solid neighbour (wall at 0.5 cell).  Returns the
    velocity array with ``nan`` in solid cells.
    """
    nz, ny = fluid.shape
    idx = {}
    for i in range(nz):
        for j in range(ny):
            if fluid[i, j]:
                idx[(i, j)] = len(idx)
    n = len(idx)
    A = sp.lil_matrix((n, n))
    rhs = np.full(n, -G / nu)
    for (i, j), k in idx.items():
        A[k, k] = -4.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if not (0 <= ii < nz and 0 <= jj < ny):
                raise ValueError("solid margin too thin: neighbour off the grid")
            if fluid[ii, jj]:
                A[k, idx[(ii, jj)]] += 1.0
            else:
                A[k, k] -= 1.0  # midpoint no-slip: u_nb = -u_c  ->  -2 u_c
    u = spl.spsolve(A.tocsr(), rhs)
    if not np.all(np.isfinite(u)):
        raise RuntimeError("digital Stokes solve produced non-finite values")
    U = np.full((nz, ny), np.nan)
    for (i, j), k in idx.items():
        U[i, j] = u[k]
    return U


# --------------------------------------------------------------------------- #
# binning (identical to run.py::ellipse_profile bin definition)
# --------------------------------------------------------------------------- #
def bin_indices(lam: np.ndarray, fluid: np.ndarray, nb: int):
    """run.py's bins: k covers k/nb <= lam < (k+1)/nb, nb = b bins."""
    lam32 = lam.astype(np.float32)
    return np.clip(np.floor(lam32 * np.float32(nb)).astype(int), 0, nb - 1), fluid


def bin_mean(field: np.ndarray, lam: np.ndarray, fluid: np.ndarray, nb: int):
    bi, fluid = bin_indices(lam, fluid, nb)
    out, cells = [], []
    for k in range(nb):
        m = fluid & (bi == k)
        out.append(float(field[m].mean()) if m.any() else float("nan"))
        cells.append(int(m.sum()))
    return np.array(out), cells


# --------------------------------------------------------------------------- #
# effective wall offset and error budget
# --------------------------------------------------------------------------- #
def _ellipse_shape(lam_edges, a, b, delta):
    """Q-matched analytic ellipse profile for semi-axes (a+delta, b+delta)."""
    # lam was built with the nominal (a,b); rescale per cell is done in analyse()
    raise NotImplementedError  # pragma: no cover  (see effective_offset)


def effective_offset(lam, fluid, u_bins, a, b, nb, lo=-2.0, hi=3.0, n=501):
    """Best isotropic wall offset delta (cells) of a measured binned profile.

    The model profile is the analytic ellipse field with semi-axes (a+delta,
    b+delta), evaluated per cell and then bin-averaged with *the same* binning,
    scaled to the measured flux (Q-matched).  One parameter, brute-force scan
    followed by parabolic refinement -> deterministic and gradient free.
    """
    ny = lam.shape[1]
    iz = np.arange(lam.shape[0], dtype=np.float64).reshape(-1, 1)
    iy = np.arange(ny, dtype=np.float64).reshape(1, -1)

    def err(delta):
        ae, be = a + delta, b + delta
        shape = 1.0 - ((iy - (a + 1)) / ae) ** 2 - ((iz - (b + 1)) / be) ** 2
        model, _ = bin_mean(shape, lam, fluid, nb)
        good = np.isfinite(model) & np.isfinite(u_bins) & (model > 0)
        if not good.any():
            return np.inf
        scale = u_bins[good].sum() / model[good].sum()
        rel = (u_bins[good] - scale * model[good]) / (scale * model[good])
        return float(np.sqrt(np.mean(rel ** 2)))

    grid = np.linspace(lo, hi, n)
    vals = np.array([err(d) for d in grid])
    k = int(np.argmin(vals))
    d0 = grid[k]
    # parabolic refinement on the three points around the minimum
    if 0 < k < n - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            step = grid[1] - grid[0]
            d0 = d0 + 0.5 * step * (y0 - y2) / den
    return float(d0), float(err(d0))


def analyse_case(a: int, b: int, case_json: Path):
    """Full audit for one grid: reference, metrics, budget."""
    lam, fluid = ellipse_mask(a, b)
    nb = int(b)
    U = solve_stokes_digital(fluid)

    rec = json.loads(Path(case_json).read_text(encoding="utf-8"))
    u_lbm = np.array(rec["u_profile"], dtype=np.float64)

    # --- continuous-ellipse analytic at the recorded s^Q (reproduces the
    #     repository's primary metric -> pipeline gate)
    s_q = float(rec["s_Q"])
    u_max = 2.0 * U_IN
    ana_cell = u_max * (1.0 - lam.astype(np.float64) / s_q ** 2)
    u_ana, cells_ana = bin_mean(ana_cell, lam, fluid, nb)
    central = np.abs(u_ana) > 0.2 * u_max
    theta_total = float(np.max(np.abs(u_lbm - u_ana)[central] / np.abs(u_ana)[central]))

    # --- digital-geometry reference (Q-matched scale: Q is an integral observable)
    u_ref, cells_ref = bin_mean(U, lam, fluid, nb)
    scale = u_lbm[np.isfinite(u_lbm)].sum() / u_ref[np.isfinite(u_ref)].sum()
    u_ref_q = u_ref * scale
    theta_num = float(np.max(np.abs(u_lbm - u_ref_q)[central] / np.abs(u_ref_q)[central]))
    theta_geom = float(
        np.max(np.abs(u_ref_q - u_ana)[central] / np.abs(u_ana)[central])
    )

    # --- effective wall offsets (diagnostic, one physical parameter each)
    d_lbm, rms_lbm = effective_offset(lam, fluid, u_lbm, a, b, nb)
    d_ref, rms_ref = effective_offset(lam, fluid, u_ref, a, b, nb)

    return {
        "a": a,
        "b": b,
        "grid": [int(lam.shape[1]), int(lam.shape[0])],
        "fluid_cells": int(fluid.sum()),
        "bins": nb,
        "cells_match_recorded": cells_ref == list(rec.get("bins_cells", [])),
        "cells_ana_match_recorded": cells_ana == list(rec.get("bins_cells", [])),
        "s_Q_recorded": s_q,
        "u_c_digital_ref": float(np.nanmax(U)),
        "u_c_digital_ref_analytic_pct": float(
            100.0 * (np.nanmax(U) - (a * a * b * b) / (2.0 * NU * (a * a + b * b)))
            / ((a * a * b * b) / (2.0 * NU * (a * a + b * b)))
        ),
        "theta_total_pct": 100.0 * theta_total,
        "theta_total_recorded_pct": float(rec["max_rel_bin_central_sQ_pct"]),
        "theta_num_pct": 100.0 * theta_num,
        "theta_geom_pct": 100.0 * theta_geom,
        "delta_eff_lbm_cells": d_lbm,
        "delta_eff_lbm_rms_pct": 100.0 * rms_lbm,
        "delta_eff_digital_cells": d_ref,
        "delta_eff_digital_rms_pct": 100.0 * rms_ref,
        "bins_lbm": [round(float(v), 8) for v in u_lbm],
        "bins_digital_ref": [round(float(v), 8) for v in u_ref_q],
        "bins_analytic_sQ": [round(float(v), 8) for v in u_ana],
        "per_bin_err_vs_digital_pct": [
            round(float(100.0 * (l - r) / r), 4) for l, r in zip(u_lbm, u_ref_q)
        ],
        "per_bin_err_vs_analytic_pct": [
            round(float(100.0 * (l - r) / r), 4) for l, r in zip(u_lbm, u_ana)
        ],
        "central_bins": int(central.sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "digital_reference_result.json"))
    args = ap.parse_args(argv)

    per_grid = [analyse_case(a, b, HERE / name) for a, b, name in CASES]
    out = {
        "case": "poiseuille_3d_ellipse_digital_reference_audit",
        "reference": (
            "digital cross-section Stokes/Poiseuille solve (machine precision) with "
            "half-way (mid-point) no-slip closure on the same staircase mask the LBM "
            "integrates; exact for fully developed duct flow at any Re"
        ),
        "nu_lb": NU,
        "u_in": U_IN,
        "metrics": {
            "theta_total": "LBM vs continuous-ellipse analytic at recorded s^Q (repo primary metric)",
            "theta_num": "LBM vs digital-geometry reference, Q-matched",
            "theta_geom": "digital-geometry reference vs continuous-ellipse analytic, Q-matched",
            "delta_eff": "isotropic effective wall offset (cells) from a Q-matched least-squares scan",
        },
        "per_grid": per_grid,
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for g in per_grid:
        print(
            f"a={g['a']:>2}  theta_total={g['theta_total_pct']:.3f}% "
            f"(recorded {g['theta_total_recorded_pct']:.3f}%)  "
            f"theta_num={g['theta_num_pct']:.3f}%  theta_geom={g['theta_geom_pct']:.3f}%  "
            f"delta_LBM={g['delta_eff_lbm_cells']:+.3f}  delta_ref={g['delta_eff_digital_cells']:+.3f}  "
            f"cells_match={g['cells_match_recorded']}"
        )
    return out


if __name__ == "__main__":
    main()