#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-3D -- case-pool entry: decay of the 3-D Taylor-Green vortex.

Reference: the **exact time-Taylor series** of the Helmholtz-projected
Navier-Stokes equations (``级数精确解``, ``src/tensorlbm/tg3d_series.py``),
``u(x,t) = sum_n t^n u^(n)(x)`` with

    u^(n+1) = ( nu*Lap(u^(n)) - P sum_{j=0..n} (u^(j).grad) u^(n-j) ) / (n+1)

and ``P`` the Helmholtz projection.  Every ``u^(n)`` is a finite Fourier series on
integer wavevectors, so the recursion has exact spatial derivatives and no time
discretisation.  That is what makes the classic (z-dependent) Taylor-Green field
comparable at all: it has *no* closed form, and this case's point is precisely the
3-D signal the z-invariant mode cannot produce (``w != 0``, vortex stretching).

The series generator is not trusted on assertion -- it is pinned first (Q0)
against two independent closed forms (the z-invariant mode and the Beltrami/ABC
flow, both of which it must reproduce as pure exponential decay to 1e-9), its own
truncation is bounded (Q5, Q8 order convergence, Q9 working-grid invariance) and
it is cross-checked against a de-aliased pseudo-spectral DNS of the same IC
(leg ``dns`` / Q10: the closed-form anchor cannot exercise the convective
term, because both closed forms are nonlinearly trivial).

Protocol: lattice units (dx = dt = 1), periodic N^3 box, D3Q27 BGK, float64,
nu = A*N/Re (fixed Reynolds number Re = 160, so refining the grid refines both
the spatial resolution and tau).

Usage
-----
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py --smoke
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py --legs anchor,convergence

Writes ``artifacts/summary.json`` (+ ``field_primary.npz``) when run outside
``--smoke``; ``--smoke`` is write-free so CI can never clobber the committed
evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parents[2]
ARTIFACTS = CASE_DIR / "artifacts"

# --------------------------------------------------------------------------- #
# predeclared protocol (no number below is chosen after seeing a result)
# --------------------------------------------------------------------------- #
CASE = "CASE-TAYLOR-GREEN-3D"
TOL_PCT = 8.0                 # feature_ledger acceptance band
AMPLITUDE = 0.05              # A  -> Ma = A*sqrt(3) = 0.0866
REYNOLDS = 160.0              # Re = A*N/nu
SERIES_ORDER = 12             # retained powers of t (primary reference)
EDDY_TIMES = 1.0              # T, in eddy times A*k*t
PRIMARY_GRID = 32
SMOKE_GRID = 24
ORDER_GRIDS = (16, 24, 32)    # grids used for the observed-order fit
ORDER_MIN = 1.5               # predeclared floor on the observed order
# Q10: series vs independent de-aliased pseudo-spectral DNS.  A priori the RK4
# global time error at advective CFL ~ A = 0.05 is O((A k dt)^4) ~ 1e-6 over the
# run and the 2/3-dealiasing band is inactive for this field at T <= 1, so two
# correct implementations of the same problem agree well inside 1e-3 -- three
# orders tighter than the pool band, i.e. this check sees a real algebra error
# long before it could hide inside the 8 % LBM tolerance.
DNS_XCHECK_REL = 1e-3
# Q11: reproducibility of the *reference field* on the evaluation grid.  A field
# comparison on an n^3 grid only means something if the reference's de-aliasing
# band (nfft/3) covers the whole evaluation band, i.e. nfft >= 3n; with the
# library default nfft = 5*order (= 60 at order 12) the same series returns
# fields differing by several percent at n = 24/32 (measured, see
# legs.reference_convergence.field_workgrid_rel_default_nfft), which is the same
# size as the solver error the case claims to measure.  The reference must
# reproduce itself to a small fraction of the 8 % band: 1e-3 = 80x margin.
FIELD_NFFT_FACTOR = 3
FIELD_WORKGRID_REL = 1e-3
REF_ORDER_LADDER = ((10, 60), (12, 60), (12, 72))  # (order, working nfft) variants

# id -> (tolerance, direction).  "le" = value must be <= tol, "ge" = >= tol.
THRESHOLDS: dict[str, tuple[float, str]] = {
    "Q0_series_vs_closed_form_rel": (1e-9, "le"),
    "Q1_energy_ratio_rel_err_pct": (TOL_PCT, "le"),
    "Q2_enstrophy_ratio_rel_err_pct": (TOL_PCT, "le"),
    "Q3_velocity_l2_rel_err_pct": (TOL_PCT, "le"),
    "Q4_vorticity_l2_rel_err_pct": (TOL_PCT, "le"),
    "Q5_series_truncation_rel": (1e-3, "le"),
    # O(Ma^2) = 7.5e-3 at Ma = 0.0866: the local density of an athermal LBM is a
    # finite-Mach quantity, not a conserved one -> the bound is ~2.7*Ma^2, not 0.
    "Q6_rho_deviation": (1e-2, "le"),
    "Q7_observed_order": (ORDER_MIN, "ge"),
    "Q8_series_order_convergence_rel": (1e-3, "le"),
    "Q9_series_workgrid_invariance_rel": (1e-9, "le"),
    "Q10_series_vs_dns_rel": (DNS_XCHECK_REL, "le"),
    "Q11_series_field_workgrid_rel": (FIELD_WORKGRID_REL, "le"),
}

LEGS = ("anchor", "convergence", "exactness", "dns", "primary", "grid")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parameters(n: int) -> dict:
    """Lattice parameters at fixed (amplitude, Reynolds)."""
    from tensorlbm.tg3d_lbm_check import tg3d_parameters

    return tg3d_parameters(n, amplitude=AMPLITUDE, reynolds=REYNOLDS)


def steps_for(n: int, eddy_times: float = EDDY_TIMES) -> int:
    p = parameters(n)
    return max(2, int(round(eddy_times / p["eddy_time_per_step"])))


def series(n: int, *, order: int = SERIES_ORDER, nfft: int | None = None):
    """Reference series for the lattice problem of size ``n``.

    The working grid defaults to ``max(default_nfft(order), FIELD_NFFT_FACTOR*n)``
    -- *not* the library default.  Rationale: the velocity field is compared on
    the ``n^3`` lattice, and the series' 2/3 de-aliasing band is ``nfft/3``; the
    field is reproducible across working grids only when ``nfft/3 >= n``.  At the
    library default (``5*order`` = 60 at order 12) the two build grids give fields
    differing by ~6 % at n = 24/32, i.e. as large as the solver error being
    measured.  Energy/enstrophy are modal sums and are invariant to this choice
    (1e-16), so the ratio gates are unaffected.
    """
    from tensorlbm.tg3d_lbm_check import series_reference
    from tensorlbm.tg3d_series import default_nfft

    if nfft is None:
        nfft = max(default_nfft(order), FIELD_NFFT_FACTOR * n)
    return series_reference(
        n, amplitude=AMPLITUDE, reynolds=REYNOLDS, order=order, nfft=nfft
    )


def _rel(a: float, b: float) -> float:
    return abs(a / b - 1.0) if b else float("nan")


def _l2_rel_pct(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sqrt(np.sum(b * b)))
    return 100.0 * float(np.sqrt(np.sum((a - b) ** 2))) / den if den else float("nan")


# --------------------------------------------------------------------------- #
# leg 0: pin the reference generator against closed forms
# --------------------------------------------------------------------------- #
def closed_form_anchor(*, n: int = PRIMARY_GRID, order: int = SERIES_ORDER,
                       eddy_times: float = EDDY_TIMES) -> dict:
    """The series must reduce to pure exponential decay where a closed form exists.

    * z-invariant Taylor-Green mode: E(t)/E(0) = exp(-2*nu*|k|^2*t), |k|^2 = 2k^2
    * Beltrami (ABC) flow:           E(t)/E(0) = exp(-2*nu*k^2*t)

    Both are exact Navier-Stokes solutions *and* nonlinearly trivial (the first has
    a gradient-only convective term, the second is Beltrami), so the series has to
    reproduce them to machine precision.  This is the layer-1 self-consistency
    check: it fails if the recursion, the projection or the normalisation inside
    the reference are tampered with, independently of any LBM run.
    """
    from tensorlbm.tg3d_series import TG3DSeries, abc_flow, taylor_green_extruded

    p = parameters(n)
    t = eddy_times / p["eddy_time_per_step"]
    k = 2.0 * math.pi / n
    out: dict[str, float] = {"t_lattice": t, "nu": p["nu"], "n": float(n), "order": float(order)}

    for name, ic_fn, k2 in (
        ("z_invariant", taylor_green_extruded, 2.0 * k * k),
        ("abc_beltrami", abc_flow, k * k),
    ):
        ser = TG3DSeries(ic_fn(AMPLITUDE), box=float(n), nu=p["nu"], order=order)
        ratio_series = ser.energy(t) / ser.energy(0.0)
        ratio_closed = math.exp(-2.0 * p["nu"] * k2 * t)
        out[f"{name}_ratio_series"] = float(ratio_series)
        out[f"{name}_ratio_closed_form"] = float(ratio_closed)
        out[f"{name}_rel_err"] = float(_rel(ratio_series, ratio_closed))
    out["max_rel_err"] = max(out["z_invariant_rel_err"], out["abc_beltrami_rel_err"])
    return out


# --------------------------------------------------------------------------- #
# leg 0c: is the reference *field* reproducible on the evaluation grid?
# --------------------------------------------------------------------------- #
def field_workgrid_check(*, n: int = PRIMARY_GRID, order: int = SERIES_ORDER,
                         factors: tuple[float, ...] = (3.0, 3.5)) -> dict:
    """Reference field across working grids that both satisfy ``nfft >= 3n``.

    Q9 (:func:`leg_reference_convergence`) varies ``nfft`` but only looks at the
    energy/enstrophy ratios, which are modal sums and therefore blind to the
    field: they are ``nfft``-invariant to 1e-16 even when the returned *field*
    moves by several percent.  The field is what Q3/Q4 gate, so it needs its own
    reproducibility statement.

    The library default work grid for order 12 is ``nfft = 60``; at n = 24/32 the
    de-aliasing band (``nfft/3 = 20``) does not cover the evaluation band and the
    field is *not* reproducible.  ``nfft = 3n`` does cover it.  Both the gated
    spread (over ``factors``, all ``>= 3``) and -- for the record -- the spread
    against the library default are reported.
    """
    from tensorlbm.tg3d_series import default_nfft

    t = float(steps_for(n))
    fields: dict[float, np.ndarray] = {}
    levels = []
    for f in factors:
        nfft = int(round(float(f) * n))
        s = series(n, order=order, nfft=nfft)
        u, v, w = s.velocity_field(t, n=n, band_limit=True)
        fields[float(f)] = np.stack([u, v, w])
        levels.append(
            {
                "factor": float(f),
                "nfft": nfft,
                "dealias_margin": float(s.dealias_margin()["margin"]),
                "field_order": int(s.last_field_order),
                "field_support": float(s.last_field_support),
                "energy_ratio": float(s.energy(t) / s.energy(0.0)),
            }
        )
    base = fields[max(fields)]
    top_nfft = int(round(max(fields) * n))
    rel = {f"nfft_{int(round(f * n))}_vs_{int(round(max(fields) * n))}_pct":
           _l2_rel_pct(fields[f], base) for f in sorted(fields)[:-1]}
    nfft_def = int(default_nfft(order))
    default_level = None
    if nfft_def < FIELD_NFFT_FACTOR * n:
        s = series(n, order=order, nfft=nfft_def)
        u, v, w = s.velocity_field(t, n=n, band_limit=True)
        bl = np.stack([u, v, w])
        clamped_order = int(s.last_field_order)
        uc, vc, wc = s.velocity_field(t, n=n)          # legacy clamping path
        default_level = {
            "nfft": nfft_def,
            "field_order_band_limit": clamped_order,
            "energy_ratio": float(s.energy(t) / s.energy(0.0)),
            "field_rel_vs_covered_band_pct": _l2_rel_pct(bl, base),
            # defect witness: the legacy path lowers the truncation order until
            # the spectral support fits below Nyquist, so the *same* series
            # returns a different field depending on the working grid
            "field_order_clamped": int(s.last_field_order),
            "clamped_field_rel_vs_band_pct": _l2_rel_pct(np.stack([uc, vc, wc]), bl),
        }
    return {
        "n": n,
        "order": order,
        "t_lattice": t,
        "factors": [float(f) for f in factors],
        "nfft": [int(round(float(f) * n)) for f in factors],
        "top_nfft": top_nfft,
        "levels": levels,
        "field_workgrid_rel": max(rel.values()) if rel else float("nan"),
        "pairs": rel,
        "default_nfft_level": default_level,
        "note": "field_workgrid_rel = max L2 spread of the reference velocity field "
                "over working grids nfft >= 3n on the n^3 evaluation grid; "
                "default_nfft_level is the (ungated) spread of the library default "
                "work grid 5*order, which is what an unguarded field comparison uses",
    }


# --------------------------------------------------------------------------- #
# leg 0b: the reference's own convergence (is the 8 % band resolvable at all?)
# --------------------------------------------------------------------------- #
def leg_reference_convergence(*, n: int = PRIMARY_GRID,
                              eddy_times: float = EDDY_TIMES,
                              variants: tuple[tuple[int, int], ...] = REF_ORDER_LADDER) -> dict:
    """Bound the reference error itself before comparing anything to it.

    Two independent knobs are moved: the retained Taylor order and the working
    (de-aliasing) grid.  A 2/3-dealiased pseudo-spectral product zeroes every mode
    above ``nfft/3``, so ``dealias_margin()`` reports ``margin == 0`` *by
    construction* for this construction -- it is not a safety statement.  The
    honest reference-error bound is therefore the observed spread below.
    """
    t = eddy_times / parameters(n)["eddy_time_per_step"]
    rows = []
    for order, nfft in variants:
        s = series(n, order=order, nfft=nfft)
        rows.append(
            {
                "order": order,
                "nfft": nfft,
                "energy_ratio": float(s.energy(t) / s.energy(0.0)),
                "enstrophy_ratio": float(s.enstrophy(t) / s.enstrophy(0.0)),
                "support_peak_index": s.dealias_margin()["support_peak_index"],
                "cutoff_index": s.dealias_margin()["cutoff_index"],
            }
        )
    ref = rows[-1]

    def spread(prev: dict, cur: dict) -> float:
        return max(
            abs(prev["energy_ratio"] / cur["energy_ratio"] - 1.0),
            abs(prev["enstrophy_ratio"] / cur["enstrophy_ratio"] - 1.0),
        )

    coarsest_nfft = min(r["nfft"] for r in rows)
    top_order = max(r["order"] for r in rows)
    by_order = [r for r in rows if r["nfft"] == coarsest_nfft]
    by_grid = [r for r in rows if r["order"] == top_order]
    order_conv = spread(by_order[-2], by_order[-1]) if len(by_order) >= 2 else float("nan")
    grid_inv = spread(by_grid[0], by_grid[-1]) if len(by_grid) >= 2 else float("nan")
    return {
        "n": n,
        "eddy_times": eddy_times,
        "t_lattice": t,
        "variants": rows,
        "order_convergence_rel": order_conv,
        "workgrid_invariance_rel": grid_inv,
        "field_workgrid": field_workgrid_check(n=n),
        "note": "order_convergence_rel = |order 10 vs order 12| at nfft=60; "
                "workgrid_invariance_rel = |nfft 60 vs nfft 72| at the primary order",
    }


# --------------------------------------------------------------------------- #
# leg 1: exactness / reference-quality diagnostics
# --------------------------------------------------------------------------- #
def leg_exactness(*, n: int = PRIMARY_GRID, order: int = SERIES_ORDER) -> dict:
    """Why the classic field needs a *series* reference, plus the truncation bounds."""
    from tensorlbm.tg3d_series import (
        abc_flow,
        convective_acceleration,
        taylor_green_classic,
        taylor_green_extruded,
    )

    p = parameters(n)
    acc = {}
    for name, ic_fn in (
        ("classic_z_dependent", taylor_green_classic),
        ("z_invariant", taylor_green_extruded),
        ("abc_beltrami", abc_flow),
    ):
        acc[name] = convective_acceleration(ic_fn(AMPLITUDE), box=float(n), n=32)

    ser = series(n, order=order)
    t_end = float(steps_for(n))
    return {
        "n": n,
        "order": order,
        "nu": p["nu"],
        "tau": p["tau"],
        "convective_acceleration": acc,
        "curl_over_adv_classic": acc["classic_z_dependent"]["curl_over_adv"],
        "curl_over_adv_z_invariant": acc["z_invariant"]["curl_over_adv"],
        "series_dealias": ser.dealias_margin(),
        "series_tail_at_T": ser.relative_tail(t_end),
        "anchor": closed_form_anchor(n=n, order=order),
    }


# --------------------------------------------------------------------------- #
# leg 2: one decay run against the reference
# --------------------------------------------------------------------------- #
def leg_decay(n: int, *, eddy_times: float = EDDY_TIMES, order: int = SERIES_ORDER,
              save_field: bool = False) -> dict:
    """Run the D3Q27 lattice and compare decay ratios + fields to the series."""
    from tensorlbm.tg3d_lbm_check import compare_to_series, run_tg3d_lbm

    t0 = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ser = series(n, order=order)
    run = run_tg3d_lbm(
        n, amplitude=AMPLITUDE, reynolds=REYNOLDS, eddy_times=eddy_times
    )
    rows = compare_to_series(run, ser, band_limit=True)
    wall = time.time() - t0
    last = rows[-1]

    hist = [
        {
            "step": r["step"],
            "eddy_time": r["eddy_time"],
            "energy_ratio_lbm": r["energy_ratio_lbm"],
            "energy_ratio_series": r["energy_ratio_series"],
            "enstrophy_ratio_lbm": r["enstrophy_ratio_lbm"],
            "enstrophy_ratio_series": r["enstrophy_ratio_series"],
            "rho_min": h["rho_min"],
            "rho_max": h["rho_max"],
        }
        for r, h in zip(rows, run["history"])
    ]
    rho_dev = max(
        max(abs(h["rho_min"] - 1.0), abs(h["rho_max"] - 1.0)) for h in run["history"]
    )
    tail = ser.relative_tail(last["time"])

    out = {
        "n": n,
        "steps": run["steps"],
        "eddy_time": last["eddy_time"],
        "tau": run["tau"],
        "nu": run["nu"],
        "ma": run["ma"],
        "reynolds": run["reynolds"],
        "energy_ratio_lbm": last["energy_ratio_lbm"],
        "energy_ratio_series": last["energy_ratio_series"],
        "energy_ratio_rel_err_pct": 100.0
        * (last["energy_ratio_lbm"] / last["energy_ratio_series"] - 1.0),
        "energy_pointwise_rel_err_pct": last["energy_rel_err_pct"],
        "enstrophy_ratio_lbm": last["enstrophy_ratio_lbm"],
        "enstrophy_ratio_series": last["enstrophy_ratio_series"],
        "enstrophy_ratio_rel_err_pct": 100.0
        * (last["enstrophy_ratio_lbm"] / last["enstrophy_ratio_series"] - 1.0),
        "velocity_l2_rel_err_pct": last["u_l2_rel_err_pct"],
        "vorticity_l2_rel_err_pct": last["vorticity_l2_rel_err_pct"],
        "u_l2_abs": last["u_l2_abs"],
        "u_l2_ref": last["u_l2_ref"],
        "series_field_order": last["field_order"],
        "series_truncation_rel": max(
            abs(tail["energy_last_order_rel"]), abs(tail["enstrophy_last_order_rel"])
        ),
        "rho_deviation": float(rho_dev),
        "wall_s": round(wall, 2),
        "history": hist,
    }

    if save_field:
        from tensorlbm.tg3d_series import vorticity_spectral

        t_end = last["time"]
        ue, ve, we = ser.velocity_field(t_end, n=n, band_limit=True)
        wl = vorticity_spectral(
            run["final"]["u"], run["final"]["v"], run["final"]["w"], float(n)
        )
        wr = vorticity_spectral(ue, ve, we, float(n))
        np.savez_compressed(
            ARTIFACTS / "field_primary.npz",
            n=n,
            time=t_end,
            eddy_time=last["eddy_time"],
            wz_lbm_midplane=wl[2][:, :, 0],
            wz_series_midplane=wr[2][:, :, 0],
            u_midplane_lbm=run["final"]["u"][:, :, 0],
            u_midplane_series=ue[:, :, 0],
        )
        out["field_saved"] = "field_primary.npz"

    return out


# --------------------------------------------------------------------------- #
# leg 2b: independent reference -- series vs de-aliased pseudo-spectral DNS
# --------------------------------------------------------------------------- #
def leg_dns(*, n: int = PRIMARY_GRID, eddy_times: float = EDDY_TIMES,
            order: int = SERIES_ORDER, substeps: tuple[int, ...] = (1, 2)) -> dict:
    """Series vs a *different* discretisation of the same problem (spectral DNS).

    The closed-form anchor (leg 0) pins the recursion only where the convective
    term vanishes: the z-invariant mode and the Beltrami flow are both
    nonlinearly trivial, so neither can detect an error in
    ``P sum_j (u^(j).grad) u^(n-j)`` -- precisely the term this 3-D case exists to
    exercise.  This leg closes that hole with a second, independent reference,
    :func:`tensorlbm.tg3d_series.spectral_dns_reference`: 2/3-dealiased Fourier
    pseudo-spectral derivatives, integrating-factor RK4 in time, same IC, same
    box, same ``nu``.  It shares no code path with the time-Taylor series (the
    series is exact in time and carries no time stepper; the DNS carries no
    series at all).

    The DNS time step is refined as well (``dt = 1/s`` for ``s`` in
    ``substeps``, physical time held fixed), so the residual series-vs-DNS
    difference is *attributed* to the DNS time error instead of merely asserted
    small: ``dt_refinement_rel`` is the change between the two step sizes.

    Gated quantity: ``Q10_series_vs_dns_rel`` at the production step ``dt = 1``
    against the predeclared ``DNS_XCHECK_REL``.
    """
    from tensorlbm.tg3d_lbm_check import tg3d_parameters
    from tensorlbm.tg3d_series import spectral_dns_reference, taylor_green_classic

    t0 = time.time()
    p = tg3d_parameters(n, amplitude=AMPLITUDE, reynolds=REYNOLDS)
    steps = steps_for(n, eddy_times)
    t_lattice = float(steps)          # dt = 1 in lattice units
    ser = series(n, order=order)
    e_ser = float(ser.energy(t_lattice) / ser.energy(0.0))
    z_ser = float(ser.enstrophy(t_lattice) / ser.enstrophy(0.0))

    rows = []
    for sub in substeps:
        sub = int(sub)
        dns = spectral_dns_reference(
            taylor_green_classic(AMPLITUDE),
            box=float(n), nu=p["nu"], steps=steps * sub, dt=1.0 / sub,
            n=n, sample_every=steps * sub,
        )
        h0, h1 = dns["history"][0], dns["history"][-1]
        e_dns = h1["energy"] / h0["energy"]
        z_dns = h1["enstrophy"] / h0["enstrophy"]
        rows.append(
            {
                "dt": 1.0 / sub,
                "steps": steps * sub,
                "energy_ratio_dns": float(e_dns),
                "enstrophy_ratio_dns": float(z_dns),
                "energy_rel_err": float(e_dns / e_ser - 1.0),
                "enstrophy_rel_err": float(z_dns / z_ser - 1.0),
                "max_rel_err": float(max(abs(e_dns / e_ser - 1.0),
                                         abs(z_dns / z_ser - 1.0))),
            }
        )
        last = dns

    # field-level (informational, not gated): same grid, same t, same layout
    try:
        ue, ve, we = ser.velocity_field(t_lattice, n=n, band_limit=True)
        num = (np.sum((last["u"] - ue) ** 2) + np.sum((last["v"] - ve) ** 2)
               + np.sum((last["w"] - we) ** 2))
        den = np.sum(ue ** 2) + np.sum(ve ** 2) + np.sum(we ** 2)
        velocity_l2_rel_pct = float(100.0 * math.sqrt(num / den)) if den > 0 else float("nan")
    except Exception as exc:  # pragma: no cover - informational only
        velocity_l2_rel_pct = float("nan")
        rows.append({"field_compare_error": repr(exc)})

    prod = rows[0]
    dt_ref = float("nan")
    if len(rows) >= 2:
        dt_ref = float(max(abs(rows[0]["energy_ratio_dns"] / rows[1]["energy_ratio_dns"] - 1.0),
                           abs(rows[0]["enstrophy_ratio_dns"] / rows[1]["enstrophy_ratio_dns"] - 1.0)))
    return {
        "n": n,
        "steps": steps,
        "eddy_times": eddy_times,
        "t_lattice": t_lattice,
        "nu": p["nu"],
        "reynolds": p["reynolds"],
        "ma": p["ma"],
        "series_order": order,
        "energy_ratio_series": e_ser,
        "enstrophy_ratio_series": z_ser,
        "levels": rows,
        "series_vs_dns_rel": prod["max_rel_err"],
        "dt_refinement_rel": dt_ref,
        "velocity_l2_rel_pct_series_vs_dns": velocity_l2_rel_pct,
        "note": "series_vs_dns_rel is the gated Q10 (production dt = 1); "
                "dt_refinement_rel = |dt=1 vs dt=1/2| of the DNS decay ratios, "
                "i.e. how much of the residual is the DNS time error; "
                "velocity_l2_rel_pct_series_vs_dns is informational (field at T, "
                "same grid, same layout)",
        "wall_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# leg 3: fixed-Re grid ladder -> observed order
# --------------------------------------------------------------------------- #
def leg_grid(*, grids: tuple[int, ...] = ORDER_GRIDS,
             eddy_times: float = EDDY_TIMES) -> dict:
    runs = [leg_decay(n, eddy_times=eddy_times) for n in grids]
    err = np.array([abs(r["energy_ratio_rel_err_pct"]) for r in runs])
    ns = np.array([float(r["n"]) for r in runs])
    slope = float(-np.polyfit(np.log(ns), np.log(err), 1)[0]) if len(runs) >= 2 else float("nan")
    return {
        "grids": [int(n) for n in ns],
        "energy_ratio_rel_err_pct": err.tolist(),
        "enstrophy_ratio_rel_err_pct": [abs(r["enstrophy_ratio_rel_err_pct"]) for r in runs],
        "velocity_l2_rel_err_pct": [r["velocity_l2_rel_err_pct"] for r in runs],
        "observed_order": slope,
        "monotone_decreasing": bool(np.all(np.diff(err) < 0)),
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _quantity(qid: str, ref: float, got: float, unit: str = "") -> dict:
    tol, direction = THRESHOLDS[qid]
    ok = (got <= tol) if direction == "le" else (got >= tol)
    return {
        "id": qid,
        "ref": ref,
        "got": got,
        "tolerance": tol,
        "direction": direction,
        "unit": unit,
        "pass": bool(ok),
    }


def build_verdict(*, grid: int = PRIMARY_GRID, eddy_times: float = EDDY_TIMES,
                  with_grid: bool = True, with_ref_conv: bool = True,
                  save_field: bool = True, anchor: dict | None = None,
                  refconv: dict | None = None, primary: dict | None = None,
                  grid_leg: dict | None = None, with_dns: bool = True,
                  dns: dict | None = None) -> dict:
    t0 = time.time()
    anchor = anchor if anchor is not None else closed_form_anchor(n=grid, eddy_times=eddy_times)
    if refconv is None:
        refconv = leg_reference_convergence(n=grid) if with_ref_conv else None
    primary = primary if primary is not None else leg_decay(
        grid, eddy_times=eddy_times, save_field=save_field)
    if grid_leg is None and with_grid:
        grid_leg = leg_grid(eddy_times=eddy_times)
    if dns is None and with_dns:
        dns = leg_dns(n=grid, eddy_times=eddy_times)

    # Q0..Q2 ref columns are the reference-side value of the same quantity where a
    # reference-side value exists; Q0/Q5/Q6 are *deviations*, so their reference
    # value is 0.
    qs = [
        _quantity("Q0_series_vs_closed_form_rel", 0.0, anchor["max_rel_err"], "rel"),
        _quantity("Q1_energy_ratio_rel_err_pct", primary["energy_ratio_series"],
                  abs(primary["energy_ratio_rel_err_pct"]), "%"),
        _quantity("Q2_enstrophy_ratio_rel_err_pct", primary["enstrophy_ratio_series"],
                  abs(primary["enstrophy_ratio_rel_err_pct"]), "%"),
        _quantity("Q3_velocity_l2_rel_err_pct", primary["u_l2_ref"],
                  primary["velocity_l2_rel_err_pct"], "%"),
        _quantity("Q4_vorticity_l2_rel_err_pct", primary["u_l2_ref"],
                  primary["vorticity_l2_rel_err_pct"], "%"),
        _quantity("Q5_series_truncation_rel", 0.0, primary["series_truncation_rel"], "rel"),
        _quantity("Q6_rho_deviation", 0.0, primary["rho_deviation"], "rel"),
    ]
    if grid_leg is not None:
        qs.append(_quantity("Q7_observed_order", ORDER_MIN,
                            abs(grid_leg["observed_order"]), "-"))
    if refconv is not None:
        qs.append(_quantity("Q8_series_order_convergence_rel", 0.0,
                            refconv["order_convergence_rel"], "rel"))
        qs.append(_quantity("Q9_series_workgrid_invariance_rel", 0.0,
                            refconv["workgrid_invariance_rel"], "rel"))
    if dns is not None:
        qs.append(_quantity("Q10_series_vs_dns_rel", 0.0,
                            dns["series_vs_dns_rel"], "rel"))
    if refconv is not None and refconv.get("field_workgrid"):
        qs.append(_quantity("Q11_series_field_workgrid_rel", 0.0,
                            refconv["field_workgrid"]["field_workgrid_rel"], "rel"))

    failed = [q["id"] for q in qs if not q["pass"]]
    return {
        "case": CASE,
        "repo": "TensorLBM",
        "reference": "exact time-Taylor series (级数精确解), src/tensorlbm/tg3d_series.py",
        "protocol": {
            "grid_primary": grid,
            "grids_ladder": list(grid_leg["grids"]) if grid_leg else [],
            "eddy_times": eddy_times,
            "amplitude": AMPLITUDE,
            "reynolds": REYNOLDS,
            "series_order": SERIES_ORDER,
            "tol_pct": TOL_PCT,
            "thresholds": {k: {"tolerance": v[0], "direction": v[1]}
                           for k, v in THRESHOLDS.items()},
        },
        "quantities": qs,
        "verdict": {"pass": not failed, "failed": failed, "n_quantities": len(qs)},
        "legs": {
            "anchor": anchor,
            "reference_convergence": refconv,
            "primary": {k: v for k, v in primary.items() if k != "history"},
            "primary_history": primary["history"],
            "grid": {k: v for k, v in (grid_leg or {}).items() if k != "runs"},
            "dns": dns,
        },
        "wall_s": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


# --------------------------------------------------------------------------- #
# pool entry point
# --------------------------------------------------------------------------- #
def main(extra: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CASE-TAYLOR-GREEN-3D (pool case)")
    ap.add_argument("--legs", default="anchor,convergence,exactness,dns,primary,grid",
                    help="comma separated subset of: " + ",".join(LEGS))
    ap.add_argument("--smoke", action="store_true",
                    help="fast, write-free check on a coarse grid (CI)")
    ap.add_argument("--grid", type=int, default=PRIMARY_GRID)
    ap.add_argument("--eddy-times", type=float, default=EDDY_TIMES)
    ap.add_argument("--out", default="")
    ap.add_argument("--grids", default="",
                    help="comma separated grids for the --legs grid leg")
    ap.add_argument("--dump-leg", action="store_true",
                    help="write the single selected leg verbatim (no verdict); used to "
                         "run legs in separate short processes, then --assemble")
    ap.add_argument("--assemble", action="store_true",
                    help="merge artifacts/legs/*.json into the verdict (artifacts/summary.json)")
    args = ap.parse_args(list(extra) if extra is not None else None)

    if args.assemble:
        lp = ARTIFACTS / "legs"
        legs = {}
        for f in sorted(lp.glob("*.json")):
            payload = json.loads(f.read_text())
            # a --dump-leg file is either the bare leg or {leg_name: leg}
            inner = list(payload.values())[0] if len(payload) == 1 else None
            legs[f.stem] = inner if isinstance(inner, dict) else payload
        primary = legs["primary"]
        res = build_verdict(
            grid=int(primary["n"]), anchor=legs["anchor"], refconv=legs["convergence"],
            primary=primary, grid_leg=legs.get("grid"), dns=legs.get("dns"))
        res["legs"]["exactness"] = legs["exactness"]
        res["assembly"] = {"per_leg_files": sorted(legs), "note": res.get("assembly", {}).get("note")}
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        for q in res["quantities"]:
            print(f"  {q['id']:34s} ref={q['ref']:<12.6g} got={q['got']:<12.6g} "
                  f"{q['direction']} {q['tolerance']:<10g} {'PASS' if q['pass'] else 'FAIL'}")
        print(f"VERDICT: {'PASS' if res['verdict']['pass'] else 'FAIL ' + str(res['verdict']['failed'])}")
        return 0 if res["verdict"]["pass"] else 1

    if args.smoke:
        s = build_verdict(grid=SMOKE_GRID, eddy_times=args.eddy_times,
                          with_grid=False, with_ref_conv=False, save_field=False)
        failed = s["verdict"]["failed"]
        print(f"[smoke] grid={SMOKE_GRID} " +
              " ".join(f"{q['id']}={q['got']:.4g}" for q in s["quantities"]))
        print(f"[smoke] verdict={'PASS' if not failed else 'FAIL ' + str(failed)}")
        return 0 if not failed else 1

    wanted = [w for w in args.legs.split(",") if w]
    unknown = [w for w in wanted if w not in LEGS]
    if unknown:
        ap.error(f"unknown legs {unknown}; known: {LEGS}")

    failed: list[str] = []
    if set(wanted) == {"anchor"}:
        res: dict = {"anchor": closed_form_anchor(n=args.grid)}
    elif set(wanted) == {"convergence"}:
        res = {"reference_convergence": leg_reference_convergence(n=args.grid)}
    elif set(wanted) == {"exactness"}:
        res = {"exactness": leg_exactness(n=args.grid)}
    elif set(wanted) == {"dns"}:
        res = {"dns": leg_dns(n=args.grid, eddy_times=args.eddy_times)}
    elif set(wanted) == {"grid"}:
        grids = tuple(int(g) for g in args.grids.split(",") if g) or ORDER_GRIDS
        res = {"grid": leg_grid(grids=grids)}
    elif set(wanted) == {"primary"}:
        res = {"primary": leg_decay(args.grid, eddy_times=args.eddy_times,
                                    save_field=args.dump_leg)}
    else:
        res = build_verdict(grid=args.grid, eddy_times=args.eddy_times,
                            with_grid="grid" in wanted)
        failed = res["verdict"]["failed"]

    if args.dump_leg:
        (ARTIFACTS / "legs").mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS / "legs" / f"{wanted[0]}.json"
    else:
        out_path = Path(args.out) if args.out else ARTIFACTS / "summary.json"
    out_path.write_text(json.dumps(res, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out_path}")
    if not args.dump_leg and "quantities" not in res:
        return 0 if not failed else 1
    if "quantities" in res:
        for q in res["quantities"]:
            print(f"  {q['id']:34s} ref={q['ref']:<12.6g} got={q['got']:<12.6g} "
                  f"{q['direction']} {q['tolerance']:<10g} {'PASS' if q['pass'] else 'FAIL'}")
        print(f"VERDICT: {'PASS' if res['verdict']['pass'] else 'FAIL ' + str(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())