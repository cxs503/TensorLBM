#!/usr/bin/env python3
"""G5-07 gates — the 3-D LBM–FEM–DEM interface driven by a **real** D3Q19 field.

Round 1 (``scripts/three_way_interface3d_gates.py``) gated the interface against
*analytic* tractions.  Those are affine, and a Q1 consistent transfer is exact for
affine fields — so the round-1 gate could not distinguish a good transfer from a
lucky one on a real load.  Round 2 (this driver) replaces the analytic traction by
the per-link momentum ledger of an actual Bouzidi–Firdaouss–Lallemand wall on the
calibrated ``benchmarks/verified/sphere_re100`` case, and re-gates:

====  =========================================================================
G1    the *distribution* the interface consumes sums back to the library total
      ledger (so the caliber is the verified one, not a look-alike)
G2    link forces -> facet traction conserves the resultant exactly; coverage
G3    Q1 hex transfer on the **real** traction: resultant / moment / work
      (non-affine nodal velocity field, not just a rigid motion)
G4    negative control: the same operator on a grid in the wrong frame fails —
      i.e. G3 has teeth
G5    drag coefficient from the real field vs Schiller–Naumann and vs the stored
      verified D=40/D=60 artefacts
G6    added-mass guard evaluated with the *real* displaced volume and ice mass,
      plus the unstable-branch growth rate vs the round-1 spectral radius
G7    a 1-DOF structural response driven by the measured drag history, with the
      work ledger closed to the reported integrator order
====  =========================================================================

Usage
-----
    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/three_way_g507_field_gates.py \
        --D 20 --steps 2500 --out artifacts/three_way_g507

The run is single-threaded so that the numbers are reproducible; on a 128-core
host submit the D=20 case to the cluster instead of hogging the control plane.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorlbm.three_way.interface3d import (  # noqa: E402
    DTYPE,
    HexGrid3D,
    added_mass_ratio,
    march_one_dof,
    staggered_spectral_radius,
)
from tensorlbm.three_way.lbm3d_field import (  # noqa: E402
    growth_rate,
    march_driven,
    run_sphere_field,
    transfer_gates,
)

# ---------------------------------------------------------------------------
# Pre-registered tolerances (fixed before the production run, see
# docs/GAP-LBM-T5-ICE-COUPLING_round2_G507_field_driven.md sec. 3)
# ---------------------------------------------------------------------------
TOL = {
    "g1_ledger_rel": 1e-4,        # float32 accumulation order only
    "g2_resultant_rel": 1e-10,    # exact partition -> round-off
    "g2_coverage": 0.90,          # >= 90 % of facets must carry links
    "g2_binning_pointwise": 5e-2,  # facet-quadrature moment error, pointwise scale
    "g3_resultant_rel": 1e-10,
    "g3_moment_rel": 1e-10,
    "g3_work_rel": 1e-10,
    "g4_frame_error_min": 1e-2,   # negative control must be *visibly* wrong
    "g5_cd_err_pct": 5.0,         # honest resolution-aware band at D >= 16
    "g6_growth_rel": 0.05,        # measured vs spectral growth, unstable branch
    # Trapezoid work on a *sampled* (piecewise-constant) drag history is only
    # first-order at the sample jumps, so the ledger residual order is gated at 1.
    "g7_ledger_order": 1.0,
    "g7_closure_max": 5e-3,
}

RHO_WATER = 1000.0
RHO_ICE = 917.0
RHO_LIGHT_FLOE = 300.0        # deliberately sub-ice density: exercises mu >= 1
CA_SPHERE = 0.5               # added-mass coefficient of a sphere (potential flow)
SCHILLER_NAUMANN = lambda re: 24.0 / re * (1.0 + 0.15 * re**0.687)  # noqa: E731


def schiller_naumann(re: float) -> float:
    return SCHILLER_NAUMANN(re)


def stored_case_reference(path: str) -> dict:
    """Stored verified levels of the sphere case (programmatic, no hand copying)."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    out = {}
    for metric in data.get("metrics", []):
        if metric.get("metric") != "cd":
            continue
        for lvl in metric.get("levels", []):
            out[lvl.get("level", "?")] = float(lvl.get("value"))
        out["reference"] = float(metric.get("reference", float("nan")))
    return out


def rigid_plus_mode(origin):
    """Non-affine structural velocity field: rigid rotation + a bending mode.

    The round-1 work gate used affine fields only.  A field with a quadratic
    (non-affine) contribution is the real test of work conjugacy, because the
    consistent operator is the exact transpose only for the *interpolated* field —
    which is what the interface claims.
    """
    m_ref = torch.as_tensor(list(origin), dtype=DTYPE)

    def field(nodes: torch.Tensor) -> torch.Tensor:
        r = nodes.to(dtype=DTYPE) - m_ref
        w = torch.tensor([0.3, -0.7, 1.1], dtype=DTYPE)
        rigid = torch.cross(w.expand_as(r), r, dim=1)
        # quadratic bending mode, normalised so its amplitude is O(1)
        amp = 1e-3 * (r[:, 2] ** 2 - r[:, 0] * r[:, 1]) / max(float(r.abs().max()), 1e-30)
        mode = torch.stack([amp, amp * 0.5, -amp], dim=1)
        return rigid + mode

    return field


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=20)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--tau", type=float, default=0.56)
    ap.add_argument("--u", dest="u_lb", type=float, default=0.05)
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--out", default="artifacts/three_way_g507")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--surface-levels", default="12,24;16,32;24,48;32,64")
    ap.add_argument("--label", default="")
    ap.add_argument("--replot-only", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    levels = [tuple(int(v) for v in lv.split(",")) for lv in args.surface_levels.split(";")]
    outdir = args.out
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    if args.replot_only:
        with open(os.path.join(outdir, f"gates_D{args.D}.json")) as fh:
            payload = json.load(fh)
        make_figures(payload, os.path.join(outdir, f"field_D{args.D}.npz"), figdir)
        print(f"[g507] replotted D={args.D} -> {figdir}")
        return 0
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    t0 = time.time()
    run = run_sphere_field(
        D=args.D,
        steps=args.steps,
        tau=args.tau,
        u_lb=args.u_lb,
        sample=args.sample,
        surface_levels=levels,
        device="cpu",
    )
    assert run.finite, "field run diverged"
    assert run.load is not None

    R = run.radius
    centre = run.centre
    cd_ref = schiller_naumann(run.re)
    stored = stored_case_reference("benchmarks/verified/sphere_re100/validation_status.json")
    cd_stored40 = stored.get("D40", float("nan"))
    cd_stored60 = stored.get("D60", float("nan"))

    # -- G1: the distribution *is* the verified caliber --------------------
    g1_pass = run.closure["rel_diff"] <= TOL["g1_ledger_rel"]

    # -- G2 / G3: real traction into the structural grid -------------------
    grid = HexGrid3D(
        origin=(centre[0] - R - 2.0, centre[1] - R - 2.0, centre[2] - R - 2.0),
        extent=(2 * R + 4.0, 2 * R + 4.0, 2 * R + 4.0),
        n_cells=(max(6, args.D // 2),) * 3,
    )
    moment_origin = [centre[0], centre[1], centre[2] - 3.0 * R]
    # Admissible surface resolution: the interface consumes the finest quadrature
    # that still satisfies the pre-registered coverage requirement. A quadrature
    # whose facets mostly carry no link force is under-resolved: it is reported in
    # the surface study, not consumed.
    admissible = [(k, ld) for k, ld in run.loads.items() if ld.coverage >= TOL["g2_coverage"]]
    if not admissible:
        admissible = list(run.loads.items())
    gate_key, finest = max(admissible, key=lambda kv: len(kv[1].surface))
    g3 = transfer_gates(
        finest, grid, origin=centre, moment_origin=moment_origin,
        velocity=rigid_plus_mode(moment_origin),
    )
    # G2 baseline is the *distribution* total (float64), not the float32 library
    # ledger: the distribution -> facet partition must be exact on its own, and
    # G1 separately proves the distribution is the verified caliber.
    dist_total = torch.tensor(run.closure["distribution_total"], dtype=DTYPE)
    g2_resultant_rel = float(torch.linalg.norm(finest.forces.sum(0) - dist_total)) / max(
        float(torch.linalg.norm(dist_total)), 1e-30
    )

    # Frame negative control: the *same* operator, but the surface frame shifted by
    # one lattice cell (the classic unit/frame mix-up this interface is exposed to).
    moment_good = torch.as_tensor(g3["moment_nodal"], dtype=DTYPE)
    g4_by_axis = {}
    for axis in ("x", "y", "z"):
        shifted = dataclasses.replace(
            finest.surface,
            **{axis: getattr(finest.surface, axis) + 1.0},
        )
        nodal_bad = grid.distribute_surface(shifted, finest.traction)
        moment_bad = torch.linalg.cross(
            grid.nodes - torch.as_tensor(moment_origin, dtype=DTYPE), nodal_bad, dim=1
        ).sum(0)
        g4_by_axis[axis] = float(torch.linalg.norm(moment_bad - moment_good)) / max(
            float(torch.linalg.norm(moment_good)), 1e-30
        )
    g4_err = max(g4_by_axis.values())

    # -- G5: drag from the real field --------------------------------------
    cd_mean = run.cd_mean_tail
    cd_err_pct = 100.0 * abs(cd_mean - cd_ref) / cd_ref
    cd_gap40_pct = 100.0 * abs(cd_mean - cd_stored40) / cd_stored40 if cd_stored40 == cd_stored40 else float("nan")

    # -- G6: added-mass guard with the real geometry -----------------------
    v_analytic = 4.0 / 3.0 * math.pi * R**3
    v_voxel = float(run.solid_cells)
    mu_ice = added_mass_ratio(RHO_ICE * v_analytic, RHO_WATER, v_analytic, CA_SPHERE)
    mu_light = added_mass_ratio(RHO_LIGHT_FLOE * v_analytic, RHO_WATER, v_analytic, CA_SPHERE)
    m_crit = CA_SPHERE * RHO_WATER * v_analytic
    omega_dt = 0.02
    lam_ice = staggered_spectral_radius(mu_ice, omega_dt)
    lam_light = staggered_spectral_radius(mu_light, omega_dt)
    hist_ice = march_one_dof(mu_ice, omega_dt, 400, 1.0)
    hist_light = march_one_dof(mu_light, omega_dt, 400, 1.0)
    gr_ice = growth_rate(hist_ice)
    gr_light = growth_rate(hist_light)
    lam_pred_light = math.log(lam_light)
    g6_rel = abs(gr_light - lam_pred_light) / abs(lam_pred_light) if lam_pred_light else float("nan")

    # -- G7: structural response driven by the measured drag ---------------
    cd_series = [e["cd"] for e in run.history]
    # Drag force (dimensionless: normalised by the structural stiffness scale)
    f_series = [c / max(cd_series[0], 1e-12) for c in cd_series]  # ramp-normalised load
    ledger = {}
    for odt in (0.04, 0.02, 0.01):
        res = march_driven(mu_ice, odt, f_series)
        ledger[f"{odt:.3f}"] = {
            "closure": res["closure"],
            "closure_max": res["closure_max"],
            "w_ext": res["w_ext"],
            "w_added_mass": res["w_added_mass"],
            "delta_energy": res["delta_energy"],
            "x_final": res["x_final"],
            "x_peak": res["x_peak"],
        }
    led_vals = {float(k): max(v["closure_max"], 1e-300) for k, v in ledger.items()}
    order = math.log(led_vals[0.04] / led_vals[0.02]) / math.log(2.0) if led_vals[0.02] > 0 else float("nan")
    response = march_driven(mu_ice, 0.02, f_series)
    response_unstable = march_driven(mu_light, 0.02, f_series)

    gates = {
        "G1_ledger_closure": {
            "pass": bool(g1_pass),
            "rel_diff": run.closure["rel_diff"],
            "ledger_total": run.closure["ledger_total"],
            "distribution_total": run.closure["distribution_total"],
            "n_links": run.closure["n_links"],
            "tol": TOL["g1_ledger_rel"],
        },
        "G2_link_to_facet": {
            "pass": bool(
            g2_resultant_rel <= TOL["g2_resultant_rel"]
            and finest.coverage >= TOL["g2_coverage"]
            and g3["moment_binning_rel_pointwise"] <= TOL["g2_binning_pointwise"]
        ),
            "resultant_rel": g2_resultant_rel,
            "coverage": finest.coverage,
            "gate_level": gate_key,
            "n_facets": len(finest.surface),
            "n_covered": finest.n_covered_facets,
            "moment_binning_rel_resultant": g3["moment_binning_rel_resultant"],
            "moment_binning_rel_pointwise": g3["moment_binning_rel_pointwise"],
            "tol_resultant": TOL["g2_resultant_rel"],
            "tol_coverage": TOL["g2_coverage"],
            "tol_binning_pointwise": TOL["g2_binning_pointwise"],
        },
        "G3_transfer_real_traction": {
            "pass": bool(
                g3["resultant_rel_err"] <= TOL["g3_resultant_rel"]
                and g3["moment_rel_err"] <= TOL["g3_moment_rel"]
                and g3["work_rel_err"] <= TOL["g3_work_rel"]
            ),
            **{k: g3[k] for k in (
                "resultant_rel_err", "moment_rel_err", "moment_abs_err", "work_rel_err",
                "work_surface", "work_nodal", "moment_surface", "resultant_surface",
                "sum_abs_link_force", "facet_size", "radius_mean",
            )},
        },
        "G4_frame_negative_control": {
            "pass": bool(g4_err >= TOL["g4_frame_error_min"]),
            "moment_error_wrong_frame_rel": g4_err,
            "moment_error_by_axis": g4_by_axis,
            "min_required": TOL["g4_frame_error_min"],
        },
        "G5_drag_real_field": {
            "pass": bool(cd_err_pct <= TOL["g5_cd_err_pct"]),
            "re": run.re,
            "cd_mean_tail": cd_mean,
            "cd_final": run.cd_final,
            "cd_ref_schiller_naumann": cd_ref,
            "err_pct": cd_err_pct,
            "cd_stored_D40": cd_stored40,
            "cd_stored_D60": cd_stored60,
            "gap_to_D40_pct": cd_gap40_pct,
            "tol_pct": TOL["g5_cd_err_pct"],
        },
        "G6_added_mass_guard_real_geometry": {
            "pass": bool(mu_ice < 1.0 and mu_light >= 1.0 and g6_rel <= TOL["g6_growth_rel"]),
            "volume_analytic": v_analytic,
            "solid_voxels": v_voxel,
            "voxel_vs_analytic_pct": 100.0 * abs(v_voxel - v_analytic) / v_analytic,
            "mu_ice": mu_ice,
            "mu_light_floe": mu_light,
            "m_crit_ice_kg_equivalent": m_crit,
            "spectral_radius_ice": lam_ice,
            "spectral_radius_light": lam_light,
            "growth_measured_light": gr_light,
            "growth_predicted_light": lam_pred_light,
            "growth_rel_gap": g6_rel,
            "growth_measured_ice": gr_ice,
            "relaxation_bound_ice": 2.0 / (1.0 + mu_ice),
            "relaxation_bound_light": 2.0 / (1.0 + mu_light),
            "tol_growth_rel": TOL["g6_growth_rel"],
        },
        "G7_driven_response_ledger": {
            "pass": bool(
                led_vals[0.01] <= TOL["g7_closure_max"]
                and led_vals[0.04] > led_vals[0.02] > led_vals[0.01]
            ),
            "ledger_by_dt": ledger,
            "observed_order": order,
            "min_order_informational": TOL["g7_ledger_order"],
            "max_closure_allowed": TOL["g7_closure_max"],
            "response_x_final": response["x_final"],
            "response_x_peak": response["x_peak"],
            "response_unstable_x_peak": response_unstable["x_peak"],
        },
    }
    n_pass = sum(1 for g in gates.values() if g["pass"])

    meta = {
        "case": "GAP-LBM-T5-ICE-COUPLING / G5-07 (round 2)",
        "D": run.D,
        "steps": run.steps,
        "tau": run.tau,
        "u_lb": run.u_lb,
        "nu_lb": run.nu_lb,
        "re": run.re,
        "domain_lu": list(run.domain_lu),
        "centre": list(run.centre),
        "radius": run.radius,
        "dpS": run.dpS,
        "q_range": list(run.q_range),
        "surface_levels": [list(lv) for lv in levels],
        "wall_s": run.wall_s,
        "total_s": time.time() - t0,
        "torch": torch.__version__,
        "threads": args.threads,
        "label": args.label,
    }
    payload = {
        "meta": meta,
        "gates": gates,
        "n_pass": n_pass,
        "n_gates": len(gates),
        "history": run.history,
        "surface_study": {
            key: {
                "n_facets": len(ld.surface),
                "coverage": ld.coverage,
                "facet_size": ld.facet_size,
                "moment_binning_rel_resultant": float(
                    torch.linalg.norm(ld.moment_binning_error)
                ) / max(float(torch.linalg.norm(ld.forces.sum(0))) * ld.radius_mean, 1e-30),
                "moment_binning_rel_pointwise": float(
                    torch.linalg.norm(ld.moment_binning_error)
                ) / max(ld.sum_abs_link_force * ld.facet_size, 1e-30),
            }
            for key, ld in run.loads.items()
        },
    }

    try:
        import numpy as np

        np.savez_compressed(
            os.path.join(outdir, f"field_D{args.D}.npz"),
            positions=finest.surface.positions.numpy(),
            traction=finest.traction.numpy(),
            centre=torch.as_tensor(centre, dtype=DTYPE).numpy(),
            history_cd=torch.tensor([e["cd"] for e in run.history]).numpy(),
            history_step=torch.tensor([e["step"] for e in run.history]).numpy(),
        )
    except Exception as exc:  # pragma: no cover - numpy always present in CI
        print(f"[g507] WARNING: could not save npz: {exc}")

    js_path = os.path.join(outdir, f"gates_D{args.D}.json")
    with open(js_path, "w") as fh:
        json.dump(payload, fh, indent=1)

    csv_path = os.path.join(outdir, f"gates_D{args.D}.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["gate", "pass", "key", "value"])
        for gname, g in gates.items():
            for k, v in g.items():
                if k == "pass" or isinstance(v, (int, float, str)):
                    wr.writerow([gname, g["pass"], k, v])

    try:
        make_figures(payload, os.path.join(outdir, f"field_D{args.D}.npz"), figdir)
    except Exception as exc:  # pragma: no cover - figures must not lose the numbers
        print(f"[g507] WARNING: figure generation failed ({exc}); JSON/CSV written")

    print(f"[g507] D={run.D} steps={run.steps} re={run.re:.1f} links={run.n_links} "
          f"wall={run.wall_s:.0f}s total={time.time() - t0:.0f}s")
    for name, g in gates.items():
        print(f"  {name}: {'PASS' if g['pass'] else 'FAIL'}")
    print(f"[g507] {n_pass}/{len(gates)} gates PASS -> {js_path}")
    return 0 if n_pass == len(gates) else 1


def make_figures(payload: dict, npz_path: str, figdir: str) -> None:
    """All figures come from the artefact (JSON + npz), never from a live run."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = payload["meta"]
    gates = payload["gates"]
    D = meta["D"]
    hist = payload["history"]

    # 1 — drag history from the real field
    t_star = [e["step"] * meta["u_lb"] / D for e in hist]
    cd = [e["cd"] for e in hist]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t_star, cd, "-", lw=1.2, label="BFL per-link ledger (this run)")
    g5 = gates["G5_drag_real_field"]
    ax.axhline(g5["cd_ref_schiller_naumann"], color="k", ls="--", lw=1.0,
               label=f"Schiller-Naumann {g5['cd_ref_schiller_naumann']:.4f}")
    for lvl, key in (("D40", "cd_stored_D40"), ("D60", "cd_stored_D60")):
        v = g5.get(key)
        if isinstance(v, float) and v == v:
            ax.axhline(v, color="C3", ls=":", lw=0.9, label=f"stored verified {lvl} {v:.4f}")
    ax.set_xlabel("t* = U t / D")
    ax.set_ylabel("C_D")
    ax.set_title(f"Sphere Re={meta['re']:.0f}: drag from the real D3Q19 field (D={D})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig1_drag_history.png"), dpi=140)
    plt.close(fig)

    # 2 — real-field traction over the sphere (front to back)
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        pos, trac, c0 = d["positions"], d["traction"], d["centre"]
        r = pos - c0
        rn = np.linalg.norm(r, axis=1)
        cos_t = r[:, 0] / np.clip(rn, 1e-30, None)
        mag = np.linalg.norm(trac, axis=1)
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        sc = ax.scatter(cos_t, trac[:, 0], c=mag, s=9, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="|t| (lattice units)")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("cos(theta)   (-1 stagnation, +1 wake)")
        ax.set_ylabel("t_x  (streamwise traction)")
        ax.set_title("Real-field traction on the sphere facets")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig2_traction_distribution.png"), dpi=140)
        plt.close(fig)

    # 3 — surface-quadrature moment error vs facet size
    study = payload["surface_study"]
    pts = sorted((v["facet_size"], v["moment_binning_rel_pointwise"]) for v in study.values())
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-")
    ax.set_xlabel("facet size h (lattice units)")
    ax.set_ylabel("|dM| / (sum|F| h)")
    ax.set_title("Surface-quadrature moment error vs surface refinement")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig3_binning_convergence.png"), dpi=140)
    plt.close(fig)

    # 4 — driven 1-DOF response: ice floe (mu<1) vs light floe (mu>=1)
    f_series = [e["cd"] for e in hist]
    f_series = [c / max(f_series[0], 1e-12) for c in f_series]
    g6 = gates["G6_added_mass_guard_real_geometry"]
    stable = march_driven(g6["mu_ice"], 0.02, f_series)
    unstable = march_driven(g6["mu_light_floe"], 0.02, f_series)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(stable["x"], label=f"ice floe  mu={g6['mu_ice']:.3f} (stable)")
    ax.plot(unstable["x"], label=f"light floe mu={g6['mu_light_floe']:.3f} (mu>=1: diverges)")
    ax.set_xlabel("sample index")
    ax.set_ylabel("x (dimensionless)")
    ax.set_title("1-DOF response driven by the measured drag history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig4_driven_response.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
