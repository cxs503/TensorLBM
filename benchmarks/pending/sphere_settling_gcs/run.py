#!/usr/bin/env python3
"""G15 third method: single-sphere Stokes settling measured with a
Galilean-invariant kinetic-flux control surface (GCS).

Physical problem (settling sphere, body frame)
----------------------------------------------
A periodic simple-cubic cell of side ``L`` contains one voxel sphere of
radius ``R`` held fixed; a uniform force density ``ρ g`` acts on the *fluid*
nodes (gravity).  At steady state the sphere is in equilibrium with the
drag, so the fluid drifts at exactly the terminal velocity ``U_t`` that the
sphere would have under the same gravity in a periodic suspension —
i.e. this *is* the sedimentation problem, in the frame where the sphere is
at rest (the two frames differ by a constant translation; the flow is steady
in both).  No moving boundary and no force feedback loop are involved: the
only measured quantity is the force.

Force instrument (the deliverable)
----------------------------------
``src/tensorlbm/force_control_surface.py`` — ``F = f·V_f^in − ∮_S Π·n dA``
with ``Π_ij = Σ_q f_q c_qi c_qj`` (raw second moment, half-force correction
for the Guo forcing scheme) integrated on an analytic sphere of radius
``R + 2…3`` cells.  Independent of the wall treatment, background-free by
construction (``∮ n dA = 0`` exactly).

References used for the verdict
-------------------------------
* unbounded Stokes settling: ``U_s = 2 a² Δρ g / (9 μ)`` ⇔ ``F = 6 π μ a U``.
* finite-Re correction (for the boundary statement only): Schiller–Naumann /
  Oseen ``F/F_Stokes = 1 + 3Re/8``.
* periodic-array confinement (cross-check only, never baked into a number):
  Hasimoto (1959) ``K(λ)``, ``λ = a/L``.

Usage
-----
    python run.py all            # full matrix (see STUDIES)
    python run.py domain|refine|reynolds|anchor
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask  # noqa: E402
from tensorlbm.control_volume_force import box_control_volume, observe_control_volume_force  # noqa: E402
from tensorlbm.d3q19 import C, W, equilibrium3d  # noqa: E402
from tensorlbm.force_control_surface import ControlSurfaceForceTracker, hasimoto_factor  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, macroscopic3d, stream3d  # noqa: E402

OUT_DIR = _HERE.parent
CS2, CS4 = 1.0 / 3.0, 1.0 / 9.0

#: viscosity used for every physics run: tau = 0.5 + 3 nu = 3.5.  See
#: docs/sphere_settling_gcs_20261107.md §5: the third-order-accurate regime of
#: this solver chain is tau <= ~4; larger tau is a *wall-model* artefact that
#: would contaminate the physics being measured.
NU_LATTICE = 1.0
TAU = 0.5 + 3.0 * NU_LATTICE


def guo_source(rho, ux, uy, uz, fx, tau):
    """Guo (2002) forcing source for a force density ``f`` in +x."""
    c = C.to(rho.device).float()
    w = W.to(rho.device).float().view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux
        + c[:, 1].view(19, 1, 1, 1) * uy
        + c[:, 2].view(19, 1, 1, 1) * uz
    )
    cF = c[:, 0].view(19, 1, 1, 1) * fx
    return (1.0 - 0.5 / tau) * w * (cF / CS2 + cu * cF / CS4)


def run_case(
    *,
    R: float,
    L: int,
    re: float,
    steps: int = 3000,
    sample_every: int = 25,
    r_offsets: tuple[float, ...] = (2.0, 2.5, 3.0),
    tail: int = 20,
    box_cv: bool = False,
    log_every: int = 1000,
) -> dict:
    """One periodic settling cell; returns a JSON-able record."""
    torch.manual_seed(0)
    nz = ny = nx = int(L)
    centre = (L / 2.0, L / 2.0, L / 2.0)
    nu = NU_LATTICE
    tau = TAU
    u_target = re * nu / (2.0 * R)          # Stokes terminal velocity target
    solid = sphere_mask(nx, ny, nz, centre[0], centre[1], centre[2], R, device=torch.device("cpu"))
    n_solid = int(solid.sum().item())
    v_cell = float(L) ** 3
    v_fluid = v_cell - n_solid
    f_stokes = 6.0 * math.pi * nu * R * u_target
    g = f_stokes / v_fluid                  # force density (rho = 1)
    lam = R / L
    r_vol = (3.0 * n_solid / (4.0 * math.pi)) ** (1.0 / 3.0)

    rho0 = torch.ones(nz, ny, nx)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0
    fluid = ~solid
    trackers = {
        float(off): ControlSurfaceForceTracker(centre=centre, radius=R + float(off), periodic=True)
        for off in r_offsets
    }
    panel = None
    if box_cv:
        panel = box_control_volume(
            (nz, ny, nx),
            x0=int(centre[0] - R - 4),
            x1=int(centre[0] + R + 4),
            y0=int(centre[1] - R - 4),
            y1=int(centre[1] + R + 4),
            z0=int(centre[2] - R - 4),
            z1=int(centre[2] + R + 4),
            periodic_axes=("x", "y", "z"),
        )
    box_samples: list[tuple[float, float, float]] = []
    u_hist: list[float] = []

    tag = f"R{R:g}_L{L}_Re{re:g}"
    print(
        f"[{tag}] L={L}^3 nu={nu:g} tau={tau:g} U_target={u_target:.6g} "
        f"g={g:.4e} n_solid={n_solid} r_vol={r_vol:.4f} F_stokes={f_stokes:.6f}",
        flush=True,
    )
    t0 = time.time()
    step_i = 0
    for step_i in range(1, steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        fx = rho * g
        f_new = collide_bgk3d(f, tau) + guo_source(rho, ux, uy, uz, fx, tau)
        f_new[:, solid] = f[:, solid]
        f_old = f
        f_pre_stream = bounce_back_cells_3d(f_new, solid)   # half-way BB (pre-stream)
        f = stream3d(f_pre_stream)                          # periodic streaming
        if step_i % sample_every == 0:
            for tr in trackers.values():
                tr.sample(f)
            rho2, ux2, _, _ = macroscopic3d(f)
            u_hist.append(float((rho2 * ux2)[fluid].sum() / rho2[fluid].sum()))
            if panel is not None:
                res = observe_control_volume_force(
                    f_old, f, f_pre_stream, panel, solid=solid, periodic_axes=("x", "y", "z")
                )
                box_samples.append(res.force_tuple)
        if not torch.isfinite(f).all():
            raise RuntimeError(f"[{tag}] diverged at step {step_i}")
        if step_i % log_every == 0:
            s = trackers[float(r_offsets[0])].summary("x", last=tail)
            print(
                f"[{tag}] step={step_i} U={u_hist[-1]:.6f} Fx_cs={s['mean']:.6f} "
                f"osc_rms={100 * s['rms_over_mean']:.4f}% ({time.time() - t0:.0f}s)",
                flush=True,
            )
    wall = time.time() - t0

    # fluid volume enclosed by each control sphere (for the body-force term)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    rec: dict = {
        "tag": tag,
        "R": R,
        "L": L,
        "lambda_R_over_L": lam,
        "Re_target": re,
        "Re_measured": None,
        "nu": nu,
        "tau": tau,
        "u_stokes_unbounded": u_target,
        "g": g,
        "steps": step_i,
        "sample_every": sample_every,
        "wall_s": wall,
        "n_solid_cells": n_solid,
        "r_volume_equivalent": r_vol,
        "F_stokes_target": f_stokes,
        "hasimoto_factor": hasimoto_factor(lam),
        "control_surface": {},
        "u_series": u_hist,
    }
    u_tail = sum(u_hist[-tail:]) / min(tail, len(u_hist))
    rec["u_measured"] = u_tail
    rec["Re_measured"] = u_tail * 2.0 * R / nu
    for off, tr in trackers.items():
        r_cv = R + off
        ball = ((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2) <= r_cv**2
        v_in = float((ball & ~solid).sum())
        sx = tr.summary("x", last=tail)
        f_cs = sx["mean"]
        f_drag = g * v_in + f_cs                 # force on the body (drag-positive)
        drift = tr.momentum_drift(dt_steps=sample_every, last=tail)
        ser = tr.series("x")
        k = min(tail, len(ser))
        w = ser[-k:]
        h = len(w) // 2
        early = sum(w[:h]) / max(1, h)
        late = sum(w[h:]) / max(1, len(w) - h)
        rec["control_surface"][f"{off:g}"] = {
            "radius": r_cv,
            "v_fluid_inside": v_in,
            "fx_control_surface": f_cs,
            "fx_body_force_term": g * v_in,
            "fx_total": f_drag,
            "fx_rms": sx["rms"],
            "fx_ptp": sx["ptp"],
            "rms_over_mean": sx["rms_over_mean"],
            "ptp_over_mean": sx["ptp_over_mean"],
            "tail_half_drift_rel": abs(late - early) / abs(late) if late else None,
            "d_momentum_dt": drift,
            "F_stokes_at_measured_U": 6.0 * math.pi * nu * R * u_tail,
            "drag_over_stokes_at_measured_U": f_drag / (6.0 * math.pi * nu * R * u_tail),
            "u_terminal_over_stokes": u_target / u_tail if u_tail else None,
            "series": ser,
        }
    if box_samples:
        k = min(tail, len(box_samples))
        rec["box_cv"] = {
            "fx_mean": sum(s[0] for s in box_samples[-k:]) / k,
            "fy_mean": sum(s[1] for s in box_samples[-k:]) / k,
            "n_samples": len(box_samples),
        }
        rec["box_cv"]["fx_total_with_body_force"] = rec["box_cv"]["fx_mean"]
    print(
        f"[{tag}] FINAL U={u_tail:.6f} (U/U_stokes={u_target / u_tail:.4f}) "
        f"F={rec['control_surface']['2.5']['fx_total']:.6f} "
        f"F/F_stokes(U)={rec['control_surface']['2.5']['drag_over_stokes_at_measured_U']:.4f} "
        f"({wall:.0f}s)",
        flush=True,
    )
    return rec


# ---------------------------------------------------------------------------
def study_domain():
    """Confinement study: λ = R/L at fixed R and Re (linear-in-λ extrapolation)."""
    return [run_case(R=4.0, L=L, re=0.05) for L in (32, 40, 48)]


def study_refine():
    """Body representation: R at fixed λ = 1/8 and Re."""
    return [run_case(R=R, L=int(8 * R), re=0.05) for R in (4.0, 6.0, 8.0)]


def study_reynolds():
    """Method boundary in Re at fixed R and λ: τ stays at 3.5, U scales with Re."""
    return [run_case(R=6.0, L=48, re=re) for re in (0.02, 0.05, 0.10, 0.20)]


def study_anchor():
    """Exact momentum-balance anchor: F on the body must equal ρ g V_fluid."""
    return [run_case(R=6.0, L=48, re=0.05, steps=6000, box_cv=True, log_every=2000)]


def _fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx
    b = my - k * mx
    return k, b, max(abs(y - (b + k * x)) for x, y in zip(xs, ys))


def verdict(domain, refine, reynolds) -> dict:
    tol = 0.03
    lam = [r["lambda_R_over_L"] for r in domain]
    ut = [r["control_surface"]["2.5"]["u_terminal_over_stokes"] for r in domain]
    k_lam, b_lam, resid_lam = _fit(lam, ut)
    invR = [1.0 / r["R"] for r in refine]
    fr = [r["control_surface"]["2.5"]["drag_over_stokes_at_measured_U"] for r in refine]
    k_R, b_R, resid_R = _fit(invR, fr)
    osc = max(
        r["control_surface"][kk]["ptp_over_mean"]
        for r in domain + refine + reynolds
        for kk in r["control_surface"]
    )
    spread = []
    for r in domain + refine + reynolds:
        vals = [r["control_surface"][kk]["fx_total"] for kk in r["control_surface"]]
        spread.append((max(vals) - min(vals)) / (sum(vals) / len(vals)))
    checks = {
        # ① terminal velocity vs analytic Stokes (λ→0 extrapolation)
        "J1_domain_extrapolated_U_over_Stokes": b_lam,
        "J1_err_pct": 100.0 * (b_lam - 1.0),
        "J1_within_3pct": abs(b_lam - 1.0) <= tol,
        "J1_fit_residual": resid_lam,
        "J1_slope": k_lam,
        # drag law at measured U, extrapolated in 1/R
        "J2_drag_over_stokes_R_inf": b_R,
        "J2_err_pct": 100.0 * (b_R - 1.0),
        "J2_within_3pct": abs(b_R - 1.0) <= tol,
        "J2_fit_residual": resid_R,
        # ② oscillation amplitude
        "J4_max_ptp_over_mean": osc,
        "J4_within_2pct": osc <= 0.02,
        # ③ control-radius independence
        "J6_max_radius_spread": max(spread),
        "J6_within_1p5pct": max(spread) <= 0.015,
        # finite-Re boundary statement
        "Reynolds_scan": {
            f"{r['Re_measured']:.4f}": {
                "u_over_stokes": r["control_surface"]["2.5"]["u_terminal_over_stokes"],
                "drag_over_stokes_at_U": r["control_surface"]["2.5"][
                    "drag_over_stokes_at_measured_U"
                ],
                "oseen_factor": 1.0 + 3.0 / 8.0 * r["Re_measured"],
            }
            for r in reynolds
        },
    }
    checks["all_pass"] = all(v for v in checks.values() if isinstance(v, bool))
    return {"domain_fit": {"slope": k_lam, "intercept": b_lam, "max_resid": resid_lam},
            "grid_fit": {"slope": k_R, "intercept": b_R, "max_resid": resid_R},
            "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["all", "domain", "refine", "reynolds", "anchor"])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / "result.json"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    steps = args.steps

    def kw(**extra):
        return extra if not steps else {**extra, "steps": steps}

    out: dict = {
        "case": "sphere_settling_gcs",
        "estimator": "galilean_invariant_kinetic_flux_control_surface",
        "module": "src/tensorlbm/force_control_surface.py",
        "lattice": "D3Q19",
        "collision": "bgk",
        "nu_lattice": NU_LATTICE,
        "tau": TAU,
        "runs": [],
    }
    t0 = time.time()
    domain, refine, reyn, anchor = [], [], [], []
    if args.study in ("all", "domain"):
        domain = [run_case(**kw(R=4.0, L=L, re=0.05)) for L in (32, 40, 48)]
        out["runs"] += domain
    if args.study in ("all", "refine"):
        refine = [run_case(**kw(R=R, L=int(8 * R), re=0.05)) for R in (4.0, 6.0, 8.0)]
        out["runs"] += refine
    if args.study in ("all", "reynolds"):
        reyn = [run_case(**kw(R=6.0, L=48, re=re)) for re in (0.02, 0.05, 0.10, 0.20)]
        out["runs"] += reyn
    if args.study in ("all", "anchor"):
        anchor = [
            run_case(
                R=6.0, L=48, re=0.05, steps=steps or 4000, box_cv=True, log_every=1000
            )
        ]
        out["runs"] += anchor
    # verdict needs the three studies; reuse whatever was run
    by_key = {(r["R"], r["L"]): r for r in out["runs"]}
    if not domain:
        domain = [by_key[(4.0, 32)]] if (4.0, 32) in by_key else []
    if not refine:
        refine = [by_key[k] for k in ((4.0, 32), (6.0, 48), (8.0, 64)) if k in by_key]
    if not reyn:
        reyn = [r for r in out["runs"] if (r["R"], r["L"]) == (6.0, 48)]
    if len(domain) >= 2 and len(refine) >= 2:
        out["verdict"] = verdict(domain, refine, reyn)
    if anchor:
        a = anchor[0]
        out["anchor"] = {
            "F_total": a["control_surface"]["2.5"]["fx_total"],
            "rho_g_Vfluid": a["g"] * (a["L"] ** 3 - a["n_solid_cells"]),
            "rel_err_pct": 100.0
            * (
                a["control_surface"]["2.5"]["fx_total"]
                / (a["g"] * (a["L"] ** 3 - a["n_solid_cells"]))
                - 1.0
            ),
            "box_cv_fx": a.get("box_cv", {}).get("fx_mean"),
        }
    out["wall_s_total"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1))
    if "verdict" in out:
        print(json.dumps(out["verdict"]["checks"], indent=1))
    if "anchor" in out:
        print("anchor:", json.dumps(out["anchor"]))
    print(f"wrote {args.out} ({out['wall_s_total']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())