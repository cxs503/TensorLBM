#!/usr/bin/env python3
"""XVAL-LBM-WALL-ORIENTATION-CONTACT-ANGLE — harness (anchor round 3).

Anchor
------
``XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS`` validated two *independent*
contact-angle wall conditions on a sessile drop resting on a **horizontal**
plate.  This round asks the question the horizontal-plate matrix cannot answer:
is the discretised wall condition **orientation invariant**?  A 90 deg rotation
of one physical experiment is the same experiment (the continuum problem is
isotropic and gravity is switched off), so all four walls of the box must
reproduce the same Young angle, by the same two arms, through the production
D2Q9 CAC solver (:class:`tensorlbm.cac_lbm2d.CAC2D`).

Arms
----
``contact``      angle-driven wall ghost ``phi_s = phi_f + cos(theta) 4 phi(1-phi)/W``.
``wall_order``   wetting from a wall order parameter ``psi_w`` through the wall free
                 energy, ``theta_Y = acos((3 psi_w - psi_w^3)/2)`` -- no angle input.

Orientations
------------
``bottom`` / ``top``     plate = 2 solid rows, drop on the plate.
``left`` / ``right``     plate = 2 solid columns (the 90 deg rotation).
The configuration is built in a canonical frame (wall at the bottom), mapped to
the target frame by :func:`from_canonical`, and the final field is mapped back
by :func:`to_canonical` before the (wall-at-bottom) estimators see it, so the
measurement code is identical for all four orientations.

Usage
-----
    python3 run_xval.py sweep
    python3 run_xval.py one --arm contact --theta 90 --orientation left
    python3 run_xval.py summarize
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

PREDECLARED = json.loads((HERE / "predeclared.json").read_text())
RUNS = HERE / "runs"
M = PREDECLARED["matrix"]
TH = PREDECLARED["predeclared_thresholds"]


# ---------------------------------------------------------------------------
# frame transforms: canonical frame = wall at the BOTTOM of the array
# ---------------------------------------------------------------------------
def to_canonical(a: np.ndarray, side: str) -> np.ndarray:
    """Map a field whose wall is at ``side`` onto the wall-at-bottom frame."""
    if side == "bottom":
        return a
    if side == "top":
        return a[::-1, :]
    if side == "left":  # (y, x) -> (y'=x, x'=ny-1-y)
        return a[::-1, :].T
    if side == "right":  # (y, x) -> (y'=nx-1-x, x'=y)
        return a[:, ::-1].T
    raise ValueError(f"unknown side {side!r}")


def from_canonical(a: np.ndarray, side: str) -> np.ndarray:
    """Inverse of :func:`to_canonical`: wall-at-bottom frame -> target frame."""
    if side == "bottom":
        return a
    if side == "top":
        return a[::-1, :]
    if side == "left":
        return a.T[::-1, :]
    if side == "right":
        return a.T[:, ::-1]
    raise ValueError(f"unknown side {side!r}")


def wall_mask_canonical(shape, thickness: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:thickness, :] = True
    return mask


def cap_field(shape, radius: float, width: float, theta_deg: float, y_wall: float) -> np.ndarray:
    """Circular cap of radius ``radius`` meeting the wall at ``theta`` (canonical frame)."""
    ny, nx = shape
    th = math.radians(theta_deg)
    cy = y_wall - radius * math.cos(th)
    cx = nx / 2.0
    y = np.arange(ny, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]
    d = radius - np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return 0.5 + 0.5 * np.tanh(2.0 * d / width)


def axes_policy(side: str) -> str:
    """Halo policy that follows the wall: ``w`` (zero fill) across the walls, ``p`` along them.

    A 90 deg rotation of a case must rotate the halo policy with the geometry.
    Keeping ``axes="wp"`` for a left-hand wall silently replaces the periodic
    direction by a zero-filled one and produces a spurious orientation
    difference of O(1) in the wall rows (measured: see
    ``tests/test_xval_wall_orientation_contact_angle.py``, negative control).
    """
    return "wp" if side in ("bottom", "top") else "pw"


def tag_of(arm: str, theta: float, side: str) -> str:
    return f"{arm}__th{theta:05.1f}__{side}"


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def run_one(args) -> dict:
    arm, theta_deg, side = args
    import torch

    torch.set_num_threads(1)
    from tensorlbm import cac_lbm2d as cac
    from tensorlbm.wetting import metrics as MM
    from tensorlbm.wetting.two_impl_contact_angle import psi_wall_from_angle
    from tensorlbm.wetting.wall_order_parameter_2d import analytic_contact_angle

    t0 = time.time()
    shape = tuple(M["shape"])
    t = int(M["wall_thickness"])
    width = float(M["width"])
    radius = float(M["radius"])
    y_wall = t - 0.5

    psi_wall = None
    theta_ref = float(theta_deg)
    if arm == "contact":
        policy, angle = "contact", float(theta_deg)
        wall_order_parameter = None
    elif arm == "wall_order":
        policy, angle = "wall_order", None
        psi_wall = float(psi_wall_from_angle(math.radians(theta_deg)))
        wall_order_parameter = psi_wall
        theta_ref = math.degrees(float(analytic_contact_angle(psi_wall)))
    else:
        raise ValueError(arm)

    mask_c = wall_mask_canonical(shape, t)
    phi_c = cap_field(shape, radius, width, theta_ref, y_wall)

    mask = torch.from_numpy(from_canonical(mask_c, side).copy())
    phi0 = torch.from_numpy(from_canonical(phi_c, side).copy())

    sim = cac.CAC2D(
        shape,
        sigma=float(M["surface_tension"]),
        width=width,
        rho_l=PREDECLARED["numerics"]["rho_l"],
        rho_g=PREDECLARED["numerics"]["rho_g"],
        mu_l=PREDECLARED["numerics"]["mu_l"],
        mu_g=PREDECLARED["numerics"]["mu_g"],
        mobility=PREDECLARED["numerics"]["mobility"],
        mu_law=PREDECLARED["numerics"]["mu_law"],
        p0=PREDECLARED["numerics"]["p0"],
        wall_mask=mask,
        wall_phi_policy=policy,
        contact_angle_deg=angle,
        wall_order_parameter=wall_order_parameter,
        gravity=tuple(PREDECLARED["numerics"]["gravity"]),
        gravity_mode="rho",
        axes=axes_policy(side),
        dtype=torch.float32,
    )
    sim.initialize(phi0)
    diag = cac.run(sim, int(M["steps"]), sample_every=int(M["sample_every"]))
    phi_phys = sim.f.sum(dim=0).detach().cpu().numpy().astype(np.float64)
    phi = to_canonical(phi_phys, side)

    try:
        circ = MM.contact_angle_circle_fit(phi, y_wall)
    except Exception:  # too few contour points: unresolved cap -> NOT_MEASURABLE, not a crash
        circ = {"theta_deg": float("nan"), "rms": float("nan")}
    try:
        cap = MM.spherical_cap_inversion(phi, y_wall)
    except Exception:
        cap = {"theta_deg": float("nan")}
    est = [float(circ["theta_deg"]), float(cap["theta_deg"])]
    finite = [e for e in est if math.isfinite(e)]
    theta_meas = float(np.median(finite)) if finite else float("nan")

    apex = float(phi.shape[0] - t - np.argmax((phi > 0.5)[::-1].any(axis=1))) if bool((phi > 0.5).any()) else 0.0
    rms = float(circ.get("rms", float("nan")))
    measurable = bool(
        math.isfinite(theta_meas)
        and apex >= TH["min_apex_over_width"] * width
        and math.isfinite(rms)
        and rms <= TH["circle_rms_over_width"] * width
    )
    errs = [abs(e - theta_ref) / theta_ref * 100.0 for e in finite]
    passed = bool(measurable and len(errs) == 2 and max(errs) <= TH["accept_pct"])

    payload = {
        "tag": tag_of(arm, theta_deg, side),
        "anchor": PREDECLARED["anchor"],
        "round": PREDECLARED["round"],
        "arm": arm,
        "wall_policy": policy,
        "orientation": side,
        "axes_policy": axes_policy(side),
        "theta_prescribed_deg": float(theta_deg),
        "theta_ref_deg": theta_ref,
        "psi_wall": psi_wall,
        "theta_measured_deg": theta_meas,
        "theta_circle_deg": est[0],
        "theta_cap_deg": est[1],
        "err_circle_pct": abs(est[0] - theta_ref) / theta_ref * 100.0 if math.isfinite(est[0]) else None,
        "err_cap_pct": abs(est[1] - theta_ref) / theta_ref * 100.0 if math.isfinite(est[1]) else None,
        "estimator_gap_deg": abs(est[0] - est[1]) if all(math.isfinite(e) for e in est) else None,
        "apex_height": apex,
        "apex_over_width": apex / width,
        "circle_rms": rms,
        "circle_rms_over_width": rms / width,
        "measurable": measurable,
        "pass": passed,
        "u_max": float(diag[-1]["u_max_fluid"]) if diag else None,
        "inputs": {
            "shape": list(shape),
            "radius": radius,
            "wall_thickness": t,
            "width": width,
            "surface_tension": float(M["surface_tension"]),
            "steps": int(M["steps"]),
            "gravity": PREDECLARED["numerics"]["gravity"],
            "solver": PREDECLARED["numerics"]["solver"],
            **{k: PREDECLARED["numerics"][k] for k in ("rho_l", "rho_g", "mu_l", "mu_g", "mobility", "mu_law", "p0")},
        },
        "wall_seconds": time.time() - t0,
    }
    _write(payload, phi if theta_deg == 90.0 else None)
    return payload


def _write(payload: dict, phi: np.ndarray | None) -> None:
    d = RUNS / payload["tag"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    if phi is not None:
        np.savez_compressed(d / "phi.npz", phi=phi)


# ---------------------------------------------------------------------------
# sweep / summary
# ---------------------------------------------------------------------------
def job_list() -> list:
    jobs = []
    for arm in PREDECLARED["arms"]:
        for theta in M["angles_deg"]:
            for side in PREDECLARED["orientations"]:
                jobs.append((arm, float(theta), side))
    return jobs


def sweep(shard: int = 0, nshards: int = 1) -> None:
    jobs = job_list()
    if nshards > 1:
        jobs = [j for k, j in enumerate(jobs) if k % nshards == shard]
    print(f"{len(jobs)} runs (shard {shard}/{nshards})", flush=True)
    for job in jobs:
        p = run_one(job)
        print(
            f"{p['tag']:34s} ref={p['theta_ref_deg']:6.1f} circ={p['theta_circle_deg']:7.2f} "
            f"cap={p['theta_cap_deg']:7.2f} meas={int(p['measurable'])} pass={int(p['pass'])} "
            f"apex/W={p['apex_over_width']:5.2f} {p['wall_seconds']:.0f}s",
            flush=True,
        )


def summarize() -> dict:
    rows = [json.loads(f.read_text()) for f in sorted(RUNS.glob("*/result.json"))]
    if not rows:
        raise SystemExit("no runs")

    def cell(arm, theta, side):
        for r in rows:
            if r["arm"] == arm and r["theta_prescribed_deg"] == theta and r["orientation"] == side:
                return r
        return None

    per_arm = {}
    for arm in PREDECLARED["arms"]:
        sub = [r for r in rows if r["arm"] == arm]
        meas = [r for r in sub if r["measurable"]]
        errs = [max(r["err_circle_pct"], r["err_cap_pct"]) for r in meas if r["err_circle_pct"] is not None]
        per_arm[arm] = {
            "n_rows": len(sub),
            "n_measurable": len(meas),
            "n_pass": sum(1 for r in meas if r["pass"]),
            "max_err_pct": float(max(errs)) if errs else None,
            "median_err_pct": float(np.median(errs)) if errs else None,
            "max_circle_rms_over_width": float(np.nanmax([r["circle_rms_over_width"] for r in sub])),
        }

    mirror = []
    orient = []
    cross = []
    for arm in PREDECLARED["arms"]:
        for theta in M["angles_deg"]:
            pairs = {k: cell(arm, theta, k) for k in PREDECLARED["orientations"]}
            for a, b in (("bottom", "top"), ("left", "right")):
                ra, rb = pairs[a], pairs[b]
                if ra is None or rb is None or not (ra["measurable"] and rb["measurable"]):
                    continue
                d = abs(ra["theta_measured_deg"] - rb["theta_measured_deg"])
                mirror.append({"arm": arm, "theta_prescribed_deg": theta, "pair": [a, b], "diff_deg": d,
                               "pass": bool(d <= TH["mirror_deg"])})
            for a in ("left", "right"):
                ra, rb = pairs["bottom"], pairs[a]
                if ra is None or rb is None or not (ra["measurable"] and rb["measurable"]):
                    continue
                d = abs(ra["theta_measured_deg"] - rb["theta_measured_deg"]) / theta * 100.0
                orient.append({"arm": arm, "theta_prescribed_deg": theta, "pair": ["bottom", a], "diff_pct": d,
                               "pass": bool(d <= TH["orientation_pct"])})
    for theta in M["angles_deg"]:
        for side in PREDECLARED["orientations"]:
            a, b = cell("contact", theta, side), cell("wall_order", theta, side)
            if a is None or b is None or not (a["measurable"] and b["measurable"]):
                continue
            d = abs(a["theta_measured_deg"] - b["theta_measured_deg"]) / theta * 100.0
            cross.append({"theta_prescribed_deg": theta, "orientation": side,
                          "theta_contact_deg": a["theta_measured_deg"],
                          "theta_wall_order_deg": b["theta_measured_deg"],
                          "diff_pct": d, "pass": bool(d <= TH["cross_arm_pct"])})

    anchor_pass = bool(
        per_arm["contact"]["n_measurable"] > 0
        and per_arm["wall_order"]["n_measurable"] > 0
        and all(per_arm[a]["n_pass"] == per_arm[a]["n_measurable"] for a in PREDECLARED["arms"])
        and mirror and all(m["pass"] for m in mirror)
        and orient and all(o["pass"] for o in orient)
        and cross and all(c["pass"] for c in cross)
    )
    summary = {
        "anchor": PREDECLARED["anchor"],
        "round": PREDECLARED["round"],
        "n_rows": len(rows),
        "thresholds": TH,
        "per_arm": per_arm,
        "mirror": mirror,
        "orientation": orient,
        "cross_arm": cross,
        "anchor_pass": anchor_pass,
        "verdict": "PASS" if anchor_pass else "FAIL",
        "rows": rows,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sweep", "one", "summarize"])
    ap.add_argument("--arm", default="contact")
    ap.add_argument("--theta", type=float, default=90.0)
    ap.add_argument("--orientation", default="bottom")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.action == "sweep":
        sweep(a.shard, a.nshards)
    elif a.action == "one":
        print(json.dumps(run_one((a.arm, a.theta, a.orientation)), indent=2, default=str))
    else:
        s = summarize()
        print(json.dumps({k: s[k] for k in ("n_rows", "per_arm", "anchor_pass", "verdict")}, indent=2))


if __name__ == "__main__":
    main()