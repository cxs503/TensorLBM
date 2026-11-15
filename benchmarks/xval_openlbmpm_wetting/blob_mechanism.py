#!/usr/bin/env python3
"""Mechanism attribution for the "far-wall blob" of the sessile-drop cross-validation.

Context
-------
Round 1/2 of ``XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`` (``PorousMediaSimulation/
openLBMPM`` geometric wetting) left the *end-to-end* sessile-drop contact angle
unaccepted: the host solver condenses a second blob of the wetting (red) phase on
the far wall of the domain, which siphons volume from the drop and corrupts both
shape estimators.  R2 quantified the blob (``blob_area``) without explaining it.

This probe is read-only instrumentation.  Per time sample it records, for the
wall-attached drop and for every stray blob:

* ``blob_area`` / ``n_blobs`` / ``blob_centroid_x`` / ``blob_centroid_y``
  (R2 only recorded ``y``; ``x`` is what discriminates a periodic image of the
  drop from a de-novo nucleus elsewhere);
* ``top_band_mass`` -- the summed ``phi`` in the top ``band`` fluid rows;
* ``film_mass`` -- ``phi`` on the first fluid row *outside* the drop footprint
  (the equilibrium wetting film that a wetting wall is supposed to host);
* ``distance_to_wall_rows`` for the blob, i.e. its distance in rows to the
  nearest solid wall, so a blob hugging the top wall can be told apart from one
  floating in the bulk.

Candidate mechanisms (to be accepted/rejected by the numbers, not by taste):

H1 ``periodic_image``  the top boundary is streaming-periodic, so the top rows
   are *coupled to the solid bottom wall*; the blob is the wetting film / drop
   re-appearing at the periodic image of the wall.  Prediction: blob centroid
   ``y`` tracks ``ny - 1 - const`` when ``ny`` grows, blob ``x`` == drop ``x``.
H2 ``recoloring_nucleation``  the anti-diffusive recolouring amplitude ``beta``
   nucleates a droplet out of the diffuse tails.  Prediction: blob area falls
   sharply with ``beta`` and is insensitive to ``ny`` / the top boundary kind.
H3 ``wetting_bc_siphon``  the wetting boundary condition itself drives the film
   and the film pinches off.  Prediction: blob area scales with wettability
   (small for theta>90) and vanishes with the BC disabled.

Usage
-----
python benchmarks/xval_openlbmpm_wetting/blob_mechanism.py \
       --cases base,tall,topwall90,topwall60,nowall,beta03 --steps 2000 \
       --ny 80 --nx 96 --radius 20 --theta 60 --out runs/xval_blob_mechanism
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.cg_wetting_geometric import sessile_drop_initial  # noqa: E402
from tensorlbm.solver import OPPOSITE_2D, stream  # noqa: E402
from tensorlbm.stream_boxed import seam_leak_fraction, stream_boxed  # noqa: E402
from wall_drop_measure import measure_wall_drop  # noqa: E402

# canonical D2Q9 opposite map (solver.OPPOSITE_2D).  R1/R2 used
# (0,3,2,1,4,7,8,5,6), which mapped the vertical directions onto
# themselves -- see docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE-R3-*.md
OPP = tuple(int(v) for v in OPPOSITE_2D)

# case -> (ny, top kind, top contact angle, wetting bc on/off, beta override)
CASES: dict[str, dict] = {
    "base": dict(top="open", top_theta=None, bc=True, beta=None),
    "boxed": dict(top="wall", top_theta=90.0, bc=True, beta=None, boxed=True),
    "boxed60": dict(top="wall", top_theta=None, bc=True, beta=None, boxed=True),
    "boxedopen": dict(top="open", top_theta=None, bc=True, beta=None, boxed=True),
    "tall": dict(top="open", top_theta=None, bc=True, beta=None, ny_scale=2.0),
    "wide": dict(top="open", top_theta=None, bc=True, beta=None, nx_scale=2.0),
    "topwall90": dict(top="wall", top_theta=90.0, bc=True, beta=None),
    "topwall60": dict(top="wall", top_theta=60.0, bc=True, beta=None),
    "nowall": dict(top="open", top_theta=None, bc=False, beta=None),
    "beta03": dict(top="open", top_theta=None, bc=True, beta=0.3),
    "beta00": dict(top="open", top_theta=None, bc=True, beta=0.0),
}


def _blob_stats(phi: np.ndarray, solid: np.ndarray, wall_row: int = 1) -> dict:
    """Blob census with x-centroid and wall distances (superset of R2's measure)."""
    mask = (phi > 0.0) & (~solid)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    out = {"n_blobs": 0, "blob_area": 0.0, "blob_centroid_x": float("nan"),
           "blob_centroid_y": float("nan"), "blob_row_gap": float("nan"),
           "blob_phi_max": float("nan")}
    if n == 0:
        return out
    wall_labels = set(np.unique(lab[wall_row][lab[wall_row] > 0]).tolist())
    if not wall_labels:
        return out
    sizes = {k: int((lab == k).sum()) for k in wall_labels}
    main = max(sizes, key=sizes.get)
    keep = np.zeros_like(mask)
    for k in range(1, n + 1):
        if k != main:
            keep |= lab == k
    if not keep.any():
        return out
    ys, xs = np.nonzero(keep)
    solid_rows = np.nonzero(solid.any(axis=1))[0]
    gap = float("nan")
    if solid_rows.size:
        gap = float(np.min(np.abs(ys[:, None] - solid_rows[None, :]).min(axis=1)))
    out.update(n_blobs=len(set(np.unique(lab[keep]).tolist()) - {main}),
               blob_area=float(keep.sum()),
               blob_centroid_x=float(xs.mean()), blob_centroid_y=float(ys.mean()),
               blob_row_gap=gap, blob_phi_max=float(phi[keep].max()))
    return out


def run_case(name: str, *, theta: float, steps: int, ny: int, nx: int, radius: float,
             xi: float, A: float, tau: float, beta: float, sample_every: int,
             outdir: Path, snapshots: list[int] | None = None) -> dict:
    spec = CASES[name]
    ny_eff = int(round(ny * spec.get("ny_scale", 1.0)))
    nx_eff = int(round(nx * spec.get("nx_scale", 1.0)))
    beta_eff = spec.get("beta") or beta
    solid = torch.zeros(ny_eff, nx_eff, dtype=torch.bool)
    solid[0, :] = True
    if spec["top"] == "wall":
        solid[ny_eff - 1, :] = True
    solid3 = solid.unsqueeze(0)
    solid_np = solid.numpy()
    f_r, f_b = sessile_drop_initial(ny_eff, nx_eff, radius, theta, xi=xi,
                                   dtype=torch.float64)
    x_c = 0.5 * nx_eff

    boxed = bool(spec.get("boxed"))
    hist: list[dict] = []
    snaps: dict[str, np.ndarray] = {}
    t0 = time.time()
    leak0 = seam_leak_fraction(f_r + f_b, solid) if not boxed else 0.0

    def _sample(it: int, f_r, f_b, prev_phi):
        rr, rb = f_r.sum(dim=0), f_b.sum(dim=0)
        rho = rr + rb
        phi = ((rr - rb) / rho.clamp(min=1e-12)).detach().cpu().numpy()
        jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(dim=0).detach().cpu().numpy()
        jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(dim=0).detach().cpu().numpy()
        u = np.sqrt((jx / rho.numpy()) ** 2 + (jy / rho.numpy()) ** 2)
        g = measure_wall_drop(phi, solid_np)
        g.update(_blob_stats(phi, solid_np))
        row = phi[1].copy()
        xl, xr = (g["x_centre"] - g["W"] / 2, g["x_centre"] + g["W"] / 2) \
            if math.isfinite(g["W"]) else (x_c, x_c)
        outside = np.ones_like(row, dtype=bool)
        outside[max(0, int(xl) - 1):int(xr) + 2] = False
        g.update(step=it, u_max=float(u[~solid_np].max()),
                 dphi_max=float(np.abs(phi - prev_phi).max()) if prev_phi is not None else float("nan"),
                 volume=float(phi.sum()), film_mass=float(np.clip(row[outside], 0, None).sum()),
                 top_band_mass=float(np.clip(phi[ny_eff - 10:ny_eff], 0, None).sum()),
                 ny=ny_eff, nx=nx_eff, beta=beta_eff)
        hist.append(g)
        if snapshots and it in snapshots and it > 0:
            snaps[f"phi_{name}_{it:06d}.npy"] = phi
        print(f"[{name}] t={it} cap={g['theta_cap_deg']:.2f} circ={g['theta_circle_deg']:.2f} "
              f"drop={g['drop_area']:.0f} blob={g['blob_area']:.0f}"
              f"@({g['blob_centroid_x']:.1f},{g['blob_centroid_y']:.1f}) gap={g['blob_row_gap']:.1f} "
              f"film={g['film_mass']:.2f} top={g['top_band_mass']:.2f} |u|={g['u_max']:.2e}",
              flush=True)
        return phi

    # t = 0: the *exact* spherical cap => this is the estimator's discretisation
    # floor (interface width xi and the level-set crossing rule bias W and H), not
    # a solver error.  Every later error must be quoted against this floor.
    prev_phi = _sample(0, f_r, f_b, None)
    hist[0]["estimator_floor"] = True

    for it in range(1, steps + 1):
        kw = dict(tau_r=tau, tau_b=tau, A=A, beta=beta_eff, solid_mask=solid,
                  visc_mode="mixture")
        if spec["bc"]:
            kw.update(wall_contact_angle=float(theta), contact_angle_convention="physical")
            if spec["top_theta"] is not None:
                kw["wall_contact_angle_by_direction"] = {(0, 1): float(spec["top_theta"])}
        f_r, f_b = color_gradient_two_tau_step(f_r, f_b, **kw)
        if boxed:
            f_r, f_b = stream_boxed(f_r, solid), stream_boxed(f_b, solid)
        else:
            f_r, f_b = stream(f_r), stream(f_b)
            f_r = torch.where(solid3, f_r[list(OPP)], f_r)
            f_b = torch.where(solid3, f_b[list(OPP)], f_b)
        if it % sample_every == 0:
            prev_phi = _sample(it, f_r, f_b, prev_phi)

    last = hist[-1]
    tail = [h for h in hist if h["step"] > 0.7 * steps] or [last]
    cap = [h["theta_cap_deg"] for h in tail if math.isfinite(h["theta_cap_deg"])]
    circ = [h["theta_circle_deg"] for h in tail if math.isfinite(h["theta_circle_deg"])]
    return {
        "case": name, "theta_cmd": theta, "beta": beta_eff, "boxed_stream": boxed,
        "seam_leak_fraction_initial": leak0,
        "params": {"steps": steps, "ny": ny_eff, "nx": nx_eff, "radius": radius, "xi": xi,
                   "A": A, "tau": tau, **{k: v for k, v in spec.items() if k != "ny_scale" and k != "nx_scale"}},
        "theta_cap_tail_deg": float(np.mean(cap)) if cap else float("nan"),
        "theta_circle_tail_deg": float(np.mean(circ)) if circ else float("nan"),
        "err_pct_cap": 100.0 * (float(np.mean(cap)) - theta) / theta if cap else float("nan"),
        "err_pct_circle": 100.0 * (float(np.mean(circ)) - theta) / theta if circ else float("nan"),
        "blob_area_final": last["blob_area"], "blob_centroid_x_final": last["blob_centroid_x"],
        "blob_centroid_y_final": last["blob_centroid_y"], "drop_area_final": last["drop_area"],
        "theta_cap_floor_deg": hist[0]["theta_cap_deg"],
        "theta_circle_floor_deg": hist[0]["theta_circle_deg"],
        "err_pct_cap_vs_floor_deg": (100.0 * (float(np.mean(cap)) - hist[0]["theta_cap_deg"]) / hist[0]["theta_cap_deg"]) if cap else float("nan"),
        "volume_initial": hist[0]["volume"], "volume_final": last["volume"],
        "wall_time_s": time.time() - t0, "history": hist,
    }, snaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="base,tall,topwall90,nowall,beta03")
    ap.add_argument("--theta", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--xi", type=float, default=2.0)
    ap.add_argument("--A", type=float, default=0.04)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--snapshot-steps", default="")
    ap.add_argument("--out", default="runs/xval_blob_mechanism")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    outdir = Path(args.out)
    (outdir / "snapshots").mkdir(parents=True, exist_ok=True)
    snaps_wanted = [int(s) for s in args.snapshot_steps.split(",") if s.strip()]

    results = []
    for name in [c.strip() for c in args.cases.split(",") if c.strip()]:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; known: {sorted(CASES)}")
        res, snaps = run_case(name, theta=args.theta, steps=args.steps, ny=args.ny, nx=args.nx,
                             radius=args.radius, xi=args.xi, A=args.A, tau=args.tau,
                             beta=args.beta, sample_every=args.sample_every, outdir=outdir,
                             snapshots=snaps_wanted)
        for fname, arr in snaps.items():
            np.save(outdir / "snapshots" / fname, arr)
        results.append(res)
        print(f"[{name}] DONE cap={res['theta_cap_tail_deg']:.2f}"
              f"({res['err_pct_cap']:+.2f}%) circ={res['theta_circle_tail_deg']:.2f}"
              f"({res['err_pct_circle']:+.2f}%) blob={res['blob_area_final']:.0f} "
              f"({res['wall_time_s']:.0f}s)", flush=True)

    path = outdir / f"blob_mechanism_theta{args.theta:g}.json"
    path.write_text(json.dumps({"params": vars(args), "results": results}, indent=2))
    print(f"[blob_mechanism] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())