"""Clean-room (v2) verification of the linear-interpolated bounce-back alpha family.

Anchor: ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``.

Why a second verifier
---------------------
``run_xval.py`` (previous mission on the same anchor) reported an ``honest FAIL``
for the OSS exactness claim and diagnosed a systematic ``+0.5`` cell displacement
of the effective wall in the production operator.  That is a claim about
``src/tensorlbm/boundary/interpolated_bounce_back.py``, so it must survive an
independent reconstruction.  This driver shares no code path with ``run_xval.py``
except the operator under test; collision, streaming, moments and diagnostics are
written from scratch in NumPy.

Published recipes used (read, not copied)
-----------------------------------------
* geometry / exact solution: Bouzidi, Firdaouss & Lallemand, *Phys. Fluids* 13
  (2001) 3452 sec. III.A -- force-driven plane Poiseuille, walls a sub-grid
  distance ``delta`` outside the first/last fluid row, ``H = (N-1) + 2 delta``,
  ``u(y) = F/(2 nu) y (H - y)``;
* discretisation: d2Q9 TRT, force added *after* the collision with isotropic
  weights ``3 w_i c_ix F``, ``Lambda+ Lambda- = delta^2/2`` for the alpha family
  (pylbm ``LinearBounceBack``; ``tests/test_linear_bounce_back.py``);
* momentum convention: the stored populations hold the force, i.e. the physical
  velocity is ``m - F/2`` with ``m = sum_i f_i c_i`` while the equilibrium is
  built from the stored momentum ``m`` (pylbm: "the momentum is the velocity
  plus F/2").

Loop order (the thing the previous mission got wrong)
-----------------------------------------------------
The production operator writes the reflected population into the **solid** cell
``x_s = x_f + c_i`` (slot ``opp(i)``) so that the *next* streaming step delivers
it to ``x_f``.  It is therefore only valid if nothing advances the solid cells
between the operator call and that streaming step -- in particular the collision
must be restricted to fluid cells.  ``backend='prod_collide'`` reproduces the
previous mission's loop (collision on every cell) and is kept as the
**falsification arm**: it must reproduce the old failure numbers, which pins the
defect on the driver and not on the operator.

Backends
--------
``fluid``        reference driver: the same closure applied on the fluid side
                 (the value is written into ``x_f`` slot ``opp(i)`` directly).
``prod``         production operator, solid-side, collision restricted to fluid
                 cells -- the contract-correct integration.
``prod_collide`` production operator, solid-side, collision everywhere (old loop).
``halfway``      production ``bounce_back_cells`` (half-way bounce-back) fluid-side.

Usage
-----
    PYTHONPATH=src python3 xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/verify_v2.py \
        --out xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/artifacts_v2 [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import C as _C_T, OPPOSITE as _OPP_T, W as _W_T  # noqa: E402

CI = np.array(_C_T.tolist(), dtype=np.int64)          # (Q, 2) as (cx, cy)
WI = np.array(_W_T.tolist(), dtype=np.float64)
OPP = np.array(_OPP_T.tolist(), dtype=np.int64)
Q = CI.shape[0]

ANCHOR = "XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME"


# --------------------------------------------------------------------------- #
# alpha-family members (q = fraction of the link from the fluid node to the wall)
# --------------------------------------------------------------------------- #
def alpha_bfl(delta):
    return np.where(delta <= 0.5, 2.0, 1.0 / np.maximum(delta, 0.5))


def alpha_cli(delta):
    return 4.0 / (1.0 + 2.0 * delta)


def alpha_half(delta):
    return np.full_like(delta, 2.0)      # q = 1/2 member == half-way bounce-back


ALPHAS = {"bfl": alpha_bfl, "cli": alpha_cli, "halfway": alpha_half}


# --------------------------------------------------------------------------- #
# closures
# --------------------------------------------------------------------------- #
def coeffs(q, alpha_fun):
    a = alpha_fun(q)
    return a, -1.0 + a * (0.5 + q), 1.0 - a * q, 1.0 - 0.5 * a


def wall_value(i, a, lam_m, g):
    """Dirichlet term ``-alpha e^-_q(x_w)`` with wall momentum ``u_w = (lam_m g, 0)``.

    ``e^-_q(x_w) = (f^eq_q - f^eq_opp(q))/2 = 3 w_q c_qx (u_w)_x`` for unit density
    (the quadratic terms cancel between opposite directions).  It vanishes for the
    wall-normal links of a wall-aligned channel but **not** for the diagonal cut
    links, contrary to the note in the previous mission's OSS_DIFF.md.
    """
    return a * 3.0 * WI[i] * float(CI[i, 0]) * lam_m * g


def reflect_fluid(f_new, f_pre, q_link, link, alpha_fun, lam_m=None, g=0.0):
    """Reference (fluid-side) application of the alpha closure.

    ``f_d(x_f, t+1) = ah f_i(x_f,t) + bh f_d(x_f,t) + beta f_i(x_f,t+1) - alpha e^-_q(x_w)``
    """
    out = f_new.copy()
    for i in range(1, Q):
        d = int(OPP[i])
        m = link[i]
        if not m.any():
            continue
        q = q_link[i][m]
        a, ah, beta, bh = coeffs(q, alpha_fun)
        val = ah * f_pre[i][m] + bh * f_pre[d][m] + beta * f_new[i][m]
        if lam_m is not None:
            val = val - wall_value(i, a, lam_m, g)
        slab = out[d]
        slab[m] = val
        out[d] = slab
    return out


def reflect_prod(f_new, f_pre, sdf, solid, alpha_fun=None, lam_m=None, g=0.0):
    """Production operator (torch), solid-side; returns NumPy."""
    import torch

    from tensorlbm.boundary.interpolated_bounce_back import interpolated_bounce_back_2d

    kw = {}
    if alpha_fun is not None and alpha_fun is not alpha_half:
        kw["alpha"] = lambda q: torch.from_numpy(alpha_fun(q.numpy()))
    out = interpolated_bounce_back_2d(
        torch.from_numpy(np.ascontiguousarray(f_new)),
        torch.from_numpy(np.ascontiguousarray(f_pre)),
        torch.from_numpy(np.ascontiguousarray(sdf)),
        torch.from_numpy(np.ascontiguousarray(solid)),
        **kw,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.numpy()


# --------------------------------------------------------------------------- #
# channel scaffold
# --------------------------------------------------------------------------- #
class Channel:
    """Walls in y, periodic in x, walls ``delta`` cells outside the fluid rows."""

    def __init__(self, n_fluid: int, delta: float, nx: int = 1, n_wall: int = 2):
        self.n_fluid, self.delta, self.nx, self.n_wall = n_fluid, delta, nx, n_wall
        self.ny = n_fluid + 2 * n_wall
        self.height = (n_fluid - 1) + 2.0 * delta
        j = np.arange(self.ny)
        self.y = delta + (j - n_wall)
        self.fluid1d = (j >= n_wall) & (j < n_wall + n_fluid)
        dist = np.minimum(self.y, self.height - self.y)
        sdf1d = np.where(self.fluid1d, dist, -np.abs(dist))
        self.sdf = np.repeat(sdf1d[:, None], nx, axis=1)
        self.solid = np.repeat((~self.fluid1d)[:, None], nx, axis=1)
        self.fluid = ~self.solid
        link = np.zeros((Q, self.ny, nx), dtype=bool)
        qq = np.zeros((Q, self.ny, nx), dtype=np.float64)
        for i in range(1, Q):
            cx, cy = int(CI[i, 0]), int(CI[i, 1])
            nb_solid = np.roll(self.solid, (-cy, -cx), axis=(0, 1))
            m = self.fluid & nb_solid
            s_s = np.roll(self.sdf, (-cy, -cx), axis=(0, 1))
            den = self.sdf - s_s
            q = np.where(np.abs(den) > 1e-300, self.sdf / np.where(den == 0, 1.0, den), 0.5)
            link[i] = m
            qq[i] = np.where(m, q, 0.5)
        self.link, self.q_link = link, qq
        self.ref_profile_y = self.y[n_wall : n_wall + n_fluid]


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
def equilibrium_np(rho, ux, uy):
    cu = CI[:, 0][:, None, None] * ux + CI[:, 1][:, None, None] * uy
    usq = ux * ux + uy * uy
    return WI[:, None, None] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def moments(f):
    return (
        f.sum(axis=0),
        np.tensordot(CI[:, 0].astype(np.float64), f, axes=(0, 0)),
        np.tensordot(CI[:, 1].astype(np.float64), f, axes=(0, 0)),
    )


def collide_trt(f, rho, ux, uy, lam_p, lam_m):
    feq = equilibrium_np(rho, ux, uy)
    sp, sm = 1.0 / (0.5 + lam_p), 1.0 / (0.5 + lam_m)
    fp, fm = 0.5 * (f + f[OPP]), 0.5 * (f - f[OPP])
    ep, em = 0.5 * (feq + feq[OPP]), 0.5 * (feq - feq[OPP])
    return f - sp * (fp - ep) - sm * (fm - em)


def collide_bgk(f, rho, ux, uy, tau):
    return f - (1.0 / tau) * (f - equilibrium_np(rho, ux, uy))


def stream_np(f):
    out = np.empty_like(f)
    for i in range(Q):
        out[i] = np.roll(f[i], (int(CI[i, 1]), int(CI[i, 0])), axis=(0, 1))
    return out


def run_case(
    n_fluid: int,
    delta: float,
    closure: str,
    carrier: str,
    *,
    nu: float = 0.1,
    u_max: float = 0.02,
    backend: str = "fluid",
    wall_term: bool = True,
    u_shift: float = 0.0,
    report_shift: float = -0.5,
    steps: int = 8000,
) -> dict:
    """One (delta, closure, carrier, backend) arm of the sweep."""
    ch = Channel(n_fluid, delta)
    H = ch.height
    g = 8.0 * nu * u_max / H**2
    if carrier == "bgk":
        lam_p = lam_m = None
        tau = 0.5 + 3.0 * nu
    else:
        lam_p = 3.0 * nu
        prod = {"trt_d2": 0.5 * delta**2, "trt_c": 3.0 / 16.0, "trt_d8": 0.5 * delta**2}[carrier]
        lam_m = prod / lam_p
    src = 3.0 * WI[:, None, None] * CI[:, 0].astype(np.float64)[:, None, None] * g
    src = np.where(ch.fluid[None], src, 0.0)
    af = ALPHAS[closure]
    wm = lam_m if (wall_term and lam_m is not None) else None

    f = equilibrium_np(np.ones((ch.ny, ch.nx)), np.zeros((ch.ny, ch.nx)), np.zeros((ch.ny, ch.nx)))
    t0 = time.time()
    cols: list[np.ndarray] = []
    for n in range(1, steps + 1):
        rho, mx, my = moments(f)
        # stored momentum (pylbm convention).  ``u_shift`` is the *wrong*
        # convention kept as a falsification arm: u = (m + F/2)/rho.
        ux = (mx + u_shift * g * ch.fluid) / rho
        uy = my / rho
        if carrier == "bgk":
            f_post = collide_bgk(f, rho, ux, uy, tau)
        else:
            f_post = collide_trt(f, rho, ux, uy, lam_p, lam_m)
        if backend == "prod_collide":         # old loop: source + collision everywhere
            f_post = f_post + src
        else:
            f_post = np.where(ch.fluid[None], f_post + src, f)
        f_st = stream_np(f_post)
        f_st[:, 0, :] = WI[:, None]           # anti-wrap: outermost rows at rest
        f_st[:, -1, :] = WI[:, None]
        if backend in ("prod", "prod_collide"):
            f = reflect_prod(f_st, f_post, ch.sdf, ch.solid,
                             None if closure == "halfway" else af, lam_m=wm, g=g)
            if backend == "prod" and wm is not None:
                f = _add_prod_wall_term(f, f_st, ch, af, wm, g)
        else:
            f = reflect_fluid(f_st, f_post, ch.q_link, ch.link, af, lam_m=wm, g=g)
        if n % 500 == 0:
            cols.append(u_profile(f, g, ch, report_shift))
            if len(cols) > 3 and np.max(np.abs(cols[-1] - cols[-2])) < 1e-12 * u_max:
                break
    u = cols[-1]
    drift = float(np.max(np.abs(cols[-1] - cols[-2]))) if len(cols) > 1 else float("nan")

    yj = ch.ref_profile_y
    u_ref = g / (2.0 * nu) * yj * (H - yj)
    coef = np.polyfit(yj, u, 2)
    roots = np.sort(np.roots(coef))
    fit_dev = float(np.abs(np.polyval(coef, yj) - u).max() / np.abs(u).max())
    return {
        "n_fluid": n_fluid, "delta": delta, "closure": closure, "carrier": carrier,
        "backend": backend, "wall_term": wall_term, "report_shift": report_shift,
        "H": H, "g": g, "nu": nu, "u_max": float(u_ref.max()),
        "lambda_product": None if lam_m is None else lam_m * lam_p,
        "lambda_minus": lam_m, "steps": n, "drift": drift,
        "l2_rel": float(np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)),
        "max_rel": float(np.abs(u - u_ref).max() / np.abs(u_ref).max()),
        "mean_offset": float(np.mean(u - u_ref)),
        "mean_offset_over_g": float(np.mean(u - u_ref) / g),
        "w_bot": float(yj[0] - roots[0]),
        "w_top": float(roots[1] - yj[-1]),
        "fit_dev": fit_dev,
        "wall_time_s": time.time() - t0,
        "y": [float(v) for v in yj], "u": [float(v) for v in u],
        "u_ref": [float(v) for v in u_ref],
    }


def u_profile(f, g, ch, report_shift):
    rho, mx, _ = moments(f)
    return (mx / rho)[ch.fluid1d, 0] + report_shift * g


def _add_prod_wall_term(f, f_st, ch, af, lam_m, g):
    """Add the Dirichlet ``-alpha e^-_q(x_w)`` term to the production operator output.

    The operator implements the zero-velocity closure only; the term is added here
    on the delivered slots (fluid side), which is where the operator writes for
    ``x_s``.  Kept separate so the operator stays unmodified for this comparison.
    """
    out = f.copy()
    for i in range(1, Q):
        d = int(OPP[i])
        cx, cy = int(CI[i, 0]), int(CI[i, 1])
        m = ch.link[i]
        if not m.any():
            continue
        a = af(ch.q_link[i][m])
        field = np.zeros(out.shape[1:], dtype=np.float64)
        field[m] = wall_value(i, a, lam_m, g)
        # value belongs to x_f and is delivered to x_s = x_f + c_i
        field = np.roll(field, (cy, cx), axis=(0, 1))
        tgt = np.roll(m, (-cy, -cx), axis=(0, 1))     # solid cells x_s
        slab = out[d]
        slab[tgt] = slab[tgt] - field[tgt]
        out[d] = slab
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("artifacts_v2")))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", default="sweep,rootcause")
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    rows = []

    def emit(r, tag):
        rows.append(dict(r, stage=tag))
        (out / "partial.json").write_text(json.dumps(rows, indent=1))
        print(f"[{tag}] N={r['n_fluid']:3d} d={r['delta']:.3f} {r['carrier']:6s} "
              f"{r['closure']:7s} {r['backend']:12s} wt={int(r['wall_term'])} "
              f"L2={100*r['l2_rel']:.3e}% off/g={r['mean_offset_over_g']:+.3f} "
              f"w=({r['w_bot']:.3f},{r['w_top']:.3f}) drift={r['drift']:.1e} "
              f"t={r['wall_time_s']:.1f}s", flush=True)

    if "sweep" in stages:
        deltas = [0.25, 0.5] if args.quick else [0.1, 0.25, 0.35, 0.5, 0.75, 0.9]
        ns = [8] if args.quick else [8, 16]
        for n in ns:
            for d in deltas:
                for carrier in ("bgk", "trt_c", "trt_d2"):
                    for closure in ("halfway", "bfl", "cli"):
                        emit(run_case(n, d, closure, carrier, steps=args.steps), "sweep")

    if "rootcause" in stages:
        # (R1) operator output destroyed by a collision on solid cells
        for backend in ("fluid", "prod", "prod_collide"):
            for d in (0.25, 0.5, 0.9):
                emit(run_case(8, d, "bfl", "trt_d2", backend=backend, steps=args.steps),
                     "rootcause")
        # (R2) momentum convention: collision fed with m + F/2 instead of m
        for ush in (0.0, 0.5):
            emit(run_case(8, 0.5, "halfway", "trt_c", backend="fluid",
                          u_shift=ush, wall_term=False, steps=args.steps), "rootcause")
        # (R3) Dirichlet wall term on/off (matters on the diagonal cut links)
        for wt in (False, True):
            for d in (0.25, 0.5, 0.9):
                emit(run_case(8, d, "bfl", "trt_d2", wall_term=wt, steps=args.steps),
                     "rootcause")

    (out / "results.json").write_text(json.dumps({"anchor": ANCHOR, "cases": rows}, indent=1))
    print(f"wrote {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())