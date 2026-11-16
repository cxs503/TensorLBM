"""Two-level verification study of the D2Q5 tube-Graetz advection-diffusion solver.

Two geometric conventions of the *same* lattice solver are compared against the
independent Graetz series of :mod:`reference_series`:

``orig``  the library convention of ``tensorlbm.graetz.graetz_lbm`` (as of the
          merge): the Poiseuille profile is built with ``R = n_r - 1`` while the
          scalar Dirichlet wall (anti-bounce-back) physically sits half a cell
          further out, at ``R + 1/2``.
``reff``  the consistent convention: ``R_eff = n_r - 1/2`` is used in the
          velocity profile, in the diameter that defines ``x*``, and in the
          cup-mixing / Nusselt normalisation.

Both are run at ``n_r`` in {40, 80} with the same physical parameterisation
(``u_mean``, ``alpha``, station list in ``x*``), so the comparison is a pure
convergence test.  Numbers are written to JSON; nothing is transcribed by hand.

Run:  python benchmarks/verified/graetz_tube/study.py --variant orig --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

import reference_series as RS  # noqa: E402
from tensorlbm.graetz import graetz_lbm  # noqa: E402

U_MEAN = 0.05
ALPHA = 0.03
X_STAR = np.array([0.002, 0.005, 0.01, 0.02, 0.04])
LEVELS = (40, 56)


def _lbm_reff(n_r: int, u_mean: float, alpha: float, x_star: np.ndarray) -> dict:
    """Lattice D2Q5 solver with the half-cell-consistent radius ``R_eff = n_r-1/2``."""
    from tensorlbm.thermal import W5

    R = (n_r - 1) + 0.5          # wall sits half-way between the last two node rows
    D = 2.0 * R
    st = np.asarray(x_star) * u_mean * D * D / alpha       # lattice stations
    n_x = int(st.max()) + 40
    n_steps = int(2.0 * st.max() / u_mean)
    w = np.asarray(W5, dtype=float)
    wb = w.reshape(5, 1, 1)
    c = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    ny = n_r
    yy = np.arange(ny, dtype=float)
    ux = np.broadcast_to((2.0 * u_mean * (1.0 - (yy / R) ** 2)).reshape(ny, 1), (ny, n_x)).copy()
    uy = np.zeros_like(ux)
    tau = 0.5 + 3.0 * alpha
    inv_tau = 1.0 / tau
    T_in, T_wall = 1.0, 0.0
    T = np.full((ny, n_x), T_in)
    cu = c[:, 0].reshape(5, 1, 1) * ux[None] + c[:, 1].reshape(5, 1, 1) * uy[None]
    g = wb * T[None] * (1.0 + 3.0 * cu)
    for _ in range(n_steps):
        T = g.sum(axis=0)
        grad = np.zeros_like(T)
        grad[1:-1] = 0.5 * (T[2:] - T[:-2])
        grad[-1] = T_wall - T[-2]
        src = np.empty_like(T)
        src[1:] = alpha * grad[1:] / yy[1:].reshape(-1, 1)
        src[0] = alpha * 2.0 * (T[1] - T[0])
        cu = c[:, 0].reshape(5, 1, 1) * ux[None] + c[:, 1].reshape(5, 1, 1) * uy[None]
        geq = wb * T[None] * (1.0 + 3.0 * cu)
        gp = g - (g - geq) * inv_tau + wb * src[None]
        g = np.empty_like(gp)
        g[0] = gp[0]
        g[1, :, 1:] = gp[1, :, :-1]
        g[2, :, :-1] = gp[2, :, 1:]
        g[3, 1:, :] = gp[3, :-1, :]
        g[4, :-1, :] = gp[4, 1:, :]
        g[1, :, 0] = -gp[2, :, 0] + 2.0 * w[1] * T_in
        g[2, :, -1] = g[2, :, -2]
        g[4, -1, :] = -gp[3, -1, :] + 2.0 * w[4] * T_wall
        g[3, 0, :] = gp[4, 0, :]
    T = g.sum(axis=0)
    rr = np.arange(ny, dtype=float)
    up = 2.0 * u_mean * (1.0 - (rr / R) ** 2)
    wgt = up * rr
    nu, tm = [], []
    for xst in st:
        col = int(round(float(xst)))
        col = min(max(col, 2), n_x - 1)
        dTdr = (T_wall - T[-1, col]) / 0.5          # wall at half-cell beyond node n_r-1
        Tm = float(np.trapezoid(wgt * T[:, col], rr) / np.trapezoid(wgt, rr))
        tm.append(Tm)
        nu.append(float(-2.0 * R * dTdr / Tm))
    return {
        "Nu": np.array(nu),
        "theta_m": np.array(tm),
        "R": R,
        "D": D,
        "n_x": n_x,
        "n_steps": n_steps,
        "tau": tau,
        "R_used_in_velocity": R,
        "wall_offset_cells": 0.5,
        "field": T,
        "r_over_R": rr / R,
    }


def run(variant: str, n_r: int) -> dict:
    t0 = time.time()
    if variant == "orig":
        D = 2.0 * (n_r - 1.0)
        st = X_STAR * U_MEAN * D * D / ALPHA
        out = graetz_lbm(n_r=n_r, n_x=int(st.max()) + 40, u_mean=U_MEAN,
                         alpha=ALPHA, stations_x=st)
        res = {"Nu": out["Nu"], "theta_m": out["theta_m"], "R": n_r - 1.0, "D": D,
               "n_x": out["meta"]["n_x"], "n_steps": out["meta"]["n_steps"],
               "tau": out["meta"]["tau"], "R_used_in_velocity": (n_r - 1.0) / (n_r - 1.0),
               "wall_offset_cells": 0.5, "field": out["field"],
               "r_over_R": out["r"] / (n_r - 1.0)}
    elif variant == "reff":
        o = _lbm_reff(n_r, U_MEAN, ALPHA, X_STAR)
        res = o
        res["R_used_in_velocity"] = 1.0
    else:
        raise SystemExit(f"unknown variant {variant}")
    ref = RS.series(X_STAR, n_modes=10, n_r=800)
    res.update({
        "variant": variant,
        "n_r": n_r,
        "u_mean": U_MEAN,
        "alpha": ALPHA,
        "x_star": X_STAR.tolist(),
        "Nu_ref": ref["Nu"].tolist(),
        "rel_err": (res["Nu"] / ref["Nu"] - 1.0).tolist(),
        "runtime_s": time.time() - t0,
        "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    res.pop("field", None)
    res.pop("r_over_R", None)
    # JSON sanitisation: every numeric array becomes a list (programmatic injection)
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="orig", choices=("orig", "reff"))
    ap.add_argument("--shard", type=int, default=0, help="1 -> orig, 2 -> reff")
    ap.add_argument("--n_r", type=int, nargs="*", default=list(LEVELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    variant = a.variant
    if a.shard == 1:
        variant = "orig"
    elif a.shard == 2:
        variant = "reff"
    out = {"variant": variant, "levels": [run(variant, n) for n in a.n_r],
           "reference": RS.series(X_STAR, n_modes=10, n_r=800)["Nu"].tolist(),
           "x_star": X_STAR.tolist(), "nu_inf": RS.NU_INF}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for lv in out["levels"]:
        print(variant, lv["n_r"], "err%", np.round(100 * np.array(lv["rel_err"]), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
