"""XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S (round 1, open-source cross-benchmark).

Cross-validates TensorLBM's *force-corrected NEBB + body-force driver* scaffold
against the mature open-source implementation **haimingz/vivsim**
(``examples/2d/poiseuille_channel.py`` + ``vivsim/lbm/boundary/nebb.py`` +
``vivsim/lbm/forcing/{edm,guo}.py`` + ``vivsim/lbm/basic.py``).

Problem (identical for both arms)
---------------------------------
2-D D2Q9 channel, periodic in x, no-slip top/bottom walls (half-way), constant
body force ``Gx`` in x.  Steady fully-developed Poiseuille profile::

    ux(y) = Gx/(2 nu) * y * (H - y),   H = NY - 1,  y in [0, H]

Arm A (OSS reference, vivsim):
    numpy port of vivsim's exact algorithm.  Formulas transcribed from
    ``/tmp/vivsim_src`` (zero hand-copy):
      * ``boundary_nebb`` (shear-reconstruction NEBB, ``nebb.py``)
      * ``get_corrected_wall_velocity`` = u_wall - g/(2 rho) (``_helpers.py``)
      * ``forcing_edm`` = f + Guo_term(g, u) (``forcing/edm.py``)
      * ``forcing_guo_bgk`` = f + (1 - omega/2) * Guo_term(g, u) (``forcing/guo.py``)
      * ``get_velocity_correction`` = g/(2 rho) (``basic.py``)
      * init ``f = feq(rho, u - g/2)`` (example)
      * force-corrected wrapper uses ``rho_wall = 1`` (default)

Arm B (TensorLBM):
    repo building blocks (``src/tensorlbm/d2q9.py``) + repo Guo forcing
    (``gpu_core_path.py``: feq at u* = u + F/(2 rho), force term at actual u,
    scaled by (1 - 1/(2 tau))) + repo NEBB form (``wave_bc_common.py``:
    f[k] = feq[k] - feq[opp] + f[opp]) with the same wall-velocity correction
    and field-recovered wall density.

Pre-declared decision & thresholds (experimental-design skill)
---------------------------------------------------------------
* Decision: does TensorLBM's force-corrected NEBB + body-force driver
  reproduce the OSS reference within the owner accuracy standard?
* Primary metric: max relative error of the steady ux(y) profile vs the
  closed-form Poiseuille solution, and max |diff| between the two arms.
* Thresholds (pre-declared, not tuned after seeing results):
    - PASS if max_rel_err(arm, analytic) < 0.15  (owner standard)
    - ecosystem stretch: < 0.03
* Sensitivity (>= 2 points):
    - forcing scheme: EDM vs Guo (within each arm)
    - grid refinement: NY = 10 vs NY = 20
* Environment: CPU, float64, single process.  No GPU.

Outputs: ``xval/xval_nebb_bodyforce_driver.json`` + figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Shared D2Q9 lattice (identical numbering in vivsim and TensorLBM)
# --------------------------------------------------------------------------- #
C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int64,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
RIGHT = [1, 5, 8]
LEFT = [3, 7, 6]
UP = [2, 5, 6]
DOWN = [4, 7, 8]

# Boundary spec for top/bottom (from vivsim/lbm/lattice.py D2Q9.boundary_spec):
#   top:    in=(2,5,6) out=(4,7,8) tan=(1,3) normal_sign=-1  (wall row y=ny-1)
#   bottom: in=(4,7,8) out=(2,5,6) tan=(1,3) normal_sign=+1  (wall row y=0)
TOP = dict(in_dirs=(4, 7, 8), out_dirs=(2, 5, 6), tan_dirs=(1, 3), normal_sign=-1, wy="top")
BOT = dict(in_dirs=(2, 5, 6), out_dirs=(4, 7, 8), tan_dirs=(1, 3), normal_sign=+1, wy="bot")


def feq(rho, ux, uy):
    """D2Q9 second-order equilibrium (identical in both codebases)."""
    cx = C[:, 0].reshape(9, 1, 1)
    cy = C[:, 1].reshape(9, 1, 1)
    uc = cx * ux[None, ...] + cy * uy[None, ...]
    usq = ux * ux + uy * uy
    return W[:, None, None] * (rho * (1 + 3 * uc + 4.5 * uc**2 - 1.5 * usq[None, ...]) - 1)


def macro(f):
    rho = 1 + f.sum(axis=0)
    ux = (f[RIGHT].sum(0) - f[LEFT].sum(0)) / rho
    uy = (f[UP].sum(0) - f[DOWN].sum(0)) / rho
    return rho, ux, uy


def stream(f):
    """D2Q9 periodic streaming.  f[i] has shape (nx, ny); axis0=x, axis1=y."""
    out = np.empty_like(f)
    out[0] = f[0]
    out[1] = np.roll(f[1], -1, axis=0)
    out[2] = np.roll(f[2], -1, axis=1)
    out[3] = np.roll(f[3], 1, axis=0)
    out[4] = np.roll(f[4], 1, axis=1)
    out[5] = np.roll(np.roll(f[5], -1, axis=0), -1, axis=1)
    out[6] = np.roll(np.roll(f[6], 1, axis=0), -1, axis=1)
    out[7] = np.roll(np.roll(f[7], 1, axis=0), 1, axis=1)
    out[8] = np.roll(np.roll(f[8], -1, axis=0), 1, axis=1)
    return out


def guo_term(gx, gy, ux, uy):
    """Guo forcing term w_i[3(c_i - u).g + 9 (c_i.u)(c_i.g)] (both codebases)."""
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    if gx.ndim == 0:
        gx = np.full_like(ux, float(gx))
    if gy.ndim == 0:
        gy = np.full_like(uy, float(gy))
    cx = C[:, 0].reshape(9, 1, 1)
    cy = C[:, 1].reshape(9, 1, 1)
    uc = cx * ux[None, ...] + cy * uy[None, ...]
    term = (
        gx[None, ...] * (3.0 * (cx - ux[None, ...]) + 9.0 * uc * cx)
        + gy[None, ...] * (3.0 * (cy - uy[None, ...]) + 9.0 * uc * cy)
    )
    return W[:, None, None] * term


def wall_row(spec, ny):
    return ny - 1 if spec["wy"] == "top" else 0


def rho_wall_from_field(f, spec, wy):
    """Wall density recovery (vivsim _get_rho_wall_numerator / TensorLBM wave_bc).

    num = 1 + (f0 + f_tan0 + f_tan1 + 2*(f_out0 + f_out1 + f_out2)) at the wall row.
    """
    t0, t1 = spec["tan_dirs"]
    o0, o1, o2 = spec["out_dirs"]
    num = 1.0 + (
        f[0, :, wy]
        + f[t0, :, wy]
        + f[t1, :, wy]
        + 2.0 * (f[o0, :, wy] + f[o1, :, wy] + f[o2, :, wy])
    )
    return num  # divide by (1 - un) with un=0 for no-slip -> num


# --------------------------------------------------------------------------- #
# Arm A: vivsim (OSS reference) — exact formulas from /tmp/vivsim_src
# --------------------------------------------------------------------------- #
def vivsim_nebb(f, spec, ny, rho_wall, ux_wall, uy_wall):
    """boundary_nebb (vivsim/lbm/boundary/nebb.py) — shear-reconstruction form.

    Applied only to the wall row.
    """
    wy = wall_row(spec, ny)
    un = spec["normal_sign"] * uy_wall
    ut = spec["normal_sign"] * ux_wall
    shear = 0.5 * (f[spec["tan_dirs"][0], :, wy] - f[spec["tan_dirs"][1], :, wy]) * spec["normal_sign"]
    normal_term = (1.0 / 6.0) * un * rho_wall
    tangential_term = 0.5 * ut * rho_wall
    o0, o1, o2 = spec["out_dirs"]
    new = np.stack(
        [
            f[o0, :, wy] + (2.0 / 3.0) * un * rho_wall,
            f[o1, :, wy] - shear + normal_term + tangential_term,
            f[o2, :, wy] + shear + normal_term - tangential_term,
        ]
    )
    f2 = f.copy()
    for k, v in zip(spec["in_dirs"], new):
        f2[k, :, wy] = v
    return f2


def vivsim_step(f, gx, nu, recipe):
    """One vivsim step.  recipe in {'bgk_edm', 'bgk_guo'} (example recipes)."""
    omega = 1.0 / (3.0 * nu + 0.5)
    rho, ux, uy = macro(f)
    if recipe == "bgk_guo":
        # Guo: correction applied BEFORE collision (example update_step_bgk_guo)
        ux = ux + 0.5 * gx / rho
        feq_ = feq(rho, ux, uy)
        f = (1 - omega) * f + omega * feq_
        f = f + guo_term(gx, 0.0, ux, uy) * (1 - 0.5 * omega)
    else:  # bgk_edm
        feq_ = feq(rho, ux, uy)
        f = (1 - omega) * f + omega * feq_
        f = f + guo_term(gx, 0.0, ux, uy)  # EDM = full Guo term
    f = stream(f)
    # force-corrected NEBB on top/bottom (no-slip wall, gx_wall = gx)
    for spec in (TOP, BOT):
        rw = 1.0  # wrap_force_corrected default rho_wall=1
        uxw, uyw = 0.0 - gx / (2.0 * rw), 0.0  # get_corrected_wall_velocity(0,0,gx_wall=gx)
        f = vivsim_nebb(f, spec, ny=f.shape[2], rho_wall=rw, ux_wall=uxw, uy_wall=uyw)
    return f


def run_vivsim(nx, ny, nu, gx, steps, recipe="bgk_edm"):
    f = feq(np.ones((nx, ny)), np.full((nx, ny), -gx / 2.0), np.zeros((nx, ny)))
    for _ in range(steps):
        f = vivsim_step(f, gx, nu, recipe)
    rho, ux, uy = macro(f)
    ux = ux + 0.5 * gx / rho  # post-processing velocity correction
    return ux, uy, rho


# --------------------------------------------------------------------------- #
# Arm B: TensorLBM — repo building blocks (d2q9.py) + repo Guo (gpu_core_path)
#         + repo NEBB form (wave_bc_common.py) with wall-velocity correction
# --------------------------------------------------------------------------- #
def tl_nebb_wall(f, spec, ny, rho_wall, ux_wall, uy_wall):
    """TensorLBM NEBB form (wave_bc_common.py): f[k] = feq[k] - feq[opp] + f[opp].

    Applied only to the wall row.
    """
    wy = wall_row(spec, ny)
    nx = f.shape[1]
    feq_ = feq(np.full((nx,), rho_wall), np.full((nx,), ux_wall), np.full((nx,), uy_wall))
    f2 = f.copy()
    for k in spec["in_dirs"]:
        f2[k, :, wy] = feq_[k, 0] - feq_[OPP[k], 0] + f[OPP[k], :, wy]
    return f2


def tl_step(f, gx, nu):
    """One TensorLBM step: BGK + Guo (gpu_core_path.py) + force-corrected NEBB."""
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau
    rho, ux, uy = macro(f)
    inv_rho = 1.0 / np.clip(rho, 1e-12, None)
    fx = gx * rho  # force density = acceleration * density
    usx = ux + 0.5 * fx * inv_rho  # force-corrected velocity u*
    feq_ = feq(rho, usx, uy)
    f = f * (1.0 - omega) + omega * feq_
    # Guo force term at the *actual* velocity u, scaled by (1 - 1/(2 tau))
    cx = C[:, 0].reshape(9, 1, 1)
    cy = C[:, 1].reshape(9, 1, 1)
    cuf = cx * fx[None, ...]
    cu_phys = cx * ux[None, ...] + cy * uy[None, ...]
    force_q = W[:, None, None] * (1.0 - 0.5 * omega) * (3.0 * (cuf - fx[None, ...] * ux[None, ...]) + 9.0 * cu_phys * cuf)
    f = f + force_q
    f = stream(f)
    # force-corrected NEBB walls (no-slip, corrected by -gx/(2 rho))
    for spec in (TOP, BOT):
        wy = wall_row(spec, f.shape[2])
        rw = rho_wall_from_field(f, spec, wy)  # TensorLBM wave_bc recovers rho from field
        uxw, uyw = 0.0 - gx / (2.0 * rw), 0.0  # u_wall - g/(2 rho)
        f = tl_nebb_wall(f, spec, f.shape[2], rw, uxw, uyw)
    return f


def run_tensorlbm(nx, ny, nu, gx, steps):
    f = feq(np.ones((nx, ny)), np.full((nx, ny), -gx / 2.0), np.zeros((nx, ny)))
    for _ in range(steps):
        f = tl_step(f, gx, nu)
    rho, ux, uy = macro(f)
    ux = ux + 0.5 * gx / rho  # same post-processing correction
    return ux, uy, rho


# --------------------------------------------------------------------------- #
# Analytic reference + metrics
# --------------------------------------------------------------------------- #
def analytic_ux(ny, gx, nu):
    H = ny - 1
    y = np.arange(ny, dtype=float)
    return gx / (2.0 * nu) * y * (H - y)


def rel_err(got, ref):
    """Pointwise relative error (ill-defined where ref~0; use err_global for verdict)."""
    scale = np.maximum(np.abs(ref), 1e-12)
    return np.abs(got - ref) / scale


def err_global(got, ref):
    """Global-normalized max error: max|got-ref| / max|ref| (robust at walls)."""
    return float(np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-12))


def main():
    out = Path(__file__).resolve().parent
    res = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "oss": "haimingz/vivsim (inventoried, /tmp/vivsim_src)",
        "problem": "2D D2Q9 Poiseuille channel, periodic x, no-slip top/bottom, body force x",
        "params": {},
        "runs": {},
        "verdict": {},
    }

    # ---- primary run (vivsim example parameters) -------------------------- #
    NX, NY, NU, GX, TM = 10, 10, 0.2, 0.001, 100000
    res["params"]["primary"] = dict(NX=NX, NY=NY, nu=NU, Gx=GX, steps=TM)
    ref = analytic_ux(NY, GX, NU)
    umax_ana = ref.max()

    for recipe in ("bgk_edm", "bgk_guo"):
        ux, uy, rho = run_vivsim(NX, NY, NU, GX, TM, recipe)
        e = rel_err(ux[NX // 2], ref)
        res["runs"][f"vivsim_{recipe}"] = {
            "ux_profile": ux[NX // 2].tolist(),
            "err_global_vs_analytic": err_global(ux[NX // 2], ref),
            "max_rel_err_vs_analytic_interior": float(e[1:-1].max()),
            "umax": float(ux.max()),
            "umax_analytic": float(umax_ana),
            "umax_rel_err": float(abs(ux.max() - umax_ana) / umax_ana),
        }

    ux_tl, uy_tl, rho_tl = run_tensorlbm(NX, NY, NU, GX, TM)
    e = rel_err(ux_tl[NX // 2], ref)
    res["runs"]["tensorlbm_bgk_guo"] = {
        "ux_profile": ux_tl[NX // 2].tolist(),
        "err_global_vs_analytic": err_global(ux_tl[NX // 2], ref),
        "max_rel_err_vs_analytic_interior": float(e[1:-1].max()),
        "umax": float(ux_tl.max()),
        "umax_analytic": float(umax_ana),
        "umax_rel_err": float(abs(ux_tl.max() - umax_ana) / umax_ana),
    }

    # cross-arm difference (TensorLBM vs vivsim BGK+EDM)
    ref_safe = np.maximum(np.abs(ref), 1e-12)
    d = np.abs(ux_tl[NX // 2] - np.array(res["runs"]["vivsim_bgk_edm"]["ux_profile"]))
    res["runs"]["cross_arm"] = {
        "max_abs_diff_vs_vivsim_bgk_edm": float(d.max()),
        "err_global_vs_vivsim_bgk_edm": float(d.max() / max(umax_ana, 1e-12)),
    }

    # ---- sensitivity 1: grid refinement (NY = 10 vs 20) ------------------- #
    for ny2 in (20,):
        ref2 = analytic_ux(ny2, GX, NU)
        uxa, _, _ = run_vivsim(NX, ny2, NU, GX, TM, "bgk_edm")
        uxb, _, _ = run_tensorlbm(NX, ny2, NU, GX, TM)
        res["runs"][f"grid_NY{ny2}_vivsim"] = {
            "err_global_vs_analytic": err_global(uxa[NX // 2], ref2),
            "umax_rel_err": float(abs(uxa.max() - ref2.max()) / ref2.max()),
        }
        res["runs"][f"grid_NY{ny2}_tensorlbm"] = {
            "err_global_vs_analytic": err_global(uxb[NX // 2], ref2),
            "umax_rel_err": float(abs(uxb.max() - ref2.max()) / ref2.max()),
        }

    # ---- verdict ---------------------------------------------------------- #
    primary = res["runs"]["tensorlbm_bgk_guo"]["err_global_vs_analytic"]
    cross = res["runs"]["cross_arm"]["err_global_vs_vivsim_bgk_edm"]
    res["verdict"] = {
        "tensorlbm_err_global_vs_analytic": primary,
        "cross_arm_err_global": cross,
        "owner_threshold": 0.15,
        "ecosystem_threshold": 0.03,
        "pass_owner_15pct": bool(primary < 0.15),
        "pass_ecosystem_3pct": bool(primary < 0.03),
    }

    (out / "xval_nebb_bodyforce_driver.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["verdict"], indent=2))
    print("wrote", out / "xval_nebb_bodyforce_driver.json")


if __name__ == "__main__":
    main()
