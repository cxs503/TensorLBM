"""XVAL-LBM-LBMCONFIG-SINGLE-POINT-CONFIG (m17952).

Cross-validation of the *single-point configuration* approach.

Reference (OSS): lssfau/lbmpy ``LBMConfig`` dataclass
    (lbmpy/creationfunctions.py:101) -- one dataclass coalesces stencil,
    method, relaxation rate, force model and boundary handling, from which
    the collision rule / streaming rule / AST / compiled kernel are derived.
    Driven here through the high-level ``LatticeBoltzmannStep``.

Test (this repo): TensorLBM D2Q9 BGK primitives
    (src/tensorlbm/d2q9.py, src/tensorlbm/solver.py) configured by explicit
    per-call arguments (tau, force) rather than one coalesced config object.

Case: 2D D2Q9 BGK/SRT pressure-driven channel (plane Poiseuille),
periodic in x, no-slip top/bottom, Guo body force in +x.
Analytic steady profile: u(y) = G/(2 nu) * s * (H+1 - s), s = y - y_w.

Decision under test: do the two configuration styles produce the same
physics?  Acceptance: max |u_got - u_ref| / max|u_ref| < 15% (owner standard).
"""
from __future__ import annotations

import json

import numpy as np

# ---------------------------------------------------------------------------
# Case definition (shared by both implementations)
# ---------------------------------------------------------------------------
CASE = {
    "name": "d2q9_bgk_poiseuille_guo",
    "nx": 128,
    "ny": 49,          # includes 2 solid wall rows (top & bottom)
    "omega": 0.10,     # relaxation rate  -> tau = 1/omega + 0.5
    "fx": 2.0e-4,      # body force per unit mass in +x
    "steps": 4000,
    "sample_from": 3000,
    "sample_every": 10,
    "seed": 20261113,
}


def omega_to_nu(omega: float) -> float:
    """LBM kinematic viscosity: nu = cs2 (tau - 0.5), cs2 = 1/3."""
    tau = 1.0 / omega + 0.5
    return (1.0 / 3.0) * (tau - 0.5)


# D2Q9 velocities / weights (shared, exact)
C2 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
               [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
W2 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP2 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def macro_from_pdf(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rho, ux, uy from a (9, ny, nx) pdf (numpy, no ghost)."""
    rho = f.sum(axis=0)
    rs = np.clip(rho, 1e-12, None)
    ux = (f * C2[:, 0:1, None]).sum(axis=0) / rs
    uy = (f * C2[:, 1:2, None]).sum(axis=0) / rs
    return rho, ux, uy


# ---------------------------------------------------------------------------
# Reference: lbmpy LBMConfig (single-point config)
# ---------------------------------------------------------------------------
def run_lbmpy(case: dict) -> dict:
    from lbmpy import LatticeBoltzmannStep, LBMConfig
    from lbmpy.enums import Stencil, Method, ForceModel
    from lbmpy.stencils import LBStencil
    from lbmpy.boundaries import NoSlip

    nx, ny = case["nx"], case["ny"]
    omega, fx = case["omega"], case["fx"]

    # --- the single-point configuration -----------------------------------
    lbm_config = LBMConfig(
        stencil=LBStencil(Stencil.D2Q9),
        method=Method.SRT,
        relaxation_rate=omega,
        force_model=ForceModel.GUO,
        force=(fx, 0.0),
    )

    lbm = LatticeBoltzmannStep(
        domain_size=(ny, nx),
        lbm_config=lbm_config,
        periodicity=(True, False),
    )
    # no-slip on top & bottom rows
    lbm.boundary_handling.set_boundary(NoSlip(), slice_obj=(0, slice(None)))
    lbm.boundary_handling.set_boundary(NoSlip(), slice_obj=(ny - 1, slice(None)), replace=False)

    # --- initial condition: rest + tiny perturbation ----------------------
    pdf = lbm.data_handling.cpu_arrays[lbm.pdf_array_name]  # (ny+2, nx+2, 9)
    pdf[:] = 1.0 / 9.0
    rng = np.random.default_rng(case["seed"])
    pdf += 1e-6 * rng.standard_normal(pdf.shape)

    # --- run --------------------------------------------------------------
    u_samples = []
    for step in range(1, case["steps"] + 1):
        lbm.time_step()
        if step >= case["sample_from"] and step % case["sample_every"] == 0:
            f_phys = pdf[1:-1, 1:-1, :]  # (ny, nx, 9)
            f_phys = np.moveaxis(f_phys, -1, 0)  # (9, ny, nx)
            _, ux, _ = macro_from_pdf(f_phys)
            u_samples.append(ux[1:ny - 1].copy())

    u_mean = np.mean(np.stack(u_samples), axis=0)
    return {
        "impl": "lbmpy.LBMConfig",
        "u_profile": u_mean.mean(axis=1),
        "y": np.arange(1, ny - 1, dtype=float),
        "nu": omega_to_nu(omega),
        "tau": 1.0 / omega + 0.5,
    }


# ---------------------------------------------------------------------------
# Test: TensorLBM D2Q9 BGK (explicit per-call configuration)
# ---------------------------------------------------------------------------
def run_tensorlbm(case: dict) -> dict:
    import sys

    import torch

    sys.path.insert(0, "/root/tp-agent/work/m17952/src")
    from tensorlbm.d2q9 import C, W, OPPOSITE, macroscopic
    from tensorlbm.solver import collide_bgk, stream

    nx, ny = case["nx"], case["ny"]
    omega, fx = case["omega"], case["fx"]
    tau = 1.0 / omega + 0.5
    dev = torch.device("cpu")

    c = C.to(dev).double()
    w = W.to(dev).double().view(9, 1, 1)
    opp = OPPOSITE.to(dev)

    def guo_force(f, ux, uy):
        """Guo split forcing, post-collision, pure +x force (matches
        tensorlbm.channel_kernel_numpy.apply_body_force in 2D)."""
        uf = ux * fx
        out = f.clone()
        for i in range(9):
            cx, cy = float(c[i, 0]), float(c[i, 1])
            c_dot_f = cx * fx
            cu = cx * ux + cy * uy
            out[i] += w[i] * (3.0 * (c_dot_f - uf) + 9.0 * cu * c_dot_f)
        return out

    rng = np.random.default_rng(case["seed"])
    f = torch.zeros((9, ny, nx), dtype=torch.float64, device=dev)
    f += 1.0 / 9.0
    f += 1e-6 * torch.from_numpy(
        rng.standard_normal((9, ny, nx)).astype(np.float64)
    )
    f[:, 0, :] = 1.0 / 9.0
    f[:, ny - 1, :] = 1.0 / 9.0

    def bounce_back(f):
        out = f.clone()
        for i in range(9):
            oi = int(opp[i])
            out[i, 0, :] = f[oi, 0, :]
            out[i, ny - 1, :] = f[oi, ny - 1, :]
        return out

    u_samples = []
    for step in range(1, case["steps"] + 1):
        f = collide_bgk(f, tau)
        rho, ux, uy = macroscopic(f)
        f = guo_force(f, ux, uy)
        f = stream(f)
        f = bounce_back(f)
        if step >= case["sample_from"] and step % case["sample_every"] == 0:
            rho, ux, uy = macroscopic(f)
            u_samples.append(ux[1:ny - 1].cpu().numpy())

    u_mean = np.mean(np.stack(u_samples), axis=0)
    return {
        "impl": "tensorlbm.d2q9.bgk",
        "u_profile": u_mean.mean(axis=1),
        "y": np.arange(1, ny - 1, dtype=float),
        "nu": omega_to_nu(omega),
        "tau": tau,
    }


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------
def analytic(case: dict, nu: float) -> tuple[np.ndarray, np.ndarray]:
    ny = case["ny"]
    fx = case["fx"]
    height = ny - 2
    y = np.arange(1, ny - 1, dtype=float)
    s = y  # wall origin at y=0 (first fluid row centre at s=1)
    upper = height + 1.0
    u = fx / (2.0 * nu) * s * (upper - s)
    return y, u


def main() -> None:
    case = dict(CASE)
    ref = run_lbmpy(case)
    got = run_tensorlbm(case)
    y, u_ana = analytic(case, ref["nu"])

    u_ref = ref["u_profile"]
    u_got = got["u_profile"]
    u_max = float(np.max(np.abs(u_ref)))

    err_ref_ana = float(np.max(np.abs(u_ref - u_ana)) / u_max)
    err_got_ana = float(np.max(np.abs(u_got - u_ana)) / u_max)
    err_cross = float(np.max(np.abs(u_got - u_ref)) / u_max)
    ucl_ref = float(u_ref[len(u_ref) // 2])
    ucl_got = float(u_got[len(u_got) // 2])
    ucl_ana = float(u_ana[len(u_ana) // 2])

    result = {
        "case": case,
        "nu": ref["nu"],
        "tau": ref["tau"],
        "ucl_ref": ucl_ref,
        "ucl_got": ucl_got,
        "ucl_ana": ucl_ana,
        "err_ref_vs_ana": err_ref_ana,
        "err_got_vs_ana": err_got_ana,
        "err_cross": err_cross,
        "ucl_err_cross": abs(ucl_got - ucl_ref) / abs(ucl_ref),
        "accept_15pct": err_cross < 0.15,
    }
    out = {
        "result": result,
        "y": ref["y"].tolist(),
        "u_ref": u_ref.tolist(),
        "u_got": u_got.tolist(),
        "u_ana": u_ana.tolist(),
    }
    with open("/root/tp-agent/work/m17952/xval/out_lbmconfig_single_point.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
