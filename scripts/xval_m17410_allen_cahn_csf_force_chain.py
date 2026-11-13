#!/usr/bin/env python3
"""XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN: TensorLBM vs lbmpy Allen-Cahn/CSF force chain.

Anchor: ``XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN`` (mission 17410, round 1, open-source
cross-validation).  Reference implementation: ``lssfau/lbmpy``,
``lbmpy/phasefield_allen_cahn/`` (Fakhari / van der Graaf Allen-Cahn model,
PhysRevE.96.053301) — read out of the installed source, nothing hand-copied.

What is compared
----------------
The *force chain* only, i.e. the mapping

    phi  ->  chemical potential mu(phi)  ->  capillary (CSF) force F = mu grad(phi)

and the endpoint physics that the chain must reproduce, the Young-Laplace jump.
Both sides are evaluated on **identical** phase fields, so any difference is a
discretisation difference, not a difference of the physical model.

Reference side (lbmpy): ``chemical_potential_symbolic`` and
``surface_tension_force`` from ``lbmpy/phasefield_allen_cahn/kernel_equations.py``
are kept symbolic and evaluated numerically by substituting the neighbour field
values (``_lbmpy_evaluator`` below).  The stencil weights printed in the report
are obtained by feeding a Kronecker delta through *lbmpy's own* expressions.

TensorLBM side: ``tensorlbm.phasefield.free_energy`` with the selectable
operator family ``central7`` (seven-point Laplacian + two-point gradient) or
``isotropic_d3q19`` (ported from lbmpy, ``tensorlbm.phasefield.isotropic_fd``).

Convention map (derived in the report, exact)
---------------------------------------------
phi_TL = 2 phi_lb - 1,  W_TL = W_lb / 2,  sigma unchanged  =>
beta_TL = beta_lb / 4,  kappa_TL = kappa_lb / 4,  F_TL == F_lb algebraically.

Metrics (predeclared in ``docs/XVAL_M17410_ALLEN_CAHN_CSF_FORCE_CHAIN.md``)
-------------------------------------------------------------------------
M1  rel-L2 of the capillary force field vs lbmpy                 (threshold 15%)
M2  recovered surface tension from the force integral, sigma_rec vs sigma input
    (Delta p = int F_r dr = 2 sigma / R in 3-D)                   (threshold 15%)
M3  spurious *tangential* force fraction ||F_t|| / ||F_r|| in the interface band
M4  net force |sum F| / sum |F| (must be ~0 in a periodic box)

Usage::

    python scripts/xval_m17410_allen_cahn_csf_force_chain.py            # full run
    python scripts/xval_m17410_allen_cahn_csf_force_chain.py --quick    # fast subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import lbmpy  # noqa: E402
from lbmpy.phasefield_allen_cahn.kernel_equations import (  # noqa: E402
    chemical_potential_symbolic,
    surface_tension_force,
)
from lbmpy.phasefield_allen_cahn.parameter_calculation import AllenCahnParameters  # noqa: E402
from lbmpy.stencils import LBStencil  # noqa: E402
from pystencils import Field  # noqa: E402

from tensorlbm.allen_cahn_lbm import free_energy_parameters  # noqa: E402
from tensorlbm.phasefield.free_energy import (  # noqa: E402
    DoubleWellFreeEnergy,
    force_mu_grad_phi,
)
from tensorlbm.phasefield.isotropic_fd import (  # noqa: E402
    FACE_DIAGONAL_OFFSETS,
    isotropic_gradient_3d,
    isotropic_laplacian_3d,
)
from tensorlbm.phasefield.operators import (  # noqa: E402
    central_gradient_3d,
    laplacian_3d,
)

LBMPY_PKG = Path(lbmpy.__file__).resolve().parent
LBMPY_SOURCES = (
    "phasefield_allen_cahn/kernel_equations.py",
    "phasefield_allen_cahn/derivatives.py",
    "phasefield_allen_cahn/parameter_calculation.py",
)
OUT_DIR = ROOT / "artifacts" / "xval_m17410"
FLOAT = torch.float64


# --------------------------------------------------------------------------------------
# reference side: lbmpy symbolic kernels evaluated numerically
# --------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lbmpy_provenance() -> dict:
    return {
        "package": "lbmpy",
        "version": getattr(lbmpy, "__version__", "unknown"),
        "origin": "lssfau/lbmpy (OSS inventory entry: allen-cahn-csf-force-chain)",
        "pystencils_version": __import__("pystencils").__version__,
        "source_files": {
            rel: {"path": str(LBMPY_PKG / rel), "sha256": _sha256(LBMPY_PKG / rel)}
            for rel in LBMPY_SOURCES
        },
        "python": platform.python_version(),
    }


def _symbol_offsets(phi_field, stencil) -> dict:
    """Map every neighbour symbol of ``phi_field`` onto its lattice offset."""
    return {phi_field.center.get_shifted(*off): tuple(off) for off in stencil.stencil_entries}


def _lbmpy_evaluator(expr: sp.Expr, sym2off: dict):
    """Return ``f(phi_zyx: np.ndarray) -> np.ndarray`` evaluating a lbmpy expression.

    Neighbour values are fetched with periodic rolls, so the evaluation uses the
    same discrete stencils that lbmpy would code-generate; only the boundary
    handling of lbmpy's generated kernel (periodic streaming) is reproduced.
    Offsets are in lbmpy's ``(x, y, z)`` order while the arrays are ``(z, y, x)``.
    """
    syms = sorted(expr.free_symbols, key=lambda s: sym2off[s])
    fn = sp.lambdify(syms, expr, "numpy")
    rolls = []
    for sym in syms:
        ox, oy, oz = sym2off[sym]
        rolls.append((ox, oy, oz))

    def evaluate(phi_zyx: np.ndarray) -> np.ndarray:
        args = []
        for ox, oy, oz in rolls:
            if ox == 0 and oy == 0 and oz == 0:
                args.append(phi_zyx)
            else:
                args.append(np.roll(phi_zyx, shift=(-oz, -oy, -ox), axis=(0, 1, 2)))
        return np.asarray(fn(*args), dtype=np.float64)

    return evaluate


class LbmpyForceChain:
    """lbmpy Allen-Cahn chemical potential + CSF (Korteweg) force, evaluated numerically."""

    def __init__(self, sigma: float, interface_thickness: float):
        self.sigma = float(sigma)
        self.W = float(interface_thickness)
        self.params = AllenCahnParameters(
            density_heavy=1.0,
            density_light=0.1,
            dynamic_viscosity_heavy=0.1,
            dynamic_viscosity_light=0.01,
            surface_tension=self.sigma,
            interface_thickness=int(round(self.W)),
        )
        self.stencil = LBStencil("D3Q19")
        self.field = Field.create_generic("phi", spatial_dimensions=3)
        self.sym2off = _symbol_offsets(self.field, self.stencil)
        # lbmpy builds beta/kappa as expressions in the *symbolic* parameters
        # (sigma, W, rho_H, ...); pin them to numbers with the parameter map that
        # lbmpy itself uses when it code-generates the kernel.
        subs = self.params.parameter_map()
        self.subs = subs
        mu_expr = sp.sympify(
            chemical_potential_symbolic(
                self.field, self.stencil, self.params.beta, self.params.kappa
            )
        ).subs(subs)
        f_exprs = [sp.sympify(e).subs(subs) for e in surface_tension_force(self.field, self.stencil, self.params)]
        self.exprs = {"mu": mu_expr, "fx": f_exprs[0], "fy": f_exprs[1], "fz": f_exprs[2]}
        self._fns = {k: _lbmpy_evaluator(v, self.sym2off) for k, v in self.exprs.items()}
        self.beta = float(sp.sympify(self.params.beta).subs(subs))
        self.kappa = float(sp.sympify(self.params.kappa).subs(subs))

    def __call__(self, phi_lb: np.ndarray) -> dict:
        mu = self._fns["mu"](phi_lb)
        # lbmpy returns the force with (x, y, z) components; store as (z, y, x)
        fz = self._fns["fz"](phi_lb)
        fy = self._fns["fy"](phi_lb)
        fx = self._fns["fx"](phi_lb)
        return {"mu": mu, "f": np.stack([fz, fy, fx])}


# --------------------------------------------------------------------------------------
# TensorLBM side
# --------------------------------------------------------------------------------------
class TensorLbmForceChain:
    """TensorLBM ``mu grad(phi)`` chain with a selectable finite-difference family."""

    def __init__(self, sigma: float, W_tl: float, operator: str):
        self.sigma = float(sigma)
        self.W = float(W_tl)
        self.operator = operator
        params = free_energy_parameters(self.sigma, self.W)
        self.beta = params.beta
        self.kappa = params.kappa
        self.free_energy = DoubleWellFreeEnergy(A=self.beta, B=self.beta, kappa=self.kappa)

    def __call__(self, phi_tl: np.ndarray) -> dict:
        phi = torch.as_tensor(phi_tl, dtype=FLOAT)
        mu = self.free_energy.chemical_potential(
            phi, boundary="periodic", operator=self.operator
        )
        fx, fy, fz = force_mu_grad_phi(
            phi, mu, boundary="periodic", operator=self.operator
        )
        return {
            "mu": mu.numpy(),
            "f": np.stack([fz.numpy(), fy.numpy(), fx.numpy()]),
        }


# --------------------------------------------------------------------------------------
# phase-field test cases (analytic equilibrium profiles)
# --------------------------------------------------------------------------------------
def planar_field(nx: int, W_lb: float, transverse: int = 8) -> np.ndarray:
    """lbmpy convention (phi in [0, 1]) planar interface at x = nx/2."""
    x = np.arange(nx, dtype=np.float64)
    phi_1d = 0.5 * (1.0 + np.tanh(2.0 * (x - 0.5 * nx) / W_lb))
    return np.broadcast_to(phi_1d, (transverse, transverse, nx)).copy()


def droplet_field(box: int, radius: float, W_lb: float) -> np.ndarray:
    """lbmpy convention (phi in [0, 1]) spherical droplet centred in a cubic box."""
    zz, yy, xx = np.meshgrid(
        *(np.arange(box, dtype=np.float64) - 0.5 * (box - 1) for _ in range(3)),
        indexing="ij",
    )
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    return 0.5 * (1.0 - np.tanh(2.0 * (r - radius) / W_lb))


def to_tensorlbm(phi_lb: np.ndarray) -> np.ndarray:
    """Convention map ``phi_lb in [0,1]  ->  phi_TL = 2 phi_lb - 1 in [-1,1]``."""
    return 2.0 * phi_lb - 1.0


def radial_unit(shape: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape
    zz, yy, xx = np.meshgrid(
        *(np.arange(n, dtype=np.float64) - 0.5 * (n - 1) for n in (nz, ny, nx)),
        indexing="ij",
    )
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    r_safe = np.where(r > 0, r, 1.0)
    return np.stack([zz / r_safe, yy / r_safe, xx / r_safe]), r


def metrics(ref: dict, got: dict, *, radius: float | None, W_lb: float, sigma: float) -> dict:
    f_ref, f_got = ref["f"], got["f"]
    mu_ref, mu_got = ref["mu"], got["mu"]
    res = {
        "rel_l2_force": float(np.linalg.norm(f_got - f_ref) / max(np.linalg.norm(f_ref), 1e-300)),
        "rel_linf_force": float(np.abs(f_got - f_ref).max() / max(np.abs(f_ref).max(), 1e-300)),
        "max_abs_force_ref": float(np.abs(f_ref).max()),
        "net_force_ratio": float(np.abs(f_got.sum(axis=(1, 2, 3))).max()
                                 / max(np.abs(f_got).sum(), 1e-300)),
    }
    if mu_ref is not None and mu_got is not None:
        res["rel_l2_mu_scaled"] = float(
            np.linalg.norm(2.0 * mu_got - mu_ref) / max(np.linalg.norm(mu_ref), 1e-300)
        )
    if radius is not None:
        n_hat, r = radial_unit(f_ref.shape[1:])
        band = np.abs(r - radius) <= 2.0 * W_lb
        f_r_ref = np.einsum("czyx,czyx->zyx", f_ref, n_hat)
        f_r_got = np.einsum("czyx,czyx->zyx", f_got, n_hat)
        f_t_ref = f_ref - f_r_ref[None] * n_hat
        f_t_got = f_got - f_r_got[None] * n_hat
        res["tangential_fraction_ref"] = float(
            np.linalg.norm(f_t_ref[:, band]) / max(np.linalg.norm(f_ref[:, band]), 1e-300)
        )
        res["tangential_fraction_got"] = float(
            np.linalg.norm(f_t_got[:, band]) / max(np.linalg.norm(f_got[:, band]), 1e-300)
        )
        force_localisation = float(np.linalg.norm(f_t_got[:, band]) / max(np.linalg.norm(f_t_ref[:, band]), 1e-300))
        res["spurious_force_amplification"] = force_localisation
        # sigma from int F_r dr = 2 sigma / R (3-D Laplace law), shell-averaged
        r_max = int(min(f_r_got.shape) // 2) - 2
        shell = np.floor(r + 0.5).astype(int)
        radial_profiles = {"r": list(range(r_max + 1)), "ref": [], "got": []}
        for k in radial_profiles["r"]:
            mask = shell == k
            radial_profiles["ref"].append(float(f_r_ref[mask].mean()) if mask.any() else 0.0)
            radial_profiles["got"].append(float(f_r_got[mask].mean()) if mask.any() else 0.0)
        # |.| : the Korteweg force points inward, so the ray integral is negative
        res["sigma_rec_ref"] = float(abs(radius / 2.0 * sum(radial_profiles["ref"])))
        res["sigma_rec_got"] = float(abs(radius / 2.0 * sum(radial_profiles["got"])))
        res["sigma_input"] = float(sigma)
        res["sigma_err_ref"] = float(res["sigma_rec_ref"] / sigma - 1.0)
        res["sigma_err_got"] = float(res["sigma_rec_got"] / sigma - 1.0)
        res["radial_profiles"] = radial_profiles
        res["radial_profile_rel_l2"] = float(
            np.linalg.norm(np.array(radial_profiles["got"]) - np.array(radial_profiles["ref"]))
            / max(np.linalg.norm(radial_profiles["ref"]), 1e-300)
        )
    return res


def stencil_weight_table() -> dict:
    """lbmpy vs TensorLBM D3Q19 operator weights, read off a Kronecker delta response."""
    from lbmpy.phasefield_allen_cahn.derivatives import (  # local import: reference only
        isotropic_gradient_symbolic,
        laplacian_symbolic,
    )

    phi_field = Field.create_generic("phi", spatial_dimensions=3)
    stencil = LBStencil("D3Q19")
    sym2off = _symbol_offsets(phi_field, stencil)
    lap_fn = _lbmpy_evaluator(laplacian_symbolic(phi_field, stencil), sym2off)
    grad_fn = _lbmpy_evaluator(isotropic_gradient_symbolic(phi_field, stencil)[0], sym2off)

    box = 5
    delta = np.zeros((box, box, box))
    delta[(2, 2, 2)] = 1.0
    lap_lb = lap_fn(delta)
    grad_lb = grad_fn(delta)

    phi_t = torch.zeros((box, box, box), dtype=FLOAT)
    phi_t[2, 2, 2] = 1.0
    lap_tl = isotropic_laplacian_3d(phi_t, boundary="periodic").numpy()
    grad_tl = isotropic_gradient_3d(phi_t, boundary="periodic")[0].numpy()

    rows = []
    for off in stencil.stencil_entries:  # pystencils (x, y, z) order
        ox, oy, oz = off
        pos = (2 + oz, 2 + oy, 2 + ox)  # array layout is (z, y, x)
        rows.append(
            {
                "offset_xyz": [int(v) for v in off],
                "l1_norm": int(abs(ox) + abs(oy) + abs(oz)),
                "lbmpy_lap": float(lap_lb[pos]),
                "tensorlbm_isotropic_lap": float(lap_tl[pos]),
                "lbmpy_grad_x": float(grad_lb[pos]),
                "tensorlbm_isotropic_grad_x": float(grad_tl[pos]),
            }
        )
    axis_row = next(r for r in rows if r["l1_norm"] == 1 and r["offset_xyz"][0] == 1)
    diag_row = next(r for r in rows if r["l1_norm"] == 2)
    centre = (2, 2, 2)
    table = {
        "lbmpy": {
            "laplacian_centre": float(lap_lb[centre]),
            "laplacian_axis": axis_row["lbmpy_lap"],
            "laplacian_face_diagonal": diag_row["lbmpy_lap"],
            "gradient_axis": axis_row["lbmpy_grad_x"],
            "gradient_face_diagonal": diag_row["lbmpy_grad_x"],
        },
        "tensorlbm_isotropic_d3q19": {
            "laplacian_centre": float(lap_tl[centre]),
            "laplacian_axis": float(lap_tl[2, 2, 3]),
            "laplacian_face_diagonal": float(lap_tl[2, 3, 3]),
            "gradient_axis": float(grad_tl[2, 2, 3]),
            "gradient_face_diagonal": float(grad_tl[2, 3, 3]),
        },
        "tensorlbm_central7": {
            "laplacian_centre": -6.0,
            "laplacian_axis": 1.0,
            "laplacian_face_diagonal": 0.0,
            "gradient_axis": 0.5,
            "gradient_face_diagonal": 0.0,
        },
        "max_abs_diff_isotropic_vs_lbmpy": float(
            max(
                abs(r["lbmpy_lap"] - r["tensorlbm_isotropic_lap"]) for r in rows
            )
            + max(abs(r["lbmpy_grad_x"] - r["tensorlbm_isotropic_grad_x"]) for r in rows)
        ),
        "rows": rows,
    }
    return table


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def _run_case(name: str, phi_lb: np.ndarray, sigma: float, W_lb: float,
              radius: float | None, operators: tuple[str, ...]) -> dict:
    ref_chain = LbmpyForceChain(sigma, W_lb)
    ref = ref_chain(phi_lb)
    phi_tl = to_tensorlbm(phi_lb)
    out = {
        "case": name,
        "grid_zyx": list(phi_lb.shape),
        "W_lb": W_lb,
        "W_tl": W_lb / 2.0,
        "sigma": sigma,
        "radius": radius,
        "lbmpy_beta": ref_chain.beta,
        "lbmpy_kappa": ref_chain.kappa,
        "variants": {},
    }
    for operator in operators:
        got = TensorLbmForceChain(sigma, W_tl=W_lb / 2.0, operator=operator)(phi_tl)
        out["variants"][operator] = metrics(ref, got, radius=radius, W_lb=W_lb, sigma=sigma)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the grid-refinement study")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    operators = ("central7", "isotropic_d3q19")
    sigma = 1.0e-2

    summary: dict = {
        "anchor": "XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN",
        "mission": 17410,
        "thresholds": {"rel_l2_force": 0.15, "sigma_error": 0.15},
        "lbmpy_provenance": lbmpy_provenance(),
        "operator_weights": stencil_weight_table(),
        "cases": [],
    }

    # C1: planar interface (isolates the local stencil, no curvature)
    summary["cases"].append(
        _run_case("C1-planar", planar_field(64, W_lb=4.0), sigma, 4.0, None, operators)
    )
    # C2: static droplet, the Laplace-law geometry
    summary["cases"].append(
        _run_case("C2-droplet-R16", droplet_field(64, 16.0, 4.0), sigma, 4.0, 16.0, operators)
    )
    # C3: interface-thickness sensitivity at fixed radius
    for W in (3.0, 4.0, 6.0):
        summary["cases"].append(
            _run_case(f"C3-droplet-W{int(W)}", droplet_field(64, 16.0, W), sigma, W, 16.0, operators)
        )
    # C4: grid refinement (R/box fixed, W fixed)
    if not args.quick:
        for radius in (8.0, 12.0, 16.0):
            summary["cases"].append(
                _run_case(
                    f"C4-droplet-R{int(radius)}",
                    droplet_field(int(4 * radius), radius, 4.0),
                    sigma,
                    4.0,
                    radius,
                    operators,
                )
            )
    # C5: surface-tension sensitivity (linear in sigma by construction)
    for sigma_case in (2.0e-2, 5.0e-3):
        summary["cases"].append(
            _run_case(
                f"C5-droplet-sigma{sigma_case:.0e}",
                droplet_field(64, 16.0, 4.0),
                sigma_case,
                4.0,
                16.0,
                operators,
            )
        )

    # ---- decision -------------------------------------------------------------
    ic = {c["case"]: c["variants"] for c in summary["cases"]}
    decision = {
        "C1_planar_rel_l2_central7": ic["C1-planar"]["central7"]["rel_l2_force"],
        "C1_planar_rel_l2_isotropic": ic["C1-planar"]["isotropic_d3q19"]["rel_l2_force"],
        "C2_droplet_rel_l2_central7": ic["C2-droplet-R16"]["central7"]["rel_l2_force"],
        "C2_droplet_rel_l2_isotropic": ic["C2-droplet-R16"]["isotropic_d3q19"]["rel_l2_force"],
        "C2_droplet_sigma_err_central7": ic["C2-droplet-R16"]["central7"]["sigma_err_got"],
        "C2_droplet_sigma_err_isotropic": ic["C2-droplet-R16"]["isotropic_d3q19"]["sigma_err_got"],
        "C2_droplet_sigma_err_lbmpy_self": ic["C2-droplet-R16"]["central7"]["sigma_err_ref"],
        "C2_spurious_frac_central7": ic["C2-droplet-R16"]["central7"]["tangential_fraction_got"],
        "C2_spurious_frac_isotropic": ic["C2-droplet-R16"]["isotropic_d3q19"]["tangential_fraction_got"],
        "C2_spurious_frac_lbmpy": ic["C2-droplet-R16"]["central7"]["tangential_fraction_ref"],
    }
    decision["isotropic_within_15pct"] = bool(
        decision["C2_droplet_rel_l2_isotropic"] < 0.15 and decision["C1_planar_rel_l2_isotropic"] < 0.15
    )
    decision["central7_within_15pct"] = bool(
        decision["C2_droplet_rel_l2_central7"] < 0.15 and decision["C1_planar_rel_l2_central7"] < 0.15
    )
    summary["decision"] = decision

    (OUT_DIR / "xval_allen_cahn_force_chain.json").write_text(json.dumps(summary, indent=2))

    # ---- flat table -----------------------------------------------------------
    lines = ["case,variant,rel_l2_force,rel_linf_force,sigma_err,tangential_fraction,net_force_ratio"]
    for case in summary["cases"]:
        for variant, m in case["variants"].items():
            lines.append(
                "{},{},{:.6e},{:.6e},{},{},{:.3e}".format(
                    case["case"],
                    variant,
                    m["rel_l2_force"],
                    m["rel_linf_force"],
                    f"{m['sigma_err_got']:.6e}" if "sigma_err_got" in m else "",
                    f"{m['tangential_fraction_got']:.6e}" if "tangential_fraction_got" in m else "",
                    m["net_force_ratio"],
                )
            )
    (OUT_DIR / "xval_allen_cahn_force_chain.csv").write_text("\n".join(lines) + "\n")

    # ---- figures --------------------------------------------------------------
    _make_figures(summary)

    print(json.dumps(summary["decision"], indent=2))
    print(f"\nartifacts written to {OUT_DIR}")
    return 0


def _make_figures(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c2 = next(c for c in summary["cases"] if c["case"] == "C2-droplet-R16")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    prof_ref = c2["variants"]["central7"]["radial_profiles"]
    ax.plot(prof_ref["r"], prof_ref["ref"], "k-", lw=2, label="lbmpy (ref)")
    for op, style in (("central7", "r--"), ("isotropic_d3q19", "b:")):
        prof = c2["variants"][op]["radial_profiles"]
        ax.plot(prof["r"], prof["got"], style, lw=1.6, label=f"TensorLBM {op}")
    ax.set_xlabel("r [lattice]")
    ax.set_ylabel(r"$F_r$ [lattice]")
    ax.set_title("Radial capillary force across the interface")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, vals = [], []
    for op in ("central7", "isotropic_d3q19"):
        labels.append(op)
        vals.append(c2["variants"][op]["rel_l2_force"])
    lb = np.array([0.0])
    ax.bar(labels + ["lbmpy(0 by def.)"], vals + [0.0], color=["tab:red", "tab:blue", "black"])
    ax.axhline(0.15, color="k", ls=":", lw=1.2, label="15% threshold")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("relative L2 error of F")
    ax.set_title("Force-chain error vs lbmpy (R=16, W=4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_force_chain_droplet.png", dpi=160)
    plt.close(fig)

    # refinement figure
    ref_cases = [c for c in summary["cases"] if c["case"].startswith("C4-droplet-R")]
    if ref_cases:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        for op, style in (("central7", "ro-"), ("isotropic_d3q19", "bs-")):
            xs = [c["radius"] for c in ref_cases]
            ys = [c["variants"][op]["rel_l2_force"] for c in ref_cases]
            ax.loglog(xs, ys, style, label=op)
        ax.axhline(0.15, color="k", ls=":", lw=1.2, label="15% threshold")
        ax.set_xlabel("droplet radius R [lattice]")
        ax.set_ylabel("relative L2 error of F")
        ax.set_title("Grid refinement (W=4 fixed)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_force_chain_refinement.png", dpi=160)
        plt.close(fig)

    # tangential (spurious) force map
    if "central7" in c2["variants"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
        W = 4.0
        for ax, op in zip(axes, ("central7", "isotropic_d3q19")):
            phi_lb = droplet_field(64, 16.0, W)
            chain = LbmpyForceChain(1e-2, W)
            ref = chain(phi_lb)["f"]
            got = TensorLbmForceChain(1e-2, W / 2.0, op)(to_tensorlbm(phi_lb))["f"]
            n_hat, r = radial_unit(phi_lb.shape)
            spurious = np.einsum("czyx,czyx->zyx", got - ref, n_hat) * 0.0
            diff = np.linalg.norm(got - ref, axis=0)
            im = ax.imshow(diff[32], cmap="inferno", origin="lower")
            ax.set_title(f"|F_TL - F_lbmpy|, {op}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_force_error_slice.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())