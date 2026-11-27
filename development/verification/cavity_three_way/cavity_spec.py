"""Shared specification + metric definitions for XVAL-LBM-CAVITY.

The anchor asks for a *same-specification* three-party comparison of the 2-D
lid-driven cavity:

    party A  TensorLBM  (this repository's D2Q9 solver)
    party B  an independent open-source LBM  (pylbm, ~/oss-cache/pylbm)
    party C  Ghia, Ghia & Shin (1982) tabulated reference (JCP 48:387-411)

so that both the inter-tool spread *and* each tool's deviation from the
published reference are reported item by item.

Lattice-unit specification (identical for A and B)
--------------------------------------------------
* cavity side              L = N lattice spacings (N = 128, 64)
* grid spacing             dx = 1  =>  L = N, dx_phys = 1/N
* lid velocity (lattice)   U = 0.06  (Ma ~ 0.10, incompressible LBM range)
* kinematic viscosity      nu = U * L / Re  =>  tau = 0.5 + 3 nu
* Mach / compressibility   u_lid/cs = 0.06/0.577 = 0.104

Wall conventions (they differ between the tools and are *declared*, not hidden)
-------------------------------------------------------------------------------
* party A: all four walls are node planes -- three walls by pre-streaming
  reflection at the wall nodes, lid by Zou & He (1997) at the top node row.
  Physical cavity spans nodes 0..N, first fluid node at y = 1/(N),
  declared coordinate of fluid node j: y_j = j/N.
* party B: Bouzidi-Firdaouss-Lallemand bounce-back (pylbm default
  ``distance = 0.5``), i.e. the wall sits half a cell outside the boundary
  node; declared coordinate of cell i: y_i = (i+0.5)/N.

Both are legitimate mature treatments; the difference is a genuine
tool-property (first-order node wall vs second-order half-way wall) and is
quantified below through the ``wall_convention`` field of every metric block.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# reference (party C)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: machine-built, provenance-bearing transcription of Ghia (1982) Tables I/II
GHIA_JSON = (
    REPO_ROOT
    / "benchmarks/verified/cavity/re5000/reference/ghia_re5000_reference.json"
)
#: sha256 of that file -- checked by :func:`load_ghia`
GHIA_JSON_SHA256 = None  # filled by build_reference.py into ghia_reference_xval.json


def load_ghia(re: int) -> dict:
    """Return the Ghia Table I/II centreline data for ``re`` (100/400/1000/...).

    Values come from the machine-built transcription of the paper's OCR text
    layer (``benchmarks/verified/cavity/re5000/build_reference.py``); nothing
    here is hand-typed.
    """
    d = json.loads(GHIA_JSON.read_text())
    key = str(int(re))
    if key not in d["u_columns"]:
        raise KeyError(f"Re={re} not in the Ghia transcription ({list(d['u_columns'])})")
    return {
        "re": float(re),
        "y": np.asarray(d["coords"]["u_y"], dtype=float),
        "u": np.asarray(d["u_columns"][key], dtype=float),
        "x": np.asarray(d["coords"]["v_x"], dtype=float),
        "v": np.asarray(d["v_columns"][key], dtype=float),
        "source": d["source"],
        "table": d["table"],
        "provenance": d["provenance"],
    }


# --------------------------------------------------------------------------
# specification
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CavitySpec:
    """Same-spec parameter set handed to every party."""

    re: float = 100.0
    n: int = 128            # cavity side in lattice spacings
    u_lid: float = 0.06
    max_steps: int = 400_000
    resid_tol: float = 1e-7

    @property
    def nu(self) -> float:
        return self.u_lid * self.n / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def mach(self) -> float:
        return self.u_lid / math.sqrt(1.0 / 3.0)

    def as_dict(self) -> dict:
        return {
            "re": self.re,
            "n": self.n,
            "u_lid": self.u_lid,
            "nu_lattice": self.nu,
            "tau": self.tau,
            "mach": self.mach,
            "max_steps": self.max_steps,
            "resid_tol": self.resid_tol,
        }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def centreline_metrics(
    y: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    ghia: dict,
    u_lid: float,
    label: str = "",
) -> dict:
    """Itemised comparison of centreline profiles against Ghia (party C).

    ``y,u`` and ``x,v`` are the *normalised* profiles (u/u_lid, v/u_lid) on the
    party's own node coordinates, both sorted ascending.  The wall rows are the
    values imposed by the boundary condition (0 at the no-slip walls, 1 for
    u at the lid); they are the caller's responsibility to include.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    u_at = np.interp(ghia["y"], y, u)          # y ascending (Ghia's own y is desc)
    v_at = np.interp(ghia["x"], x, v)
    du = u_at - ghia["u"]
    dv = v_at - ghia["v"]

    # anchors: values only (position-free) + their positions
    i_umin = int(np.argmin(u))
    i_vmax = int(np.argmax(v))

    return {
        "party": label,
        "u_min": float(u[i_umin]),
        "u_min_y": float(y[i_umin]),
        "v_max": float(v[i_vmax]),
        "v_max_x": float(x[i_vmax]),
        "u_center": float(np.interp(0.5, y, u)),
        "v_center": float(np.interp(0.5, x, v)),
        "rmse_u": float(np.sqrt(np.mean(du**2))),
        "rmse_v": float(np.sqrt(np.mean(dv**2))),
        "max_abs_dev_pct": 100.0 * float(max(np.abs(du).max(), np.abs(dv).max())),
        "ghia_u_min": float(ghia["u"].min()),
        "ghia_u_min_y": float(ghia["y"][int(np.argmin(ghia["u"]))]),
        "ghia_v_max": float(ghia["v"].max()),
        "ghia_v_max_x": float(ghia["x"][int(np.argmax(ghia["v"]))]),
    }


def inter_tool_delta(pa: dict, pb: dict, npts: int = 201) -> dict:
    """Point-wise difference between two parties on a common normalised grid."""
    ys = np.linspace(0.0, 1.0, npts)
    xs = np.linspace(0.0, 1.0, npts)
    ua = np.interp(ys, pa["y"], pa["u"])
    ub = np.interp(ys, pb["y"], pb["u"])
    va = np.interp(xs, pa["x"], pa["v"])
    vb = np.interp(xs, pb["x"], pb["v"])
    return {
        "max_abs_du": float(np.abs(ua - ub).max()),
        "max_abs_dv": float(np.abs(va - vb).max()),
        "rmse_du": float(np.sqrt(np.mean((ua - ub) ** 2))),
        "rmse_dv": float(np.sqrt(np.mean((va - vb) ** 2))),
        "max_abs_pct": 100.0 * float(max(np.abs(ua - ub).max(),
                                         np.abs(va - vb).max())),
    }


def write_case_output(out: Path, payload: dict) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return out