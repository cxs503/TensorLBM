"""Pointwise reference operator for LBM collision kernels, built from lbmpy.

Transplant of the OSS regression discipline used by ``lssfau/lbmpy``: a
collision operator is *constructed symbolically* (from the moment space, an
equilibrium, and a set of relaxation rates) and then evaluated **pointwise** --
per lattice node, per discrete direction -- on the same PDF field the kernel
under test sees.  Everything below drives lbmpy's own API; nothing is
transcribed by hand:

* the stencil and the moment basis come from ``lbmpy.stencils`` /
  ``lbmpy.moments``,
* the operator comes from ``lbmpy.methods.create_srt`` / ``create_trt`` or from
  ``MomentBasedLbMethod`` on a moment basis reconstructed out of the in-repo
  M matrix,
* the pointwise evaluator is ``LbmCollisionRule.lambdify`` (the same symbolic
  equations a generated kernel would execute),
* the direction permutation between TensorLBM's and lbmpy's index order is
  *derived* from the velocity tuples of the two lattices, never typed in.

The module is import-safe without lbmpy: the import failure is reported, and
callers decide whether that is a skip or a failure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

_CS_SYMBOL_NAME = "c_s"
_CS_VALUE = sp.sqrt(sp.Rational(1, 3))


def _require(mod_name: str):
    try:
        return __import__(mod_name, fromlist=["*"])
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"lbmpy reference unavailable: cannot import {mod_name}: {exc}") from exc


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def lbmpy_provenance() -> dict[str, Any]:
    """Version + sha256 of every lbmpy module this module drives."""
    lbmpy = _require("lbmpy")
    root = Path(lbmpy.__file__).resolve().parent
    rel_files = [
        "methods/creationfunctions.py",
        "methods/abstractlbmethod.py",
        "methods/momentbased/momentbasedmethod.py",
        "methods/conservedquantitycomputation.py",
        "moments.py",
        "stencils.py",
        "equilibrium/continuousmaxwellian.py",
    ]
    digests = {}
    for rel in rel_files:
        path = root / rel
        if path.exists():
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        pystencils = _require("pystencils")
        pystencils_version = getattr(pystencils, "__version__", "?")
    except RuntimeError:
        pystencils_version = None
    return {
        "lbmpy_version": getattr(lbmpy, "__version__", "?"),
        "lbmpy_root": str(root),
        "pystencils_version": pystencils_version,
        "source_sha256": digests,
    }


# ---------------------------------------------------------------------------
# lattice bookkeeping
# ---------------------------------------------------------------------------
def tensorlbm_velocities(lattice: str) -> np.ndarray:
    """Lattice velocities in TensorLBM's own direction order, read from the repo."""
    if lattice == "D2Q9":
        from tensorlbm.d2q9 import C  # noqa: PLC0415

        return C.numpy().astype(int)
    if lattice == "D3Q19":
        from tensorlbm.d3q19 import C  # noqa: PLC0415

        return C.numpy().astype(int)
    raise ValueError(f"unsupported lattice {lattice!r}")


def lbmpy_stencil(lattice: str):
    stencils = _require("lbmpy.stencils")
    return stencils.LBStencil(stencils.Stencil[lattice])


def direction_permutation(lattice: str) -> np.ndarray:
    """``perm[tl_i] = lbmpy_i`` where both directions have the same velocity.

    Derived by matching velocity tuples, so it stays correct if either library
    reorders its lattice.  Raises if the two lattices are not the same set.
    """
    tl_v = tensorlbm_velocities(lattice)
    lb_v = np.array([tuple(int(c) for c in v) for v in lbmpy_stencil(lattice)], dtype=int)
    lut = {tuple(int(x) for x in v): i for i, v in enumerate(lb_v)}
    perm = np.empty(len(tl_v), dtype=int)
    for i, v in enumerate(tl_v):
        key = tuple(int(x) for x in v)
        if key not in lut:
            raise ValueError(f"velocity {key} of TensorLBM {lattice} missing from lbmpy stencil")
        perm[i] = lut[key]
    if sorted(perm.tolist()) != list(range(len(lb_v))):
        raise ValueError("permutation is not a bijection")
    return perm


def permute(tl_major: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Re-index a ``(Q, *spatial)`` array from TensorLBM order to lbmpy order."""
    out = np.empty_like(tl_major)
    out[perm] = tl_major
    return out


def unpermute(lb_major: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`permute` (``perm`` is a bijection, not an involution)."""
    return lb_major[perm]


# ---------------------------------------------------------------------------
# moment basis reconstruction (Lallemand-Luo style M matrix -> lbmpy moments)
# ---------------------------------------------------------------------------
_MONOMIALS_D2Q9 = ["1", "x", "y", "x**2", "x*y", "y**2", "x**2*y", "x*y**2", "x**2*y**2"]
_MONOMIALS_D3Q19 = [
    "1",
    "x",
    "y",
    "z",
    "x**2",
    "y**2",
    "z**2",
    "x*y",
    "x*z",
    "y*z",
    "x**2*y",
    "x*y**2",
    "x**2*z",
    "x*z**2",
    "y**2*z",
    "y*z**2",
    "x**2*y**2",
    "x**2*z**2",
    "y**2*z**2",
]
_MONOMIALS = {"D2Q9": _MONOMIALS_D2Q9, "D3Q19": _MONOMIALS_D3Q19}


def moment_polynomials_from_matrix(matrix: np.ndarray, lattice: str) -> list[sp.Expr]:
    """Express every row of an M matrix as a polynomial in lbmpy's moment symbols.

    The ``(Q, Q)`` matrix rows are functions on the lattice; this solves exactly
    for their coefficients in a monomial basis that is unisolvent on the given
    stencil.  Rows come back in the *input* (TensorLBM) order.
    """
    moments_mod = _require("lbmpy.moments")
    xsyms = moments_mod.MOMENT_SYMBOLS
    vel = tensorlbm_velocities(lattice)
    dim = int(vel.shape[1])
    basis = [sp.sympify(term, locals={"x": xsyms[0], "y": xsyms[1], "z": xsyms[2]})
             for term in _MONOMIALS[lattice]]
    if matrix.shape != (len(vel), len(vel)):
        raise ValueError(f"M must be {len(vel)}x{len(vel)}, got {matrix.shape}")
    subs = [{xsyms[i]: sp.Integer(int(v[i])) for i in range(dim)} for v in vel]
    vandermonde = sp.Matrix(
        [[sp.expand(b.subs(s)) for b in basis] for s in subs]
    )
    if vandermonde.det() == 0:
        raise ValueError("monomial basis is not unisolvent on this stencil")
    inv = vandermonde.inv()
    polys = []
    for row in matrix:
        rhs = sp.Matrix([sp.nsimplify(float(x), rational=True) for x in row])
        coeffs = inv * rhs
        poly = sp.expand(sum(c * b for c, b in zip(coeffs, basis)))
        polys.append(poly)
    return polys


# ---------------------------------------------------------------------------
# the reference operator
# ---------------------------------------------------------------------------
class PointwiseReference:
    """A collision operator taken from lbmpy and evaluated pointwise.

    Parameters
    ----------
    lattice:
        ``"D2Q9"`` or ``"D3Q19"``.
    model:
        ``"srt"``, ``"trt"``, ``"mrt"`` (lbmpy ``create_mrt_raw`` default basis)
        or ``"mrt_from_matrix"`` (basis reconstructed from a supplied M matrix).
    params:
        model parameters, see :func:`build`.
    """

    def __init__(self, lattice: str, model: str, **params: Any) -> None:
        self.lattice = lattice
        self.model = model
        self.params = dict(params)
        self.stencil = lbmpy_stencil(lattice)
        self.method = build(lattice, model, **params)
        self.perm = direction_permutation(lattice)
        rule = self.method.get_collision_rule()
        q = len(tensorlbm_velocities(lattice))
        f_syms = [sp.Symbol(f"f_{i}") for i in range(q)]
        fixed = {}
        for sym in rule.free_symbols:
            name = str(sym)
            if name.startswith("f_"):
                continue
            if name == _CS_SYMBOL_NAME:
                fixed[sym] = _CS_VALUE
                continue
            raise RuntimeError(f"unfixed symbolic parameter {name!r} in the reference rule")
        self._func = rule.lambdify(f_syms, fixed_symbols=fixed) if fixed else rule.lambdify(f_syms)
        self._q = q
        self._d_syms = [sp.Symbol(f"d_{i}") for i in range(q)]
        self.symbolic_excerpt = str(rule.main_assignments[0])[:400]

    # -- evaluation ---------------------------------------------------------
    def apply(self, f_tl: np.ndarray) -> np.ndarray:
        """Collide ``f`` (``(Q, *spatial)``, TensorLBM order) pointwise."""
        f_lb = permute(np.asarray(f_tl, dtype=np.float64), self.perm)
        out = self._func(*[f_lb[i] for i in range(self._q)])
        d_lb = np.stack([out[sym] for sym in self._d_syms], axis=0).astype(np.float64)
        return unpermute(d_lb, self.perm)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PointwiseReference({self.lattice}, {self.model}, {self.params})"


def build(lattice: str, model: str, **params: Any):
    """Construct the lbmpy method object for a model."""
    creation = _require("lbmpy.methods.creationfunctions")
    st = lbmpy_stencil(lattice)
    if model == "srt":
        return creation.create_srt(st, 1.0 / params["tau"], compressible=True,
                                   continuous_equilibrium=params.get("continuous_equilibrium", True))
    if model == "trt":
        tau_plus = params["tau_plus"]
        lam = params.get("lambda_trt", 3.0 / 16.0)
        tau_minus = 0.5 + lam / (tau_plus - 0.5)
        return creation.create_trt(st, 1.0 / tau_plus, 1.0 / tau_minus, compressible=True)
    if model == "mrt":
        rates = params["relaxation_rates"]
        return creation.create_mrt_raw(st, rates, compressible=True)
    if model == "mrt_from_matrix":
        mbm = _require("lbmpy.methods.momentbased.momentbasedmethod")
        eq_mod = _require("lbmpy.equilibrium")
        cqc_mod = _require("lbmpy.methods.conservedquantitycomputation")
        matrix = np.asarray(params["moment_matrix"], dtype=float)
        rates = params["relaxation_rates"]
        polys = moment_polynomials_from_matrix(matrix, lattice)
        if len(rates) != len(polys):
            raise ValueError("one relaxation rate per moment row is required")
        eq = eq_mod.DiscreteHydrodynamicMaxwellian(st, order=2, compressible=True)
        cqc = cqc_mod.DensityVelocityComputation(st, compressible=True, zero_centered=False)
        return mbm.MomentBasedLbMethod(
            st, eq, {poly: float(rate) for poly, rate in zip(polys, rates)},
            conserved_quantity_computation=cqc,
        )
    raise ValueError(f"unknown model {model!r}")


def moment_basis_check(matrix: np.ndarray, lattice: str) -> float:
    """Max |M_lbmpy - M_repo| for a reconstructed basis (should be ~0)."""
    moments_mod = _require("lbmpy.moments")
    polys = moment_polynomials_from_matrix(matrix, lattice)
    st = lbmpy_stencil(lattice)
    lb = np.array(moments_mod.moment_matrix(polys, st), dtype=np.float64)
    perm = direction_permutation(lattice)
    repo = np.asarray(matrix, dtype=float)
    return float(np.abs(lb[:, perm] - repo).max())


# ---------------------------------------------------------------------------
# post-equilibrium test field (shared, bit-identical for both sides)
# ---------------------------------------------------------------------------
def equilibrium_field(lattice: str, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Discrete 2nd-order Maxwellian in TensorLBM order (``(Q, *spatial)``)."""
    vel = tensorlbm_velocities(lattice).astype(np.float64)
    w = _weights(lattice).reshape((len(vel),) + (1,) * rho.ndim)
    cu = np.tensordot(vel, u, axes=(1, 0))  # (Q, *spatial)
    u2 = (u ** 2).sum(axis=0, keepdims=True)
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2)


def _weights(lattice: str) -> np.ndarray:
    if lattice == "D2Q9":
        from tensorlbm.d2q9 import W_EXACT  # noqa: PLC0415

        return np.array(W_EXACT, dtype=np.float64)
    if lattice == "D3Q19":
        from tensorlbm.d3q19 import W_EXACT64  # noqa: PLC0415

        return np.array(W_EXACT64, dtype=np.float64)
    raise ValueError(lattice)


def make_test_field(lattice: str, *, tau: float, seed: int = 0, amplitude: float = 5e-2,
                    noise: float = 1e-3, shape: tuple[int, ...] | None = None,
                    dtype=np.float64) -> np.ndarray:
    """A positive, weakly compressible near-equilibrium PDF field.

    ``amplitude`` is the rms of each velocity component, so ``max|u| ~ 3 *
    amplitude``; ``noise`` adds an O(noise) non-equilibrium part so the
    non-hydrodynamic moments are actually exercised (a pure Maxwellian is
    invariant under most collision operators and would hide defects).
    """
    if shape is None:
        shape = (8, 8) if lattice == "D2Q9" else (4, 4, 4)
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    ndim = 2 if lattice == "D2Q9" else 3
    u = amplitude * rng.standard_normal((ndim, *shape))
    f = equilibrium_field(lattice, rho, u)
    f = f + noise * rng.standard_normal(f.shape)
    if dtype is np.float32:
        f = f.astype(np.float32).astype(np.float64)
    return f


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compare(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Pointwise deviation metrics between a kernel output and a reference."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = got - ref
    scale = float(np.abs(ref).max())
    denom = np.maximum(np.abs(ref), 1e-300)
    return {
        "eps_scaled_inf": float(np.abs(diff).max() / max(scale, 1e-300)),
        "eps_rel_pointwise_max": float((np.abs(diff) / denom).max()),
        "eps_rel_rms": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-300)),
        "abs_diff_max": float(np.abs(diff).max()),
        "scale_abs_max": scale,
    }


def moments(f: np.ndarray, lattice: str) -> dict[str, np.ndarray]:
    """rho and momentum of a ``(Q, *spatial)`` field in TensorLBM order."""
    vel = tensorlbm_velocities(lattice).astype(np.float64)
    rho = f.sum(axis=0)
    j = np.tensordot(vel, f, axes=(0, 0))
    return {"rho": rho, "j": j}


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=float)