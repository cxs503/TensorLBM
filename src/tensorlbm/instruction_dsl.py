"""Declarative **instruction DSL** for lattice-Boltzmann runs.

Motivation
----------
TensorLBM grows one hand-written solver per physics case.  The OSS practice this
module ports is the *declarative* one: a run is described by **instructions**
(stencil, moment polynomials, equilibrium moments, relaxation rates, domain,
boundaries, initial state, forcing, run control), the description is validated
against lattice/physics invariants, and only then lowered into an executable
plan.  Reference (read-only cache ``/root/oss-cache/pylbm``, commit ``f9f5d16``):

=====================================  =========================================
pylbm artefact                         what it does
=====================================  =========================================
``pylbm/stencil.py:643``               ``Stencil(dico, need_validation=True)`` --
                                       a *dict* instruction block declaring
                                       ``dim``, ``scheme_velocity`` and the
                                       per-scheme ``velocities`` (standard
                                       ``Velocity(dim, num)`` D2Q9 numbering).
``pylbm/validator.py:187``             ``validate(dico, name)`` -- recursive
                                       schema/type validation *before* numerics.
``pylbm/scheme.py:238``                ``_get_equilibrium`` -- equilibrium declared
                                       as symbolic expressions in the moments.
``pylbm/scheme.py:506``                ``_create_moments_matrices`` -- builds
                                       ``M[i,j] = P_i(c_j * LA)``, its inverse,
                                       hence ``f_eq = M^-1 . EQ``.
``demo/examples/scheme/eqpde_d2q9_navier_stokes.py``
                                       canonical D2Q9 instruction block
                                       (polynomials in ``X, Y``, equilibrium in
                                       ``rho, qx, qy``, ``s = 1/(0.5 + sigma)``).
=====================================  =========================================

This module keeps that semantics with a line-oriented text grammar (a run becomes
a reviewable artifact instead of Python code) and adds the two things the OSS
leaves to the user: invariant self-checks on the lowered scheme (partition of
unity, moment recovery, isotropy, stability guards) and an executable plan in
which the *declared* scheme -- not a hard-coded one -- drives collision.

Grammar (``#`` starts a comment; only ``polynomial`` / ``equilibrium`` carry
spaces inside their argument)::

    stencil dim=2 velocity=1.0 velocities=(0,0);(1,0);(0,1);(-1,0);(0,-1);(1,1);(-1,1);(-1,-1);(1,-1)
    polynomial <expr in X,Y>            # one per moment, in moment-space order
    equilibrium <expr in rho,ux,uy>     # same count as polynomials
    relaxation 0,0,0,1.8,1.8,1.8,1.8,1.8,1.8
    domain nx=64 ny=32
    boundary xmin=periodic xmax=periodic ymin=bounce_back ymax=bounce_back
    init uniform rho=1.0 ux=0.0 uy=0.0   |   init taylor_green rho=1.0 amp=0.05 mode=1
    force gx=1e-5 gy=0.0
    run steps=20000 tol=1e-10

Expressions are evaluated numerically through a whitelisted Python AST
(``+ - * / **``; ``^`` is accepted for ``**``) with only the declared symbols
visible, so an instruction file cannot execute arbitrary code.

Error taxonomy (all derive from :class:`DSLError`):

* :class:`DSLSyntaxError` -- bad grammar, unknown instruction/argument/symbol;
* :class:`DSLSemanticError` -- well formed but lattice/physics inconsistent:
  singular moment matrix, negative/non-finite weights, mass or momentum not
  recovered by the declared equilibrium, relaxation outside ``[0, 2)``, non
  positive viscosity, unsupported boundary mix.

The lowering covers the two-dimensional all-periodic and channel (bounce-back
walls in ``y``, periodic in ``x``) layouts; anything else raises
:class:`DSLSemanticError` instead of silently running a different problem.
"""

from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "DSLError",
    "DSLSyntaxError",
    "DSLSemanticError",
    "Program",
    "Scheme",
    "RunResult",
    "parse",
    "build_scheme",
    "run",
    "D2Q9_SCHEME_INSTRUCTIONS",
    "D2Q9_POLYNOMIALS",
    "D2Q9_EQUILIBRIA",
    "taylor_green_velocity",
    "decay_rate_from_history",
    "poiseuille_error",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class DSLError(ValueError):
    """Base class for instruction-DSL failures."""


class DSLSyntaxError(DSLError):
    """The instruction text is not well formed."""


class DSLSemanticError(DSLError):
    """The instruction text parses but violates a lattice/physics invariant."""


# ---------------------------------------------------------------------------
# safe expression evaluation
# ---------------------------------------------------------------------------
_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Name,
    ast.Load,
)


def _compile_expression(src: str, allowed: Sequence[str], *, where: str) -> Any:
    """Compile ``src`` after whitelisting the AST and the visible symbols."""
    text = src.replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise DSLSyntaxError(f"{where}: cannot parse expression {src!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise DSLSyntaxError(
                f"{where}: unsupported construct {type(node).__name__} in {src!r}"
            )
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise DSLSyntaxError(
                f"{where}: unknown symbol {node.id!r} in {src!r} "
                f"(allowed: {list(allowed)})"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise DSLSyntaxError(f"{where}: non-numeric constant in {src!r}")
    return compile(tree, f"<instruction_dsl:{where}>", "eval")


def _evaluate(code: Any, env: Mapping[str, Any]) -> Any:
    return eval(code, {"__builtins__": {}}, dict(env))  # noqa: S307 - AST whitelisted


# ---------------------------------------------------------------------------
# the D2Q9 second-order scheme written in the grammar
# ---------------------------------------------------------------------------
_D2Q9_VELOCITIES = "(0,0);(1,0);(0,1);(-1,0);(0,-1);(1,1);(-1,1);(-1,-1);(1,-1)"

D2Q9_POLYNOMIALS: Tuple[str, ...] = (
    "1",
    "X",
    "Y",
    "X^2+Y^2",
    "X^2-Y^2",
    "X*Y",
    "X*Y^2",
    "X^2*Y",
    "X^2*Y^2",
)

#: Isothermal second-order equilibrium written in the conserved variables.
D2Q9_EQUILIBRIA: Tuple[str, ...] = (
    "rho",
    "rho*ux",
    "rho*uy",
    "2/3*rho + rho*(ux^2+uy^2)",
    "rho*(ux^2-uy^2)",
    "rho*ux*uy",
    "1/3*rho*ux + rho*ux*uy^2",
    "1/3*rho*uy + rho*ux^2*uy",
    "rho*(1/3+ux^2)*(1/3+uy^2)",
)

_SYMBOLS_POLYNOMIAL = ("X", "Y")
_SYMBOLS_EQUILIBRIUM = ("rho", "ux", "uy")

_KNOWN_INSTRUCTIONS = (
    "stencil",
    "polynomial",
    "equilibrium",
    "relaxation",
    "domain",
    "boundary",
    "init",
    "force",
    "run",
)

_BOUNDARY_KINDS = ("periodic", "bounce_back")
_BOUNDARY_KEYS = ("xmin", "xmax", "ymin", "ymax")


def _d2q9_scheme_instructions(relaxation: float = 1.8) -> str:
    """Emit the D2Q9 scheme block as instruction text."""
    lines = [
        "# D2Q9 second-order scheme declared in the instruction grammar.",
        f"stencil dim=2 velocity=1.0 velocities={_D2Q9_VELOCITIES}",
    ]
    lines += [f"polynomial {p}" for p in D2Q9_POLYNOMIALS]
    lines += [f"equilibrium {e}" for e in D2Q9_EQUILIBRIA]
    rates = [0.0, 0.0, 0.0] + [float(relaxation)] * 6
    lines.append("relaxation " + ",".join(f"{r:g}" for r in rates))
    return "\n".join(lines) + "\n"


#: Canonical D2Q9 scheme block (no domain/run instructions).
D2Q9_SCHEME_INSTRUCTIONS: str = _d2q9_scheme_instructions()


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def _pairs(rest: str, *, where: str, allowed: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in rest.split():
        if "=" not in token:
            raise DSLSyntaxError(f"{where}: expected key=value, got {token!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise DSLSyntaxError(f"{where}: empty key in {token!r}")
        if allowed and key not in allowed:
            raise DSLSyntaxError(
                f"{where}: unknown argument {key!r} (allowed: {list(allowed)})"
            )
        if key in out:
            raise DSLSyntaxError(f"{where}: duplicate argument {key!r}")
        out[key] = value
    return out


def _as_float(pairs: Mapping[str, str], key: str, where: str, default=None) -> float:
    if key not in pairs:
        if default is None:
            raise DSLSyntaxError(f"{where}: missing required argument {key!r}")
        return float(default)
    try:
        return float(pairs[key])
    except ValueError as exc:
        raise DSLSyntaxError(f"{where}: {key}={pairs[key]!r} is not a number") from exc


def _as_int(pairs: Mapping[str, str], key: str, where: str, default=None) -> int:
    value = _as_float(pairs, key, where, default)
    if abs(value - round(value)) > 1e-12:
        raise DSLSyntaxError(f"{where}: {key} must be an integer, got {value}")
    return int(round(value))


def _parse_velocities(text: str, where: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not (chunk.startswith("(") and chunk.endswith(")")):
            raise DSLSyntaxError(f"{where}: velocity {chunk!r} must look like (cx,cy)")
        parts = [p.strip() for p in chunk[1:-1].split(",")]
        if len(parts) != 2:
            raise DSLSyntaxError(f"{where}: velocity {chunk!r} must have 2 components")
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise DSLSyntaxError(f"{where}: velocity {chunk!r} is not numeric") from exc
    if not out:
        raise DSLSyntaxError(f"{where}: empty velocity set")
    return out


@dataclass
class Program:
    """A parsed, structurally validated instruction program."""

    source: str
    dim: int
    velocity: float
    velocities: List[Tuple[float, float]]
    polynomials: List[str]
    equilibria: List[str]
    relaxation: List[float]
    nx: int
    ny: int
    boundary: Dict[str, str]
    init: Dict[str, Any]
    force: Tuple[float, float]
    steps: int
    tol: float
    _poly_codes: List[Any] = field(default_factory=list, repr=False)
    _eq_codes: List[Any] = field(default_factory=list, repr=False)

    @property
    def digest(self) -> str:
        """SHA-256 of the exact instruction text (provenance, zero hand-copy)."""
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "digest": self.digest,
            "dim": self.dim,
            "velocity": self.velocity,
            "velocities": [list(v) for v in self.velocities],
            "polynomials": list(self.polynomials),
            "equilibria": list(self.equilibria),
            "relaxation": list(self.relaxation),
            "domain": {"nx": self.nx, "ny": self.ny},
            "boundary": dict(self.boundary),
            "init": dict(self.init),
            "force": list(self.force),
            "run": {"steps": self.steps, "tol": self.tol},
        }


def parse(text: str) -> Program:
    """Parse and structurally validate an instruction program."""
    if not isinstance(text, str) or not text.strip():
        raise DSLSyntaxError("empty instruction program")

    stencil: Dict[str, str] | None = None
    polynomials: List[str] = []
    poly_codes: List[Any] = []
    equilibria: List[str] = []
    eq_codes: List[Any] = []
    relaxation: List[float] | None = None
    domain: Dict[str, str] | None = None
    boundary: Dict[str, str] = {}
    init: Dict[str, Any] | None = None
    force_pairs: Dict[str, str] = {}
    run_pairs: Dict[str, str] | None = None

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"line {lineno}"
        head, _, rest = line.partition(" ")
        name = head.strip().lower()
        rest = rest.strip()
        if name not in _KNOWN_INSTRUCTIONS:
            raise DSLSyntaxError(
                f"{where}: unknown instruction {head!r} "
                f"(known: {list(_KNOWN_INSTRUCTIONS)})"
            )
        if name in ("polynomial", "equilibrium"):
            if not rest:
                raise DSLSyntaxError(f"{where}: {name} needs an expression")
            symbols = (
                _SYMBOLS_POLYNOMIAL if name == "polynomial" else _SYMBOLS_EQUILIBRIUM
            )
            code = _compile_expression(rest, symbols, where=f"{where} {name}")
            if name == "polynomial":
                polynomials.append(rest)
                poly_codes.append(code)
            else:
                equilibria.append(rest)
                eq_codes.append(code)
        elif name == "stencil":
            stencil = _pairs(rest, where=where, allowed=("dim", "velocity", "velocities"))
            for required in ("dim", "velocities"):
                if required not in stencil:
                    raise DSLSyntaxError(f"{where}: stencil needs {required}=...")
        elif name == "relaxation":
            if not rest:
                raise DSLSyntaxError(f"{where}: relaxation needs a rate list")
            try:
                relaxation = [float(v) for v in rest.split(",") if v.strip()]
            except ValueError as exc:
                raise DSLSyntaxError(
                    f"{where}: relaxation rates must be numbers, got {rest!r}"
                ) from exc
        elif name == "domain":
            domain = _pairs(rest, where=where, allowed=("nx", "ny"))
            for required in ("nx", "ny"):
                if required not in domain:
                    raise DSLSyntaxError(f"{where}: domain needs {required}=...")
        elif name == "boundary":
            for key, value in _pairs(rest, where=where, allowed=_BOUNDARY_KEYS).items():
                if value not in _BOUNDARY_KINDS:
                    raise DSLSyntaxError(
                        f"{where}: boundary {key}={value!r} is not one of "
                        f"{list(_BOUNDARY_KINDS)}"
                    )
                boundary[key] = value
        elif name == "init":
            words = rest.split()
            if not words:
                raise DSLSyntaxError(f"{where}: init needs a kind")
            kind = words[0].lower()
            body = _pairs(
                " ".join(words[1:]),
                where=where,
                allowed=("rho", "ux", "uy", "amp", "mode"),
            )
            init = {"kind": kind, **body}
        elif name == "force":
            force_pairs = _pairs(rest, where=where, allowed=("gx", "gy"))
        elif name == "run":
            run_pairs = _pairs(rest, where=where, allowed=("steps", "tol"))

    for required, present in (
        ("stencil", stencil is not None),
        ("polynomial", bool(polynomials)),
        ("equilibrium", bool(equilibria)),
        ("relaxation", relaxation is not None),
        ("domain", domain is not None),
        ("init", init is not None),
        ("run", run_pairs is not None),
    ):
        if not present:
            raise DSLSyntaxError(f"missing instruction: {required}")

    dim = _as_int(stencil, "dim", "stencil")
    velocities = _parse_velocities(stencil["velocities"], "stencil")
    if len(velocities) != len(set(velocities)):
        raise DSLSemanticError("stencil: duplicated velocity")
    if len(polynomials) != len(equilibria):
        raise DSLSemanticError(
            "polynomial/equilibrium counts differ: "
            f"{len(polynomials)} polynomials vs {len(equilibria)} equilibria"
        )
    if len(relaxation) != len(velocities):
        raise DSLSemanticError(
            f"relaxation has {len(relaxation)} rates but the stencil has "
            f"{len(velocities)} velocities"
        )

    nx = _as_int(domain, "nx", "domain")
    ny = _as_int(domain, "ny", "domain")
    if nx < 3 or ny < 3:
        raise DSLSemanticError(f"domain must be at least 3x3, got {nx}x{ny}")

    if init["kind"] == "uniform":
        init_norm: Dict[str, Any] = {
            "kind": "uniform",
            "rho": _as_float(init, "rho", "init", default=1.0),
            "ux": _as_float(init, "ux", "init", default=0.0),
            "uy": _as_float(init, "uy", "init", default=0.0),
        }
    elif init["kind"] == "taylor_green":
        init_norm = {
            "kind": "taylor_green",
            "rho": _as_float(init, "rho", "init", default=1.0),
            "amp": _as_float(init, "amp", "init"),
            "mode": _as_int(init, "mode", "init", default=1),
        }
    else:
        raise DSLSemanticError(
            f"init kind {init['kind']!r} is not implemented "
            "(supported: uniform, taylor_green)"
        )

    steps = _as_int(run_pairs, "steps", "run")
    if steps < 1:
        raise DSLSemanticError(f"run steps must be >= 1, got {steps}")

    return Program(
        source=text,
        dim=dim,
        velocity=_as_float(stencil, "velocity", "stencil", default=1.0),
        velocities=velocities,
        polynomials=polynomials,
        equilibria=equilibria,
        relaxation=relaxation,
        nx=nx,
        ny=ny,
        boundary={
            "xmin": boundary.get("xmin", "periodic"),
            "xmax": boundary.get("xmax", "periodic"),
            "ymin": boundary.get("ymin", "periodic"),
            "ymax": boundary.get("ymax", "periodic"),
        },
        init=init_norm,
        force=(
            _as_float(force_pairs, "gx", "force", default=0.0),
            _as_float(force_pairs, "gy", "force", default=0.0),
        ),
        steps=steps,
        tol=_as_float(run_pairs, "tol", "run", default=0.0),
        _poly_codes=poly_codes,
        _eq_codes=eq_codes,
    )


# ---------------------------------------------------------------------------
# lowering: instructions -> numeric scheme + invariant self-checks
# ---------------------------------------------------------------------------
_CHECK_TOL = 1e-10


@dataclass
class Scheme:
    """Numeric scheme lowered from a :class:`Program`, plus its check report."""

    velocity: float
    velocities: np.ndarray  # (nq, 2) integer lattice velocities
    weights: np.ndarray  # rest equilibrium == Gauss-Hermite weights
    cs2: float
    M: np.ndarray  # moment matrix M[i, j] = P_i(c_j * LA)
    Minv: np.ndarray
    relaxation: np.ndarray
    tau: float
    nu: float
    polynomials: Tuple[str, ...]
    equilibria: Tuple[str, ...]
    checks: Dict[str, float] = field(default_factory=dict)
    _eq_codes: List[Any] = field(default_factory=list, repr=False)

    @property
    def nq(self) -> int:
        return int(self.velocities.shape[0])

    def equilibrium_moments(self, rho, ux, uy) -> np.ndarray:
        """Equilibrium in *moment* space, ``EQ_i(rho, ux, uy)``."""
        rho = np.asarray(rho, dtype=float)
        ux = np.asarray(ux, dtype=float)
        uy = np.asarray(uy, dtype=float)
        shape = np.broadcast(rho, ux, uy).shape
        env = {
            "rho": np.broadcast_to(rho, shape),
            "ux": np.broadcast_to(ux, shape),
            "uy": np.broadcast_to(uy, shape),
        }
        qeq = np.empty((self.nq,) + shape, dtype=float)
        for i, code in enumerate(self._eq_codes):
            value = _evaluate(code, env)
            qeq[i] = np.broadcast_to(np.asarray(value, dtype=float), shape)
        return qeq

    def equilibrium(self, rho, ux, uy) -> np.ndarray:
        """Population-space equilibrium ``f_eq = M^-1 . EQ(rho, ux, uy)``."""
        rho = np.asarray(rho, dtype=float)
        ux = np.asarray(ux, dtype=float)
        uy = np.asarray(uy, dtype=float)
        shape = np.broadcast(rho, ux, uy).shape
        env = {
            "rho": np.broadcast_to(rho, shape),
            "ux": np.broadcast_to(ux, shape),
            "uy": np.broadcast_to(uy, shape),
        }
        qeq = np.empty((self.nq,) + shape, dtype=float)
        for i, code in enumerate(self._eq_codes):
            value = _evaluate(code, env)
            qeq[i] = np.broadcast_to(np.asarray(value, dtype=float), shape)
        return np.einsum("ji,i...->j...", self.Minv, qeq)

    def macros(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moments of ``f`` (no forcing correction)."""
        rho = f.sum(axis=0)
        safe = np.where(rho > 0.0, rho, 1.0)
        ux = np.einsum("j,j...->...", self.velocities[:, 0].astype(float), f) / safe
        uy = np.einsum("j,j...->...", self.velocities[:, 1].astype(float), f) / safe
        return rho, ux, uy

    def report(self) -> Dict[str, Any]:
        return {
            "velocity": self.velocity,
            "nq": self.nq,
            "weights": self.weights.tolist(),
            "cs2": self.cs2,
            "relaxation": self.relaxation.tolist(),
            "tau": self.tau,
            "nu": self.nu,
            "checks": dict(self.checks),
            "polynomials": list(self.polynomials),
            "equilibria": list(self.equilibria),
        }


def build_scheme(prog: Program) -> Scheme:
    """Lower a program to a numeric scheme and run its invariant checks."""
    if prog.dim != 2:
        raise DSLSemanticError(
            f"stencil dim={prog.dim} is not supported by the 2-D lowering"
        )
    velocities = np.asarray(prog.velocities, dtype=float)
    if not np.allclose(velocities, np.round(velocities)):
        raise DSLSemanticError("stencil velocities must be integers in the 2-D lowering")
    velocities = np.round(velocities).astype(np.int64)
    if not np.any(np.all(velocities == 0, axis=1)):
        raise DSLSemanticError("stencil must contain the zero velocity (0,0)")
    nq = velocities.shape[0]
    la = float(prog.velocity)

    M = np.empty((nq, nq), dtype=float)
    for i, code in enumerate(prog._poly_codes):
        env = {"X": velocities[:, 0] * la, "Y": velocities[:, 1] * la}
        M[i] = np.broadcast_to(np.asarray(_evaluate(code, env), dtype=float), (nq,))
    if not np.all(np.isfinite(M)):
        raise DSLSemanticError("moment matrix contains non-finite entries")
    rank = int(np.linalg.matrix_rank(M))
    if rank < nq:
        raise DSLSemanticError(f"moment matrix is singular (rank {rank} < {nq})")
    Minv = np.linalg.inv(M)

    # equilibrium at rest == the lattice weights
    q0 = np.empty(nq)
    rest_env = {"rho": 1.0, "ux": 0.0, "uy": 0.0}
    for i, code in enumerate(prog._eq_codes):
        value = _evaluate(code, rest_env)
        if not np.all(np.isfinite(value)):
            raise DSLSemanticError(f"equilibrium moment {i} is not finite at rest")
        q0[i] = float(value)
    weights = Minv @ q0
    if not np.all(np.isfinite(weights)):
        raise DSLSemanticError("degenerate weights from the declared equilibrium")
    if np.any(weights < -_CHECK_TOL):
        raise DSLSemanticError(
            f"declared equilibrium gives negative weights {weights.tolist()}"
        )

    cx = velocities[:, 0].astype(float)
    cy = velocities[:, 1].astype(float)
    checks = {
        "sum_w": float(weights.sum()),
        "sum_w_cx": float(weights @ cx),
        "sum_w_cy": float(weights @ cy),
        "sum_w_cx2": float(weights @ cx**2),
        "sum_w_cy2": float(weights @ cy**2),
        "sum_w_cxcy": float(weights @ (cx * cy)),
        "sum_w_cx2cy2": float(weights @ ((cx**2) * (cy**2))),
        "cond_M": float(np.linalg.cond(M)),
    }
    if abs(checks["sum_w"] - 1.0) > 1e-8:
        raise DSLSemanticError(
            "equilibrium does not recover mass: sum(weights) = "
            f"{checks['sum_w']!r} != 1"
        )
    for key in ("sum_w_cx", "sum_w_cy"):
        if abs(checks[key]) > 1e-8:
            raise DSLSemanticError(
                f"equilibrium does not recover momentum at rest: {key} = "
                f"{checks[key]!r}"
            )
    cs2 = checks["sum_w_cx2"]
    if not (0.0 < cs2 < 1.0):
        raise DSLSemanticError(f"implausible sound speed c_s^2 = {cs2!r}")
    checks["cs2_expected"] = la * la / 3.0
    checks["cs2_rel_error"] = abs(cs2 - la * la / 3.0) / (la * la / 3.0)

    relaxation = np.asarray(prog.relaxation, dtype=float)
    if not np.all(np.isfinite(relaxation)):
        raise DSLSemanticError("relaxation rates must be finite")
    for i, s in enumerate(relaxation[:3]):
        if s != 0.0 and not (0.0 < s < 2.0):
            raise DSLSemanticError(
                f"conserved moment {i} has relaxation {s} outside [0, 2)"
            )
    for i, s in enumerate(relaxation[3:], start=3):
        if not (0.0 < s < 2.0):
            raise DSLSemanticError(
                f"non-conserved moment {i} has relaxation {s} outside (0, 2)"
            )

    # shear relaxation = the moment whose polynomial is X^2 - Y^2 when declared,
    # otherwise the first non-conserved moment (pylbm convention s = 1/(0.5+sigma))
    shear_index = 3
    for i, poly in enumerate(prog.polynomials):
        if poly.replace(" ", "").replace("^", "**") == "X**2-Y**2":
            shear_index = i
            break
    if shear_index >= nq:
        raise DSLSemanticError("no shear moment available to define the viscosity")
    s_shear = float(relaxation[shear_index])
    tau = 1.0 / s_shear
    nu = (tau - 0.5) / 3.0
    if nu <= 0.0:
        raise DSLSemanticError(
            f"declared shear relaxation s={s_shear} gives nu={nu} (must be > 0)"
        )
    checks["shear_index"] = float(shear_index)
    checks["s_shear"] = s_shear

    scheme = Scheme(
        velocity=la,
        velocities=velocities,
        weights=weights,
        cs2=cs2,
        M=M,
        Minv=Minv,
        relaxation=relaxation,
        tau=tau,
        nu=nu,
        polynomials=tuple(prog.polynomials),
        equilibria=tuple(prog.equilibria),
        checks=checks,
        _eq_codes=prog._eq_codes,
    )

    # equilibrium consistency at a representative macroscopic state
    rho, ux, uy = 1.037, 0.041, -0.023
    feq = scheme.equilibrium(rho, ux, uy)
    checks["feq_mass_rel_error"] = float(abs(feq.sum() - rho) / rho)
    checks["feq_momentum_x_rel_error"] = float(abs(feq @ cx - rho * ux) / (rho * abs(ux)))
    checks["feq_momentum_y_rel_error"] = float(abs(feq @ cy - rho * uy) / (rho * abs(uy)))
    for key in (
        "feq_mass_rel_error",
        "feq_momentum_x_rel_error",
        "feq_momentum_y_rel_error",
    ):
        if checks[key] > 1e-10:
            raise DSLSemanticError(
                f"declared equilibrium does not recover {key} = {checks[key]:.3e}"
            )
    return scheme


# ---------------------------------------------------------------------------
# initial conditions and reference helpers
# ---------------------------------------------------------------------------
def taylor_green_velocity(
    nx: int, ny: int, *, amp: float, mode: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Divergence-free Taylor-Green field.

    ``u_x = A cos(k x) sin(k y)``, ``u_y = -A sin(k x) cos(k y)``,
    ``k = 2 pi mode / n`` in lattice units with ``x = i``, ``y = j``.
    """
    kx = 2.0 * np.pi * mode / nx
    ky = 2.0 * np.pi * mode / ny
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    ux = amp * np.cos(kx * x) * np.sin(ky * y)
    uy = -amp * np.sin(kx * x) * np.cos(ky * y)
    return ux, uy


def decay_rate_from_history(
    history: Sequence[Mapping[str, float]], k: float
) -> Tuple[float, float]:
    """Least-squares fit of ``ln A(t) = ln A0 - 2 nu k^2 t``.

    ``A`` is the Taylor-Green amplitude reconstructed from the recorded kinetic
    energy as ``A = sqrt(2 KE)``.  Returns ``(nu_eff, r2)``.
    """
    t = np.asarray([h["step"] for h in history], dtype=float)
    ke = np.asarray([h["kinetic_energy"] for h in history], dtype=float)
    amp = np.sqrt(2.0 * ke)
    keep = amp > 1e-12
    t, amp = t[keep], amp[keep]
    if t.size < 2:
        raise DSLSemanticError("need at least two samples to fit a decay rate")
    slope, intercept = np.polyfit(t, np.log(amp), 1)
    pred = slope * t + intercept
    resid = np.log(amp) - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((np.log(amp) - np.log(amp).mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return float(-slope / (2.0 * k * k)), r2


def poiseuille_error(
    ux: np.ndarray, ny: int, *, nu: float, body_force: float
) -> Dict[str, float]:
    """Relative error of a channel profile against the analytic parabola.

    Geometry of the channel lowering: rows ``y = 0`` and ``y = ny - 1`` are
    solid, so the closed links -- and therefore the half-way bounce-back wall
    planes -- sit at ``y = 1/2`` and ``y = ny - 3/2`` and the channel height is
    ``H = ny - 2``.  The steady solution is

        ``u(y) = g (y - 1/2) (ny - 3/2 - y) / (2 nu)``

    on the fluid rows ``y = 1 .. ny - 2`` (measured numerically in
    ``artifacts/xval_instruction_dsl/physics.json``: fitted wall positions
    0.513 / 20.487 for ``ny = 22``, i.e. ``H = ny - 2``).
    """
    y = np.arange(1, ny - 1, dtype=float)
    top = ny - 1.5
    analytic = body_force * (y - 0.5) * (top - y) / (2.0 * nu)
    profile = np.asarray(ux).mean(axis=0)[1 : ny - 1]
    scale = float(np.max(np.abs(analytic)))
    err = profile - analytic
    return {
        "u_max_analytic": scale,
        "u_max_numeric": float(np.max(np.abs(profile))),
        "max_abs_error": float(np.max(np.abs(err))),
        "max_rel_error": float(np.max(np.abs(err)) / scale),
        "l2_rel_error": float(np.linalg.norm(err) / np.linalg.norm(analytic)),
    }


# ---------------------------------------------------------------------------
# execution plan
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Outcome of a DSL run: fields, history, scheme report, provenance."""

    program: Program
    scheme: Scheme
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    steps: int
    history: List[Dict[str, float]] = field(default_factory=list)
    wall_time_s: float = 0.0
    converged: bool = False

    @property
    def kinetic_energy(self) -> float:
        return float(0.5 * np.mean(self.ux**2 + self.uy**2))

    def summary(self) -> Dict[str, Any]:
        return {
            "digest": self.program.digest,
            "steps": self.steps,
            "converged": self.converged,
            "kinetic_energy": self.kinetic_energy,
            "u_max": float(np.max(np.hypot(self.ux, self.uy))),
            "nu": self.scheme.nu,
            "tau": self.scheme.tau,
            "cs2": self.scheme.cs2,
            "wall_time_s": self.wall_time_s,
        }


def _require_supported_layout(prog: Program) -> str:
    b = prog.boundary
    if (
        b["xmin"] == "periodic"
        and b["xmax"] == "periodic"
        and b["ymin"] == "periodic"
        and b["ymax"] == "periodic"
    ):
        return "periodic"
    if (
        b["xmin"] == "periodic"
        and b["xmax"] == "periodic"
        and b["ymin"] == "bounce_back"
        and b["ymax"] == "bounce_back"
    ):
        return "channel"
    raise DSLSemanticError(
        "unsupported boundary mix "
        f"{b} -- the 2-D lowering supports all-periodic or "
        "(x periodic + y bounce_back)"
    )


def _stream_periodic(f: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    for j in range(velocities.shape[0]):
        cx, cy = int(velocities[j, 0]), int(velocities[j, 1])
        out[j] = np.roll(f[j], (cx, cy), axis=(0, 1))
    return out


def _stream_channel(f: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Periodic in ``x``, half-way bounce-back on the solid rows ``y = 0, ny-1``."""
    nq = velocities.shape[0]
    ny = f.shape[2]
    out = np.zeros_like(f)
    for j in range(nq):
        cx, cy = int(velocities[j, 0]), int(velocities[j, 1])
        a = np.roll(f[j], cx, axis=0)  # periodic in x
        if cy == 0:
            out[j, :, 1 : ny - 1] = a[:, 1 : ny - 1]
        else:
            out[j, :, 1 : ny - 1] = a[:, 1 - cy : ny - 1 - cy]
    index = {tuple(int(v) for v in row): i for i, row in enumerate(velocities)}
    for j in range(nq):
        cx, cy = int(velocities[j, 0]), int(velocities[j, 1])
        if cy == 0 or (cx, cy) not in index:
            continue
        opp = index[(-cx, -cy)]
        if cy < 0:  # leaving the first fluid row into the wall at y = 1/2
            out[opp, :, 1] += f[j, :, 1]
        else:  # leaving the last fluid row into the wall at y = ny - 1/2
            out[opp, :, ny - 2] += f[j, :, ny - 2]
    return out


def run(
    prog: Program | str,
    *,
    scheme: Scheme | None = None,
    sample_every: int = 1,
    max_steps: int | None = None,
) -> RunResult:
    """Build the scheme (unless given) and advance the declared problem.

    Forcing is applied as **moment increments split by half steps** (Strang /
    exact-difference form, cf. pylbm's ``source`` instructions which add
    ``dMoments``): the momentum moments receive ``g/2`` before and after the
    relaxation, the equilibrium is evaluated at the midpoint velocity
    ``u = (sum_i f_i c_i + g/2) / rho`` and the conserved moments are *not*
    relaxed.  One step therefore adds exactly ``g`` of momentum, independent of
    the relaxation rates, and the scheme is second order in time.
    """
    if isinstance(prog, str):
        prog = parse(prog)
    if scheme is None:
        scheme = build_scheme(prog)
    layout = _require_supported_layout(prog)

    nx, ny = prog.nx, prog.ny
    C = scheme.velocities
    steps = prog.steps if max_steps is None else min(prog.steps, int(max_steps))
    gx, gy = prog.force
    forced = gx != 0.0 or gy != 0.0

    # -- initial state -----------------------------------------------------
    if prog.init["kind"] == "uniform":
        rho0 = np.full((nx, ny), float(prog.init["rho"]))
        ux0 = np.full((nx, ny), float(prog.init["ux"]))
        uy0 = np.full((nx, ny), float(prog.init["uy"]))
    else:
        rho0 = np.full((nx, ny), float(prog.init["rho"]))
        ux0, uy0 = taylor_green_velocity(
            nx, ny, amp=float(prog.init["amp"]), mode=int(prog.init["mode"])
        )
    f = scheme.equilibrium(rho0, ux0, uy0)
    fluid = np.ones((nx, ny), dtype=bool)
    if layout == "channel":
        fluid[:, 0] = False
        fluid[:, ny - 1] = False
        f = np.where(fluid[None, :, :], f, 0.0)

    u_ref = float(np.max(np.hypot(ux0, uy0))) if prog.init["kind"] != "uniform" else 0.0

    t_start = time.perf_counter()
    history: List[Dict[str, float]] = []
    converged = False
    s = scheme.relaxation[:, None, None]

    def _record(step: int):
        rho, ux, uy = scheme.macros(f)
        ke = 0.5 * float(np.mean(ux**2 + uy**2))
        u_max = float(np.max(np.hypot(ux, uy)))
        history.append(
            {
                "step": float(step),
                "kinetic_energy": ke,
                "u_max": u_max,
                "rho_mean": float(np.mean(rho[fluid])),
            }
        )
        return rho, ux, uy, u_max

    rho, ux, uy, u_max = _record(0)
    if u_max / np.sqrt(scheme.cs2) > 0.3:
        raise DSLSemanticError(
            f"mach guard: u_max/c_s = {u_max / np.sqrt(scheme.cs2):.3f} > 0.3"
        )
    prev_speed = np.hypot(ux, uy)
    u_ref = max(u_ref, 1e-30)

    for step in range(1, steps + 1):
        q = np.einsum("ij,j...->i...", scheme.M, f)
        rho = q[0]
        rho_safe = np.where(rho > 0.0, rho, 1.0)
        if forced:
            q[1] = q[1] + 0.5 * gx
            q[2] = q[2] + 0.5 * gy
        ux = q[1] / rho_safe
        uy = q[2] / rho_safe
        qeq = scheme.equilibrium_moments(rho, ux, uy)
        q = q + s * (qeq - q)
        if forced:
            q[1] = q[1] + 0.5 * gx
            q[2] = q[2] + 0.5 * gy
        f = np.einsum("ji,i...->j...", scheme.Minv, q)
        if layout == "periodic":
            f = _stream_periodic(f, C)
        else:
            f = _stream_channel(f, C)
            f = np.where(fluid[None, :, :], f, 0.0)

        if step % max(1, sample_every) == 0 or step == steps:
            rho, ux, uy, u_max = _record(step)
            if u_max / np.sqrt(scheme.cs2) > 0.5:
                raise DSLSemanticError(
                    f"diverged at step {step}: u_max/c_s = "
                    f"{u_max / np.sqrt(scheme.cs2):.3f}"
                )
            if prog.tol > 0.0:
                speed = np.hypot(ux, uy)
                change = float(np.max(np.abs(speed - prev_speed)) / u_ref)
                prev_speed = speed
                if change < prog.tol:
                    converged = True
                    break

    wall = time.perf_counter() - t_start
    rho, ux, uy = scheme.macros(f)
    return RunResult(
        program=prog,
        scheme=scheme,
        rho=rho,
        ux=ux,
        uy=uy,
        steps=int(history[-1]["step"]) if history else 0,
        history=history,
        wall_time_s=wall,
        converged=converged,
    )
