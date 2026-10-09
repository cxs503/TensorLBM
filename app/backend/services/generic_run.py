"""Generic simulation engine — the platform's single common-module case runner.

This module implements the *generic-run* fusion called for by
``PLATFORM_ANALYSIS.md`` §4.2 on top of the benchmark compile-route
standard (PR #180): one data-driven execution path through which every
supported case runs, composed exclusively of the package's common
modules —

* ``tensorlbm.solver``        — collide_bgk / collide_mrt / stream
* ``tensorlbm.d2q9``          — equilibrium / macroscopic / OPPOSITE
* ``tensorlbm.boundaries``    — wall masks, Zou-He in/outlet, far-field,
                                obstacle bounce-back, momentum-exchange
                                forces, sponge layer (2-D)
* ``tensorlbm.lid_driven_cavity`` — moving-lid BC, Ghia references
* ``tensorlbm.postprocess``   — detect_strouhal

— with the *whole-step* chain routed through ``benchmarks/compile_route``
(→ ``tensorlbm.compile_utils``) exactly like the verified benchmark
suite: the step index and every host-side monitoring sync stay outside
the compiled domain, cudagraph-class modes are rejected by the shared
validator, and the ``"eager"`` spelling maps to the canonical ``None``.

No case gets bespoke kernel code here.  A case is *data*: a registry
entry declaring its initial condition, an ordered chain of common-module
BC primitives, default physics/grid parameters, and a metrics hook.
Adding a case means adding one registry entry that composes existing
primitives — the driver below never branches on the case name.

The three tiny BC primitives the verified benchmarks define inline
(pre-streaming half-way bounce-back, its moving-wall extension, and the
Zou-He pressure inlet mirroring
:func:`tensorlbm.boundaries.zou_he_outlet_pressure`) are provided here
once as *generic, parametrised* helpers so the platform and the
benchmarks share the same semantics.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import job_manager

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Benchmark-identical compile routing
# ---------------------------------------------------------------------------
#
# The single adapter used by every ``benchmarks/verified/*/run.py`` is
# ``benchmarks/compile_route.py``.  This engine routes its whole-step
# chains through that *very module* (loaded by file path — ``benchmarks/``
# is not a package), so the platform and the benchmark suite take
# byte-identical routing decisions and print the same audit banner.
# When the benchmarks tree is absent (e.g. a packaged install) a minimal
# fallback with identical semantics routes directly through the shared
# ``tensorlbm.compile_utils``.

_COMPILE_ROUTE: Any = None


class _DirectCompileRoute:
    """Fallback adapter mirroring ``benchmarks/compile_route`` semantics.

    Only used when ``<repo>/benchmarks/compile_route.py`` cannot be
    loaded; the normal path reuses the benchmark module itself.
    """

    EAGER_CLI_SPELLINGS = frozenset({"eager", ""})

    def normalize_compile_mode(self, mode: str | None) -> str | None:
        from tensorlbm.compile_utils import validate_compile_mode

        if isinstance(mode, str) and mode.lower() in self.EAGER_CLI_SPELLINGS:
            mode = None
        validate_compile_mode(mode)
        return mode

    def route_step(
        self,
        step_fn: Callable[..., Any],
        mode: str | None = "default",
        *,
        name: str = "generic",
        warmup_hint: str | None = None,
        quiet: bool = False,
    ) -> Callable[..., Any]:
        from tensorlbm.compile_utils import compile_step

        canonical = self.normalize_compile_mode(mode)
        wrapped = compile_step(
            step_fn, canonical,
            warmup_hint=warmup_hint
            or f"generic-run {name!r}: one whole-step graph per grid shape",
        )
        if not quiet:
            routed = ("eager (compile_step passthrough)" if canonical is None
                      else f"torch.compile(mode={canonical!r})")
            print(f"[compile_route] {name}: mode={mode!r} -> {routed}", flush=True)
        return wrapped


def get_compile_route() -> Any:  # noqa: ANN401 — module-or-adapter by design
    """Return the benchmark ``compile_route`` module (cached).

    Falls back to a direct ``tensorlbm.compile_utils`` adapter with
    identical semantics when the benchmarks tree is not present.
    """
    global _COMPILE_ROUTE
    if _COMPILE_ROUTE is not None:
        return _COMPILE_ROUTE
    path = _REPO_ROOT / "benchmarks" / "compile_route.py"
    if path.is_file():
        spec = importlib.util.spec_from_file_location(
            "tensorlbm_platform_compile_route", path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("tensorlbm_platform_compile_route", mod)
        spec.loader.exec_module(mod)
        _COMPILE_ROUTE = mod
    else:
        _COMPILE_ROUTE = _DirectCompileRoute()
    return _COMPILE_ROUTE


def normalize_compile_mode(mode: str | None) -> str | None:
    """Validate *mode* through the shared compile-route normalisation.

    Raises the shared ``ValueError`` (cudagraph structural reason /
    unknown mode) — the router surfaces it as HTTP 422.
    """
    return get_compile_route().normalize_compile_mode(mode)


# ---------------------------------------------------------------------------
# Generic BC primitives (mirrors of the verified benchmark helpers)
# ---------------------------------------------------------------------------

_CS2 = 1.0 / 3.0


def pre_halfway_bounce_back(
    f_post: torch.Tensor, f_pre: torch.Tensor, wall: torch.Tensor
) -> torch.Tensor:
    """Pre-streaming half-way bounce-back on *wall* cells (generic).

    At wall cells the post-collision populations are replaced by the
    reflected pre-collision ones, so no momentum enters the wall rows.
    Identical to the helper the cavity/poiseuille benchmarks define
    inline (``torch.where(wall, f_pre[OPPOSITE], f)``).
    """
    from tensorlbm.d2q9 import OPPOSITE

    opp = OPPOSITE.to(f_pre.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f_post)


def pre_moving_wall_bounce_back(
    f_post: torch.Tensor,
    f_pre: torch.Tensor,
    wall: torch.Tensor,
    u_wall: torch.Tensor,
) -> torch.Tensor:
    """Half-way bounce-back with moving-wall momentum injection (generic).

    ``u_wall`` is a per-cell wall-velocity field ``(ny, nx)`` in +x; the
    standard term ``2·w_q·rho·(c_q · u_wall)/cs²`` injects exactly the
    momentum required by the moving wall.  Identical to the couette
    benchmark's inline helper.
    """
    from tensorlbm.d2q9 import OPPOSITE, C, W

    device = f_pre.device
    opp = OPPOSITE.to(device)
    c = C.to(device)
    w = W.to(device)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f_post)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / _CS2
    return f_new + injection * wall.unsqueeze(0)


def zou_he_pressure_inlet(f: torch.Tensor, rho_in: float) -> torch.Tensor:
    """Zou-He pressure (density) inlet at the left column x=0 (generic).

    Prescribes ``rho = rho_in`` and ``uy = 0`` and reconstructs the
    unknown in-flowing populations analytically — the mirror image of the
    common :func:`tensorlbm.boundaries.zou_he_outlet_pressure`.  Identical
    to the poiseuille benchmark's inline helper (Zou & He 1997).
    """
    f0, f2, f3, f4, f6, f7 = (
        f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0], f[6, :, 0], f[7, :, 0],
    )
    rho = torch.full_like(f0, rho_in)
    ux = 1.0 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho
    f_new = f.clone()
    f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux
    f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    return f_new


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

PreBC = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PostBC = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class CaseAssembly:
    """Everything the generic driver needs to run one case.

    The driver composes ``collide → pre_bcs → stream → post_bcs``
    (or ``stream → collide`` for ``stream_first`` periodic families),
    optionally extracting per-step obstacle forces between streaming and
    the post-BCs (post-stream, pre-bounce-back — the Ladd convention).
    """

    case: str
    family: str
    lattice: str
    description: str
    grid: dict[str, int]
    physics_resolved: dict[str, Any]
    f0: torch.Tensor
    collide: Callable[[torch.Tensor], torch.Tensor]
    pre_bcs: list[PreBC] = field(default_factory=list)
    post_bcs: list[PostBC] = field(default_factory=list)
    stream_first: bool = False
    force_mask: torch.Tensor | None = None
    collision: str = "bgk"
    metrics: Callable[..., dict[str, Any]] | None = None
    monitor: Callable[[torch.Tensor], dict[str, Any]] | None = None
    default_steps: int = 1000
    modules_used: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParamSpec:
    """Declarative parameter spec for one case (validation + discovery)."""

    grid: dict[str, tuple[int, int, str]]       # name -> (default, minimum, desc)
    physics: dict[str, tuple[float, str]]       # name -> (default, desc)
    collision: str                              # default when request says "auto"
    steps: int


class ParamError(ValueError):
    """Invalid case/parameter combination (surfaced as HTTP 422)."""


# ── cavity (lid-driven, Ghia family) ────────────────────────────────────────


def _build_cavity(
    grid: dict[str, int], physics: dict[str, float], collision: str, device: torch.device,
) -> CaseAssembly:
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.lid_driven_cavity import make_cavity_wall_mask
    from tensorlbm.solver import collide_bgk, collide_mrt

    nx = grid.get("nx", 64)
    ny = nx
    re = physics.get("Re", 100.0)
    u_lid = physics.get("u_lid", 0.06)
    tau = 3.0 * u_lid * nx / re + 0.5             # benchmark convention

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f0 = equilibrium(rho0, u0, u0)

    wall = make_cavity_wall_mask(ny, nx, device, include_top=False)
    collide_fn = collide_mrt if collision == "mrt" else collide_bgk
    collide = lambda f: collide_fn(f, tau=tau)  # noqa: E731

    assembly = CaseAssembly(
        case="cavity",
        family="lid-driven cavity (Ghia 1982)",
        lattice="d2q9",
        description=(
            "Lid-driven square cavity; three stationary walls via pre-streaming "
            "half-way bounce-back, moving lid via Zou-He."
        ),
        grid={"nx": nx, "ny": ny},
        physics_resolved={"Re": re, "u_lid": u_lid, "tau": tau, "nu": (tau - 0.5) / 3.0},
        f0=f0,
        collide=collide,
        pre_bcs=[lambda f_post, f_pre: pre_halfway_bounce_back(f_post, f_pre, wall)],
        post_bcs=[lambda f: _zou_he_moving_lid(f, u_lid)],
        collision=collision,
        default_steps=2000,
        modules_used=[
            "tensorlbm.solver.collide_bgk/collide_mrt",
            "tensorlbm.solver.stream",
            "tensorlbm.d2q9.equilibrium/macroscopic",
            "tensorlbm.lid_driven_cavity.make_cavity_wall_mask",
            "tensorlbm.lid_driven_cavity.zou_he_moving_lid",
            "tensorlbm.lid_driven_cavity.compare_ghia",
        ],
        context={"nx": nx, "ny": ny, "re": re, "u_lid": u_lid, "wall": wall},
    )
    assembly.metrics = _cavity_metrics
    assembly.monitor = _field_monitor
    return assembly


def _zou_he_moving_lid(f: torch.Tensor, u_lid: float) -> torch.Tensor:
    from tensorlbm.lid_driven_cavity import zou_he_moving_lid

    return zou_he_moving_lid(f, u_lid)


def _cavity_metrics(f: torch.Tensor, ctx: dict[str, Any], aux: dict[str, Any]) -> dict[str, Any]:
    """Centreline metrics — identical formulas to benchmarks cavity/re100."""
    from tensorlbm.d2q9 import macroscopic
    from tensorlbm.lid_driven_cavity import GHIA_RE100, GHIA_RE400, GHIA_RE1000, compare_ghia

    nx, ny = ctx["nx"], ctx["ny"]
    u_lid = ctx["u_lid"]
    wall = ctx["wall"]

    _rho, ux, uy = macroscopic(f)
    ux_w = ux.masked_fill(wall, 0.0)
    uy_w = uy.masked_fill(wall, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid

    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)
    u_cl = ux_np[:, nx // 2]
    v_cl = uy_np[ny // 2, :]

    metrics: dict[str, Any] = {
        "u_mid": float(np.interp(0.5, y_pos, u_cl)),
        "u_bot": float(np.interp(0.0625, y_pos, u_cl)),
        "v_mid": float(np.interp(0.5, x_pos, v_cl)),
    }

    # Ghia comparison through the common module when a reference exists.
    re_int = int(round(ctx["re"]))
    ref = {100: GHIA_RE100, 400: GHIA_RE400, 1000: GHIA_RE1000}.get(re_int)
    if ref is not None:
        metrics["ghia"] = compare_ghia(ux, uy, u_lid, ref)
    return metrics


def _field_monitor(f: torch.Tensor) -> dict[str, Any]:
    """Generic field monitor (host sync lives in the eager driver, by design)."""
    from tensorlbm.d2q9 import macroscopic

    _rho, ux, uy = macroscopic(f)
    speed = (ux * ux + uy * uy).sqrt()
    return {"max_abs_u": float(speed.max().item())}


# ── poiseuille (pressure-driven channel) ────────────────────────────────────


def _build_poiseuille(
    grid: dict[str, int], physics: dict[str, float], collision: str, device: torch.device,
) -> CaseAssembly:
    from tensorlbm.boundaries import make_channel_wall_mask, zou_he_outlet_pressure
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk, collide_mrt

    H = grid.get("H", 20)
    ny = H + 2
    nx = grid.get("nx", 3 * H)
    tau = physics.get("tau", 0.8)
    u_max_target = physics.get("u_max", 0.04)

    nu = (tau - 0.5) / 3.0
    delta_rho = 24.0 * nu * u_max_target / (_CS2 * H)
    rho_in = 1.0 + delta_rho / 2.0
    rho_out = 1.0 - delta_rho / 2.0
    u_max_ana = delta_rho * _CS2 * H * H / (8.0 * nu * nx)

    # Initial condition: rest + linear density ramp (benchmark convention)
    xx = torch.arange(nx, device=device, dtype=torch.float32)
    rho0 = (rho_out + (rho_in - rho_out) * (1.0 - xx / (nx - 1))).view(1, nx).expand(ny, nx)
    f0 = equilibrium(
        rho0.contiguous(),
        torch.zeros((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
    )

    wall = make_channel_wall_mask(
        ny, nx, torch.zeros((ny, nx), dtype=torch.bool, device=device), device
    )
    collide_fn = collide_mrt if collision == "mrt" else collide_bgk
    collide = lambda f: collide_fn(f, tau)  # noqa: E731

    assembly = CaseAssembly(
        case="poiseuille",
        family="pressure-driven channel (analytic parabola)",
        lattice="d2q9",
        description=(
            "2-D channel driven by a Zou-He pressure inlet/outlet pair; walls via "
            "pre-streaming half-way bounce-back."
        ),
        grid={"nx": nx, "ny": ny, "H": H},
        physics_resolved={
            "tau": tau, "nu": nu, "rho_in": rho_in, "rho_out": rho_out,
            "delta_rho": delta_rho, "u_max_ana": u_max_ana,
            "Re": (2.0 / 3.0) * u_max_ana * H / nu,
        },
        f0=f0,
        collide=collide,
        pre_bcs=[lambda f_post, f_pre: pre_halfway_bounce_back(f_post, f_pre, wall)],
        post_bcs=[
            lambda f: zou_he_pressure_inlet(f, rho_in),
            lambda f: zou_he_outlet_pressure(f, rho_out),
        ],
        collision=collision,
        default_steps=6000,
        modules_used=[
            "tensorlbm.solver.collide_bgk/collide_mrt",
            "tensorlbm.solver.stream",
            "tensorlbm.d2q9.equilibrium/macroscopic",
            "tensorlbm.boundaries.make_channel_wall_mask",
            "tensorlbm.boundaries.zou_he_outlet_pressure",
        ],
        context={"nx": nx, "ny": ny, "H": H, "u_max_ana": u_max_ana},
    )
    assembly.metrics = _poiseuille_metrics
    assembly.monitor = _field_monitor
    return assembly


def _poiseuille_metrics(
    f: torch.Tensor, ctx: dict[str, Any], aux: dict[str, Any]
) -> dict[str, Any]:
    """Final-state profile error vs the analytic parabola (benchmark formulas)."""
    from tensorlbm.d2q9 import macroscopic

    nx, ny, H = ctx["nx"], ctx["ny"], ctx["H"]
    u_max_ana = ctx["u_max_ana"]
    col = nx // 2

    _rho, ux, _uy = macroscopic(f)
    u_num = ux[:, col].detach().cpu().numpy()[1 : ny - 1]
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = 4.0 * u_max_ana * (y_phys / H) * (1.0 - y_phys / H)

    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    imax = int(np.argmax(u_num))
    u_max_ana_at_row = 4.0 * u_max_ana * (y_phys[imax] / H) * (1.0 - y_phys[imax] / H)
    u_max_err = abs(float(u_num[imax]) - u_max_ana_at_row) / u_max_ana_at_row * 100.0
    mask = u_ana > 0.2 * u_max_ana
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)

    return {
        "l2_rel_err": l2_rel,
        "u_max_num": float(u_num[imax]),
        "u_max_ana": u_max_ana,
        "u_max_err_pct": u_max_err,
        "max_rel_err_central_pct": max_rel,
    }


# ── couette (moving-wall channel) ───────────────────────────────────────────


def _build_couette(
    grid: dict[str, int], physics: dict[str, float], collision: str, device: torch.device,
) -> CaseAssembly:
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk, collide_mrt

    H = grid.get("H", 20)
    ny = H + 2
    nx = grid.get("nx", H)
    tau = physics.get("tau", 0.8)
    U0 = physics.get("U0", 0.05)

    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    u_wall = torch.zeros((ny, nx), device=device)
    u_wall[-1, :] = U0

    # Initial condition: equilibrium at the analytic linear ramp (benchmark)
    yy = torch.arange(ny, device=device, dtype=torch.float32)
    u0 = torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0).view(ny, 1).expand(ny, nx)
    f0 = equilibrium(
        torch.ones((ny, nx), device=device),
        u0.contiguous(),
        torch.zeros((ny, nx), device=device),
    )

    collide_fn = collide_mrt if collision == "mrt" else collide_bgk
    collide = lambda f: collide_fn(f, tau)  # noqa: E731

    assembly = CaseAssembly(
        case="couette",
        family="Couette channel (analytic linear profile)",
        lattice="d2q9",
        description=(
            "Channel with stationary bottom wall and moving top wall (pre-streaming "
            "moving-wall bounce-back); periodic in x."
        ),
        grid={"nx": nx, "ny": ny, "H": H},
        physics_resolved={"tau": tau, "nu": nu, "U0": U0, "Re": U0 * H / nu},
        f0=f0,
        collide=collide,
        pre_bcs=[
            lambda f_post, f_pre: pre_moving_wall_bounce_back(f_post, f_pre, wall, u_wall)
        ],
        collision=collision,
        default_steps=6000,
        modules_used=[
            "tensorlbm.solver.collide_bgk/collide_mrt",
            "tensorlbm.solver.stream",
            "tensorlbm.d2q9.equilibrium/macroscopic",
        ],
        context={"nx": nx, "ny": ny, "H": H, "U0": U0},
    )
    assembly.metrics = _couette_metrics
    assembly.monitor = _field_monitor
    return assembly


def _couette_metrics(f: torch.Tensor, ctx: dict[str, Any], aux: dict[str, Any]) -> dict[str, Any]:
    from tensorlbm.d2q9 import macroscopic

    ny, H, U0 = ctx["ny"], ctx["H"], ctx["U0"]
    _rho, ux, _uy = macroscopic(f)

    # Mean profile over x (periodic) on the fluid rows
    u_prof = ux[1 : ny - 1, :].mean(dim=1).detach().cpu().numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * y_phys / H

    l2_rel = float(np.linalg.norm(u_prof - u_ana) / np.linalg.norm(u_ana))
    return {
        "l2_rel_err": l2_rel,
        "u_top_num": float(u_prof[-1]),
        "u_top_ana": float(u_ana[-1]),
        "u_top_err_pct": abs(float(u_prof[-1]) - float(u_ana[-1])) / U0 * 100.0,
    }


# ── shear_wave (periodic analytic decay) ────────────────────────────────────


def _build_shear_wave(
    grid: dict[str, int], physics: dict[str, float], collision: str, device: torch.device,
) -> CaseAssembly:
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk

    n = grid.get("n", 64)
    tau = physics.get("tau", 0.8)
    u0 = physics.get("u0", 0.05)

    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / float(n)

    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * y)
    uy0 = torch.zeros_like(ux0)
    f0 = equilibrium(torch.ones_like(ux0), ux0, uy0)

    assembly = CaseAssembly(
        case="shear_wave",
        family="shear-wave decay (analytic exp decay)",
        lattice="d2q9",
        description=(
            "Periodic shear wave u = u0·sin(ky) decaying at rate νk²; fully "
            "periodic, no boundary conditions."
        ),
        grid={"n": n},
        physics_resolved={
            "tau": tau, "nu": nu, "k": k, "u0": u0, "gamma_theory": nu * k * k,
        },
        f0=f0,
        collide=lambda f: collide_bgk(f, tau),
        stream_first=True,
        collision="bgk",
        default_steps=3000,
        modules_used=[
            "tensorlbm.solver.collide_bgk",
            "tensorlbm.solver.stream",
            "tensorlbm.d2q9.equilibrium/macroscopic",
        ],
        context={"n": n, "u0": u0, "nu": nu, "k": k, "e0": u0 * u0 / 4.0},
    )
    assembly.metrics = _decay_metrics
    assembly.monitor = _energy_monitor
    return assembly


def _energy_monitor(f: torch.Tensor) -> dict[str, Any]:
    from tensorlbm.d2q9 import macroscopic

    _rho, ux, uy = macroscopic(f)
    e = float((0.5 * (ux * ux + uy * uy)).mean().item())
    umax = float((ux * ux + uy * uy).sqrt().max().item())
    return {"energy": e, "max_abs_u": umax}


def _decay_metrics(f: torch.Tensor, ctx: dict[str, Any], aux: dict[str, Any]) -> dict[str, Any]:
    from tensorlbm.d2q9 import macroscopic

    steps = aux["steps"]
    _rho, ux, uy = macroscopic(f)
    e_meas = float((0.5 * (ux * ux + uy * uy)).mean().item())
    e0 = ctx["e0"]
    gamma = ctx["nu"] * ctx["k"] ** 2
    ratio_meas = e_meas / e0
    ratio_theory = math.exp(-2.0 * gamma * steps)
    return {
        "energy_final": e_meas,
        "energy_ratio_meas": ratio_meas,
        "energy_ratio_theory": ratio_theory,
        "decay_err_pct": abs(ratio_meas - ratio_theory) / ratio_theory * 100.0,
        "gamma_theory": gamma,
    }


# ── cylinder (external flow, forces + Strouhal) ─────────────────────────────


def _build_cylinder(
    grid: dict[str, int], physics: dict[str, float], collision: str, device: torch.device,
) -> CaseAssembly:
    from tensorlbm.boundaries import cylinder_mask, far_field_bc_2d, make_sponge_strength
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_mrt

    D = grid.get("D", 12)
    domain_d = grid.get("domain_D", 20)
    cyl_x_d = grid.get("cyl_x_D", 5)
    sponge_d = grid.get("sponge_D", 5)
    re = physics.get("Re", 100.0)
    u_in = physics.get("u_in", 0.05)
    sponge_alpha = physics.get("sponge_alpha", 10.0)

    nx = int(domain_d * D)
    ny = nx
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    mask = cylinder_mask(nx, ny, cyl_x_d * D, ny / 2.0, D / 2.0, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - sponge_d * D), int(sponge_d * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + sponge_alpha * sigma)   # τ_eff = τ·(1+α·σ) — common-module sponge

    rho0 = torch.ones((ny, nx), device=device)
    f0 = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    assembly = CaseAssembly(
        case="cylinder",
        family="cylinder free-stream (Braza 1986)",
        lattice="d2q9",
        description=(
            "Circular cylinder in free stream: far-field Dirichlet BC + obstacle "
            "bounce-back, downstream sponge via tau_field, Ladd momentum-exchange "
            "forces, Strouhal from the lift history."
        ),
        grid={"nx": nx, "ny": ny, "D": D},
        physics_resolved={
            "Re": re, "u_in": u_in, "nu": nu, "tau": tau, "sponge_alpha": sponge_alpha,
        },
        f0=f0,
        collide=lambda f: collide_mrt(f, tau, tau_field=tau_field),
        post_bcs=[lambda f: far_field_bc_2d(f, u_in, mask)],
        force_mask=mask,
        collision="mrt",
        default_steps=4000,
        modules_used=[
            "tensorlbm.solver.collide_mrt",
            "tensorlbm.solver.stream",
            "tensorlbm.d2q9.equilibrium/macroscopic",
            "tensorlbm.boundaries.cylinder_mask",
            "tensorlbm.boundaries.far_field_bc_2d",
            "tensorlbm.boundaries.make_sponge_strength",
            "tensorlbm.boundaries.compute_obstacle_forces",
            "tensorlbm.postprocess.detect_strouhal",
        ],
        context={"nx": nx, "ny": ny, "D": D, "u_in": u_in},
    )
    assembly.metrics = _cylinder_metrics
    assembly.monitor = _field_monitor
    return assembly


def _cylinder_metrics(f: torch.Tensor, ctx: dict[str, Any], aux: dict[str, Any]) -> dict[str, Any]:
    from tensorlbm.postprocess import detect_strouhal

    D = ctx["D"]
    u_in = ctx["u_in"]
    steps = aux["steps"]
    fx_hist, fy_hist = aux["fx_hist"], aux["fy_hist"]
    fx_np = torch.stack(fx_hist).detach().cpu().numpy().astype(np.float64)
    fy_np = torch.stack(fy_hist).detach().cpu().numpy().astype(np.float64)

    q_dyn = 0.5 * 1.0 * u_in * u_in * D
    cd_series = fx_np / q_dyn
    cl_series = fy_np / q_dyn
    w0 = max(1, int(0.5 * steps))

    cd_mean = float(cd_series[w0:].mean())
    cl_w = cl_series[w0:] - cl_series[w0:].mean()

    st = None
    if cl_w.size >= 256:
        st = detect_strouhal(
            cl_w.tolist(), sample_rate=1.0, u_ref=u_in, length_ref=D, min_cycles=3
        )

    return {
        "cd_mean": cd_mean,
        "cd_ref": 1.35,  # Braza et al. 1986 (Re=100)
        "cl_amplitude": float(np.abs(cl_w).max()) if cl_w.size else 0.0,
        "strouhal": st,
    }


# ---------------------------------------------------------------------------
# Case registry (data, not branches)
# ---------------------------------------------------------------------------

CASE_REGISTRY: dict[str, dict[str, Any]] = {
    "cavity": {
        "builder": _build_cavity,
        "spec": ParamSpec(
            grid={"nx": (64, 8, "Grid nodes per side (square domain)")},
            physics={
                "Re": (100.0, "Reynolds number (Re = u_lid·H/ν)"),
                "u_lid": (0.06, "Lid velocity (lattice units)"),
            },
            collision="mrt",
            steps=2000,
        ),
    },
    "poiseuille": {
        "builder": _build_poiseuille,
        "spec": ParamSpec(
            grid={
                "H": (20, 4, "Channel height in fluid nodes (walls add 2 rows)"),
                "nx": (60, 8, "Channel length (default 3·H)"),
            },
            physics={
                "tau": (0.8, "Relaxation time (ν = (τ−½)/3)"),
                "u_max": (0.04, "Target centreline velocity (sets the driving Δρ)"),
            },
            collision="bgk",
            steps=6000,
        ),
    },
    "couette": {
        "builder": _build_couette,
        "spec": ParamSpec(
            grid={
                "H": (20, 4, "Gap height in fluid nodes (walls add 2 rows)"),
                "nx": (20, 4, "Periodic streamwise length (default H)"),
            },
            physics={
                "tau": (0.8, "Relaxation time (ν = (τ−½)/3)"),
                "U0": (0.05, "Top-wall velocity (lattice units)"),
            },
            collision="bgk",
            steps=6000,
        ),
    },
    "shear_wave": {
        "builder": _build_shear_wave,
        "spec": ParamSpec(
            grid={"n": (64, 8, "Periodic square domain size")},
            physics={
                "tau": (0.8, "Relaxation time (ν = (τ−½)/3)"),
                "u0": (0.05, "Initial shear-wave amplitude"),
            },
            collision="bgk",
            steps=3000,
        ),
    },
    "cylinder": {
        "builder": _build_cylinder,
        "spec": ParamSpec(
            grid={
                "D": (12, 4, "Cylinder diameter in nodes"),
                "domain_D": (20, 8, "Domain size in diameters (square)"),
                "cyl_x_D": (5, 2, "Cylinder centre distance from inlet, in diameters"),
                "sponge_D": (5, 2, "Downstream sponge width in diameters"),
            },
            physics={
                "Re": (100.0, "Reynolds number (Re = u_in·D/ν)"),
                "u_in": (0.05, "Free-stream velocity (lattice units)"),
                "sponge_alpha": (10.0, "Sponge strength (τ_eff = τ·(1+α·σ))"),
            },
            collision="mrt",
            steps=4000,
        ),
    },
}


def list_cases() -> dict[str, Any]:
    """Registry introspection for the ``GET /api/sim/generic/cases`` endpoint."""
    cases: dict[str, Any] = {}
    for name, entry in CASE_REGISTRY.items():
        spec: ParamSpec = entry["spec"]
        cases[name] = {
            "grid": {k: {"default": d, "minimum": m, "description": s}
                     for k, (d, m, s) in spec.grid.items()},
            "physics": {k: {"default": d, "description": s}
                        for k, (d, s) in spec.physics.items()},
            "collision_default": spec.collision,
            "collision_options": ["auto", "bgk", "mrt"],
            "default_steps": spec.steps,
            "compile_modes": [None, "default", "max-autotune-no-cudagraphs"],
        }
    return {
        "count": len(cases),
        "compile_route": "benchmarks/compile_route.py -> tensorlbm.compile_utils",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_request(
    *,
    case: str,
    grid: dict[str, int],
    physics: dict[str, float],
    steps: int,
    collision: str,
    compile_mode: str | None,
) -> tuple[dict[str, int], dict[str, float], int, str, str | None]:
    """Validate a generic-run request against the case registry.

    Returns the normalised ``(grid, physics, steps, collision,
    canonical_compile_mode)`` tuple.  Raises :class:`ParamError` or the
    shared compile-mode ``ValueError`` (both surfaced as HTTP 422 by the
    router; the compile error carries the structural cudagraph reason).
    """
    entry = CASE_REGISTRY.get(case)
    if entry is None:
        raise ParamError(
            f"Unknown case {case!r}. Available: {sorted(CASE_REGISTRY)}"
        )
    spec: ParamSpec = entry["spec"]

    norm_grid: dict[str, int] = {}
    for key, value in grid.items():
        if key not in spec.grid:
            raise ParamError(
                f"Unknown grid parameter {key!r} for case {case!r}. "
                f"Available: {sorted(spec.grid)}"
            )
        _default, minimum, _desc = spec.grid[key]
        if int(value) < minimum:
            raise ParamError(
                f"grid.{key}={value} is below the minimum {minimum} for case {case!r}"
            )
        norm_grid[key] = int(value)

    norm_physics: dict[str, float] = {}
    for key, value in physics.items():
        if key not in spec.physics:
            raise ParamError(
                f"Unknown physics parameter {key!r} for case {case!r}. "
                f"Available: {sorted(spec.physics)}"
            )
        norm_physics[key] = float(value)

    if steps < 0:
        raise ParamError("steps must be >= 0 (0 = case default)")
    resolved_steps = steps if steps > 0 else spec.steps

    if collision == "auto":
        resolved_collision = spec.collision
    elif collision in ("bgk", "mrt"):
        resolved_collision = collision
    else:
        raise ParamError(
            f"collision must be 'auto', 'bgk' or 'mrt'; got {collision!r}"
        )

    canonical_mode = normalize_compile_mode(compile_mode)
    return norm_grid, norm_physics, resolved_steps, resolved_collision, canonical_mode


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_step_fn(asm: CaseAssembly) -> Callable[..., Any]:
    """Compose the whole-step chain from the assembly (pure tensor fn)."""
    from tensorlbm.solver import stream

    collide = asm.collide
    pre_bcs = asm.pre_bcs
    post_bcs = asm.post_bcs
    force_mask = asm.force_mask
    if force_mask is not None:
        from tensorlbm.boundaries import compute_obstacle_forces

    def _step(f: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if asm.stream_first:
            f = collide(stream(f))
        else:
            f_post = collide(f)
            f_pre = f
            for bc in pre_bcs:
                f_post = bc(f_post, f_pre)
            f = stream(f_post)
        fx = fy = None
        if force_mask is not None:
            # Ladd convention: sample post-stream, pre-bounce-back
            fx, fy = compute_obstacle_forces(f, force_mask)
        for bc in post_bcs:
            f = bc(f)
        if fx is not None:
            return f, fx, fy
        return f

    return _step


def run_generic_simulation(
    job: job_manager.Job,
    *,
    case: str,
    grid: dict[str, int],
    physics: dict[str, float],
    steps: int,
    collision: str,
    compile_mode: str | None,
    device: str,
    seed: int,
    monitor_interval: int,
) -> dict[str, Any]:
    """Background job body: run one case through the common-module path.

    The step index, host-side monitoring (``.item()``) and cancellation
    checks stay in this eager driver loop; only the pure whole-step chain
    is routed through the benchmark compile path.
    """
    t0 = time.time()
    job_id = job.job_id
    torch.manual_seed(seed)
    dev = torch.device(device)

    # ── Assemble the case from the registry (data-driven, no branches) ──
    asm: CaseAssembly = CASE_REGISTRY[case]["builder"](grid, physics, collision, dev)
    n_steps = steps
    interval = monitor_interval if monitor_interval > 0 else max(1, min(100, n_steps // 20))

    # ── Route the whole-step chain exactly like the benchmark suite ──
    route = get_compile_route()
    step_fn = route.route_step(
        _build_step_fn(asm), compile_mode, name=f"generic[{case}]",
        quiet=False,  # the banner is the audit trail
    )
    canonical_mode = normalize_compile_mode(compile_mode)
    routed_desc = (
        "eager (compile_step passthrough)" if canonical_mode is None
        else f"torch.compile(mode={canonical_mode!r})"
    )

    job_manager.push_diagnostic(job_id, {
        "kind": "generic_sim_setup",
        "case": case,
        "grid": asm.grid,
        "physics": asm.physics_resolved,
        "collision": collision,
        "compile": {"requested_mode": compile_mode, "routed": routed_desc},
        "steps": n_steps,
        "device": str(dev),
    })

    # ── Driver loop (eager; monitoring + step index stay OUTSIDE compile) ──
    f = asm.f0
    fx_hist: list[torch.Tensor] = []
    fy_hist: list[torch.Tensor] = []
    job_manager.raise_if_cancelled(job_id)

    for step in range(1, n_steps + 1):
        if step % 25 == 0:
            job_manager.raise_if_cancelled(job_id)
        out = step_fn(f)
        if asm.force_mask is not None:
            f, fx, fy = out
            fx_hist.append(fx)
            fy_hist.append(fy)
        else:
            f = out

        if asm.monitor is not None and (step % interval == 0 or step == n_steps):
            data: dict[str, Any] = {
                "kind": "generic_sim_step", "step": step, "total_steps": n_steps,
                "elapsed_s": time.time() - t0,
            }
            data.update(asm.monitor(f))
            if asm.force_mask is not None and fx_hist:
                # host sync on the eager side only
                data["force_x_instant"] = float(fx_hist[-1].item())
            job_manager.push_diagnostic(job_id, data)
            job.logs.append(
                f"[generic:{case}] step={step}/{n_steps} "
                + " ".join(f"{k}={v:.4e}" for k, v in data.items()
                           if isinstance(v, float))
            )

    # ── Metrics (case hook, common modules only) ──
    metrics: dict[str, Any] = {}
    if asm.metrics is not None:
        aux = {"steps": n_steps, "fx_hist": fx_hist, "fy_hist": fy_hist}
        metrics = asm.metrics(f, dict(asm.context), aux)

    finite = bool(torch.isfinite(f).all().item())
    if not finite:
        raise RuntimeError(
            f"case {case!r} produced non-finite populations after {n_steps} steps"
        )

    result: dict[str, Any] = {
        "case": case,
        "family": asm.family,
        "lattice": asm.lattice,
        "grid": asm.grid,
        "physics": asm.physics_resolved,
        "collision": asm.collision,
        "compile": {
            "requested_mode": compile_mode,
            "canonical_mode": canonical_mode,
            "routed": routed_desc,
            "route": ("benchmarks/compile_route.py"
                      if not isinstance(route, _DirectCompileRoute)
                      else "tensorlbm.compile_utils (direct fallback)"),
        },
        "steps": n_steps,
        "monitor_interval": interval,
        "device": str(dev),
        "seed": seed,
        "metrics": metrics,
        "finite": finite,
        "elapsed_s": round(time.time() - t0, 2),
        "modules_used": asm.modules_used,
    }

    with contextlib.suppress(Exception):
        (job.output_dir / "generic_sim_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    return result
