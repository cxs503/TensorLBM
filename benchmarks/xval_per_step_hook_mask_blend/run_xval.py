#!/usr/bin/env python3
"""XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND -- cross-validation benchmark.

Reference (OSS, inventoried): ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-
Tensorflow``.  Two coupled idioms are up for adoption:

* ``Domain.Solve(sess, Tf, initialize_step, setup_step, save_step, save_interval)``
  -- a driver loop whose per-step pre-processing and output cadence are injected
  as callbacks (``lid_cavity.py`` uses the hook to write the moving-lid velocity
  into ``domain.Vel``).
* ``Domain.CollideSC`` -- the boundary condition is one tensor expression over a
  float mask, ``f = f*(1 - boundary) + conv1x1(f*boundary, BOUNCE)``, so arbitrary
  geometry costs no code.

The port lives in :mod:`tensorlbm.step_hooks`.  Three questions decide adoption:

1. **Implementation** -- is the mask-blend operator bit-wise the same operator as
   the in-house explicit index-assignment BC, and is the hook driver
   physics-neutral?  (equivalence KPIs)
2. **Physics** -- does the ported formulation still hit the closed forms?
   Plane Couette (nothing but the moving wall drives it: the mask blend *is* the
   physics) and plane Poiseuille (curvature + body force) both have exact
   solutions.
3. **Difference** -- upstream's mask blend injects no wall momentum, so a moving
   wall can only be *declared* through the setup hook.  The arm
   ``D_upstream_lid`` runs that faithfully and measures how much flow it drives.

Run:  python benchmarks/xval_per_step_hook_mask_blend/run_xval.py
      python benchmarks/xval_per_step_hook_mask_blend/run_xval.py --quick
Docs: docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md is rendered from
      ``artifacts/xval_per_step_hook_mask_blend/result.json`` by the same command, so
      no number in the write-up is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import step_hooks as SH  # noqa: E402
from tensorlbm.d2q9 import C as C2D, OPPOSITE, W_EXACT, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.streaming_conv import bounce_blend  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "xval_per_step_hook_mask_blend"
DOC_PATH = ROOT / "docs" / "XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md"
ANCHOR = "XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND"
MISSION = 18130
OSS_REPO = "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow"
OSS_CANDIDATES = (
    Path("/tmp/oss_lbm/Lattice-Boltzmann-fluid-flow-in-Tensorflow-master"),
    ROOT / "benchmarks/xval_conv_as_streaming/oss_ref/checkout",
)
PREV_PROVENANCE = ROOT / "benchmarks/xval_conv_as_streaming/oss_ref/provenance.json"
DT = torch.float64
THREADS = 4


# ------------------------------------------------------------------ utilities
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f_digest(f: torch.Tensor) -> str:
    return sha256_bytes(f.detach().cpu().numpy().tobytes())


def l2_pct(got: np.ndarray, ref: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def rest_field(ny: int, nx: int) -> torch.Tensor:
    ones = torch.ones(ny, nx, dtype=DT)
    zeros = torch.zeros(ny, nx, dtype=DT)
    return equilibrium(ones, zeros, zeros)


def row_masks(ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.zeros(ny, nx, dtype=DT)
    hi = torch.zeros(ny, nx, dtype=DT)
    lo[0, :] = 1.0
    hi[-1, :] = 1.0
    return lo, hi


def plain_source(f: torch.Tensor, force: tuple[float, float]) -> torch.Tensor:
    """Momentum source ``dF_q = 3 w_q (c_q . F)`` shared by every forced arm.

    Summing ``dF_q * c_q`` returns exactly ``F`` per step, matching the steady
    balance ``u_peak = g H^2 / (8 nu)``.
    """
    c = C2D.to(dtype=f.dtype)
    w = torch.as_tensor(W_EXACT, dtype=f.dtype)
    cf = c[:, 0].view(9, 1, 1) * force[0] + c[:, 1].view(9, 1, 1) * force[1]
    return 3.0 * w.view(9, 1, 1) * cf


def velocity_with_force(f: torch.Tensor, force: tuple[float, float]) -> torch.Tensor:
    rho, ux, _uy = macroscopic(f)
    return ux + 0.5 * force[0] / rho


def moving_momentum(u_x: float, dtype: torch.dtype = DT) -> torch.Tensor:
    """``6 w_i (c_i . u_wall)`` as a ``(9, 1, 1)`` tensor for a wall moving in +x."""
    w = torch.as_tensor(W_EXACT, dtype=dtype)
    cu = C2D[:, 0].to(dtype) * u_x
    return 6.0 * w.view(9, 1, 1) * cu.view(9, 1, 1)


def ramp_value(i: int, ramp_steps: int, target: float) -> float:
    frac = 1.0 if ramp_steps <= 0 else min(1.0, float(i + 1) / float(ramp_steps))
    return target * frac


def explicit_step(
    f: torch.Tensor,
    tau: float,
    layers: list[tuple[torch.Tensor, float]],
    *,
    force: tuple[float, float] | None = None,
) -> torch.Tensor:
    """In-house style BC: explicit index assignment on cell sets.

    ``layers`` is a list of ``(mask, wall_speed_x)``; a stationary layer is
    ``wall_speed_x = 0``.  This is the baseline/control arm: no hooks, no blend.
    """
    f_streamed = stream(f)
    f_post = collide_bgk(f_streamed, tau)
    if force is not None:
        f_post = f_post + plain_source(f_streamed, force)
    out = f_post.clone()
    idx = OPPOSITE.to(torch.int64)
    for mask, wall_speed in layers:
        sel = mask > 0
        n = int(sel.sum())
        if n == 0:
            continue
        vals = f_streamed[idx][:, sel]
        if wall_speed:
            vals = vals + moving_momentum(wall_speed, f.dtype).view(9, 1)
        out[:, sel] = vals
    return out


# ------------------------------------------------------------- OSS provenance
def find_oss_checkout() -> Path | None:
    for cand in OSS_CANDIDATES:
        if (cand / "LatFlow" / "Domain.py").exists():
            return cand
    return None


def extract_oss_evidence(src: Path | None) -> dict:
    """Machine-extract the upstream idioms (nothing transcribed by hand)."""
    prev = json.loads(PREV_PROVENANCE.read_text()) if PREV_PROVENANCE.exists() else {}
    record: dict = {
        "repo": OSS_REPO,
        "tarball_url": prev.get("tarball_url"),
        "tarball_sha256": prev.get("tarball_sha256"),
        "recorded_file_sha256": prev.get("files", {}),
        "checkout": str(src) if src else None,
        "verified_file_sha256": {},
        "snippets": {},
    }
    if src is None:
        record["extraction"] = (
            "WITHHELD: no checkout available at run time; only the previously "
            "recorded sha256 are kept, no upstream text is quoted"
        )
        return record

    want = {
        "LatFlow/Domain.py": [
            ("solve_signature", "def Solve(self"),
            ("blend_line_split", "f_no_boundary = tf.multiply(f, (1.0-self.boundary))"),
            ("blend_line_sum", "f = f_no_boundary + f_boundary"),
            ("bounce_piece", "f_boundary = simple_conv(f_boundary, self.Op)"),
            ("mask_constant", "self.boundary = tf.constant(boundary)"),
            ("velocity_recompute", "Vel = simple_conv(f_pad, self.C)"),
        ],
        "lid_cavity.py": [("lid_setup_signature", "def lid_setup_step")],
    }
    for rel, needles in want.items():
        path = src / rel
        if not path.exists():
            continue
        raw = path.read_bytes()
        record["verified_file_sha256"][rel] = sha256_bytes(raw)
        lines = raw.decode("utf-8", "replace").splitlines()
        for tag, needle in needles:
            for lineno, line in enumerate(lines, 1):
                if needle in line:
                    record["snippets"][tag] = {
                        "file": rel,
                        "line": lineno,
                        "text": line.strip(),
                    }
                    break
        starts = {
            "solve_body": "def Solve(self",
            "lid_setup_step": "def lid_setup_step",
        }
        for tag, needle in starts.items():
            start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(needle)), None)
            if start is not None:
                record["snippets"][tag] = {
                    "file": rel,
                    "line": start + 1,
                    "text": "\n".join(lines[start : start + 16]).strip(),
                }
    record["extraction"] = "extracted verbatim from the checkout; per-file sha256 recorded"
    same = all(
        record["verified_file_sha256"].get(rel) == digest
        for rel, digest in prev.get("files", {}).items()
        if rel in record["verified_file_sha256"]
    )
    record["matches_recorded_provenance"] = bool(same)
    return record


# --------------------------------------------------------------- Couette case
COUETTE_ARMS = ("A_explicit", "B_hook_blend", "D_upstream_lid")


def run_couette(
    ny: int,
    nx: int,
    tau: float,
    u_wall: float,
    steps: int,
    arm: str,
    *,
    ramp_steps: int = 200,
    save_interval: int = 0,
    wall_convention: str = "halfway",
) -> dict:
    """Plane Couette: the lower wall is at rest, the lid moves with ``u_wall``.

    Analytic steady state ``u(y) = u_wall * (y - 0.5) / H`` with ``H = Ny - 2``
    for half-way bounce-back (walls at ``y = 0.5`` and ``y = Ny - 1.5``).
    """
    nu = (tau - 0.5) / 3.0
    lo, hi = row_masks(ny, nx)
    f0 = rest_field(ny, nx)
    tables = SH.tables_for(f0)
    saved: list[int] = []

    def saver(state: dict, t: float, i: int) -> None:
        saved.append(i)

    def init_vel(state: dict, t: float) -> dict:
        _, ux, uy = macroscopic(state["f"])
        state["vel"] = (ux, uy)
        return state

    t0 = time.perf_counter()
    with torch.no_grad():
        hook_bits = None
        if arm == "A_explicit":
            f = f0
            for i in range(steps):
                f = explicit_step(f, tau, [(lo, 0.0), (hi, ramp_value(i, ramp_steps, u_wall))])
            f_final = f
        elif arm == "B_hook_blend":
            layers = [
                SH.stationary_bounce_layer("lower_wall", lo),
                SH.moving_bounce_layer("lid", hi, lambda state, like: state["wall_velocity"]),
            ]
            op = SH.masked_step(tau=tau, layers=layers, tables=tables)
            setup = SH.velocity_ramp_hook(
                key="wall_velocity", velocity=(u_wall, 0.0), ramp_steps=ramp_steps
            )
            state, _ = SH.solve_with_hooks(
                {"f": f0.clone()},
                steps,
                operator=op,
                setup_step=setup,
                save_step=saver if save_interval else None,
                save_interval=save_interval,
            )
            f_final = state["f"]
        elif arm == "D_upstream_lid":
            # upstream lid_cavity.py: stationary bounce everywhere; the lid
            # velocity is only written into the solver's velocity variable by
            # the setup hook and recomputed from the populations each stream.
            layers = [SH.stationary_bounce_layer("walls", lo + hi)]
            base_op = SH.masked_step(tau=tau, layers=layers, tables=tables)

            def op(state: dict, t: float) -> dict:  # noqa: F811
                state = base_op(state, t)
                _, ux, uy = macroscopic(state["f"])
                state["vel"] = (ux, uy)
                return state

            def upstream_setup(state: dict, t: float, i: int) -> dict:
                ux, uy = state["vel"]
                ux = ux.clone()
                ux[-1, :] = ramp_value(i, ramp_steps, u_wall)  # lid_setup_step
                state["vel"] = (ux, uy)
                return state

            state, _ = SH.solve_with_hooks(
                {"f": f0.clone()},
                steps,
                operator=op,
                initialize_step=init_vel,
                setup_step=upstream_setup,
                save_step=saver if save_interval else None,
                save_interval=save_interval,
            )
            f_final = state["f"]
            ref_state, _ = SH.solve_with_hooks(
                {"f": f0.clone()}, steps, operator=op, initialize_step=init_vel
            )
            hook_bits = int((ref_state["f"] != f_final).sum())
        else:
            raise ValueError(f"unknown arm {arm!r}")
    wall_s = time.perf_counter() - t0

    H = float(ny - 2) if wall_convention == "halfway" else float(ny - 1)
    rows = np.arange(1, ny - 1, dtype=float)
    exact = u_wall * (rows - 0.5) / H
    with torch.no_grad():
        _, ux, _uy = macroscopic(f_final)
    got = ux.numpy()[1:-1, nx // 2]
    # no-slip measure: least-squares line through the fluid rows, extrapolated to
    # the two wall locations y = 0.5 (should be 0) and y = Ny - 1.5 (should be U)
    slope, intercept = np.polyfit(rows, got, 1)
    u_lower = intercept + slope * 0.5
    u_lid = intercept + slope * (ny - 1.5)
    return {
        "case": "couette",
        "slip_lower_pct": 100.0 * abs(float(u_lower)) / u_wall,
        "slip_lid_pct": 100.0 * abs(float(u_lid) - u_wall) / u_wall,
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "u_wall": u_wall,
        "H": H,
        "wall_convention": wall_convention,
        "mach": math.sqrt(3.0) * u_wall,
        "Re_H": u_wall * H / nu,
        "steps": steps,
        "ramp_steps": ramp_steps,
        "save_interval": save_interval,
        "saved_steps": saved,
        "profile_l2_pct": l2_pct(got, exact),
        "max_err_pct": 100.0 * float(np.abs(got - exact).max() / u_wall),
        "interior_umax_over_uwall_pct": 100.0 * float(np.abs(ux.numpy()[1:-1, :]).max() / u_wall),
        "interior_umax_rel_pct": 100.0
        * abs(float(np.abs(ux.numpy()[1:-1, :]).max()) - u_wall * (H - 0.5) / H)
        / (u_wall * (H - 0.5) / H),
        "finite": bool(torch.isfinite(f_final).all()),
        "f_sha256": f_digest(f_final),
        "hook_bits_changed": hook_bits,
        "wall_s": wall_s,
        "s_per_step_ms": 1000.0 * wall_s / max(1, steps),
        "_profile": got,
        "_exact": exact,
    }


# ------------------------------------------------------------ Poiseuille case
POISEUILLE_ARMS = ("A_explicit", "B_hook_blend")


def run_poiseuille(ny: int, nx: int, tau: float, u_peak: float, steps: int, arm: str) -> dict:
    nu = (tau - 0.5) / 3.0
    H = float(ny - 2)
    g = 8.0 * nu * u_peak / H**2
    force = (g, 0.0)
    lo, hi = row_masks(ny, nx)
    f0 = rest_field(ny, nx)
    tables = SH.tables_for(f0)

    t0 = time.perf_counter()
    with torch.no_grad():
        if arm == "A_explicit":
            f = f0
            for _ in range(steps):
                f = explicit_step(f, tau, [(lo, 0.0), (hi, 0.0)], force=force)
            f_final = f
        else:
            layers = [
                SH.stationary_bounce_layer("lower_wall", lo),
                SH.stationary_bounce_layer("upper_wall", hi),
            ]
            op = SH.masked_step(
                tau=tau,
                layers=layers,
                tables=tables,
                source=lambda f_str, state: plain_source(f_str, force),
            )
            state, _ = SH.solve_with_hooks({"f": f0.clone()}, steps, operator=op)
            f_final = state["f"]
    wall_s = time.perf_counter() - t0

    with torch.no_grad():
        ux = velocity_with_force(f_final, force)
    prof = ux.numpy()[1:-1, nx // 2]
    y_pos = np.arange(1, ny - 1, dtype=float) - 0.5
    exact = g / (2.0 * nu) * y_pos * (H - y_pos)
    return {
        "case": "poiseuille",
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "H": H,
        "g": g,
        "u_peak_target": u_peak,
        "mach": math.sqrt(3.0) * u_peak,
        "Re_H": u_peak * H / nu,
        "steps": steps,
        "profile_l2_pct": l2_pct(prof, exact),
        "peak_err_pct": 100.0 * abs(float(prof.max()) - float(exact.max())) / float(exact.max()),
        "finite": bool(torch.isfinite(f_final).all()),
        "f_sha256": f_digest(f_final),
        "wall_s": wall_s,
        "s_per_step_ms": 1000.0 * wall_s / max(1, steps),
        "_profile": prof,
        "_exact": exact,
    }


# ------------------------------------------------------ arbitrary-mask checks
def run_mask_invariance(ny: int = 24, nx: int = 24, tau: float = 0.8, steps: int = 500) -> dict:
    """Arbitrary geometry: a blob mask plus a disjoint moving patch.

    1. one step from an identical state: explicit BC vs mask blend (equivalence);
    2. 500 steps in a closed box: mass drift;
    3. no-penetration: normal velocity carried by the masked cells.
    """
    torch.manual_seed(20261113)
    f0 = torch.rand(9, ny, nx, dtype=DT) * 0.1 + 0.05
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=DT), torch.arange(nx, dtype=DT), indexing="ij"
    )
    blob = (((yy - 8.0) ** 2 + (xx - 16.0) ** 2) < 16.0).to(DT)
    patch = torch.zeros(ny, nx, dtype=DT)
    patch[0, :] = 1.0
    assert float((blob + patch).max()) <= 1.0 + 1e-12

    f_streamed = stream(f0)
    f_post = collide_bgk(f_streamed, tau)
    explicit = explicit_step(f0, tau, [(patch, 0.0), (blob, 0.0)])
    tables = SH.tables_for(f0)
    blended = SH.compose_masked_boundaries(
        f_post,
        f_streamed,
        [SH.stationary_bounce_layer("patch", patch), SH.stationary_bounce_layer("blob", blob)],
        tables=tables,
    )
    rel_blend = float((blended - explicit).abs().max() / explicit.abs().max())
    conv = bounce_blend(f_post, f_streamed, blob)
    single_explicit = f_post.clone()
    sel = blob > 0
    single_explicit[:, sel] = f_streamed[OPPOSITE.to(torch.int64)][:, sel]
    rel_conv = float((conv - single_explicit).abs().max() / single_explicit.abs().max())

    op = SH.masked_step(
        tau=tau,
        layers=[
            SH.stationary_bounce_layer("patch", patch),
            SH.stationary_bounce_layer("blob", blob),
        ],
        tables=tables,
    )
    with torch.no_grad():
        state, _ = SH.solve_with_hooks({"f": f0.clone()}, steps, operator=op)
        mass0 = float(f0.sum())
        mass1 = float(state["f"].sum())
        _, ux, uy = macroscopic(state["f"])
        umax = float(ux.abs().max())
        # diagnostic only: bounce-back encodes no-slip through the population
        # exchange, so a wall cell keeps a non-zero *macroscopic* velocity
        wall_cell_speed = float(uy[sel].abs().max() / umax) if umax > 0 else float("nan")
    return {
        "case": "mask_invariance",
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "steps": steps,
        "masked_cells": int(sel.sum()) + int((patch > 0).sum()),
        "equivalence_max_rel": rel_blend,
        "conv_vs_explicit_max_rel": rel_conv,
        "mass_drift_rel": abs(mass1 - mass0) / mass0,
        "wall_cell_speed_rel": wall_cell_speed,
        "f_sha256": f_digest(state["f"]),
        "finite": bool(torch.isfinite(state["f"]).all()),
    }


def run_composite_channel(
    ny: int = 16, nx: int = 32, tau: float = 0.8, u_in: float = 0.02, steps: int = 6000
) -> dict:
    """Composite mask: inlet (equilibrium) + walls (bounce) + hook-driven outlet.

    No closed-form reference exists for this developing channel, so only
    invariants are claimed (mass flux in vs out, no-penetration on the walls).
    """
    nu = (tau - 0.5) / 3.0
    f0 = rest_field(ny, nx)
    tables = SH.tables_for(f0)
    lo = torch.zeros(ny, nx, dtype=DT)
    hi = torch.zeros(ny, nx, dtype=DT)
    inlet = torch.zeros(ny, nx, dtype=DT)
    outlet = torch.zeros(ny, nx, dtype=DT)
    lo[0, 1:-1] = 1.0
    hi[-1, 1:-1] = 1.0
    inlet[1:-1, 0] = 1.0
    outlet[1:-1, -1] = 1.0
    layers = [
        SH.stationary_bounce_layer("lower_wall", lo),
        SH.stationary_bounce_layer("upper_wall", hi),
        SH.equilibrium_layer("inlet", inlet, (u_in, 0.0), density=1.0),
        SH.equilibrium_layer(
            "outlet", outlet, lambda state, like: state["u_out"], density=1.0
        ),
    ]

    def zero_gradient_hook(state: dict, t: float, i: int) -> dict:
        _, ux, _uy = macroscopic(state["f"])
        state["u_out"] = torch.tensor([float(ux[1:-1, -2].mean()), 0.0], dtype=state["f"].dtype)
        return state

    op = SH.masked_step(tau=tau, layers=layers, tables=tables)
    t0 = time.perf_counter()
    with torch.no_grad():
        state, log = SH.solve_with_hooks(
            {"f": f0.clone(), "u_out": torch.zeros(2, dtype=DT)},
            100,
            operator=op,
            setup_step=zero_gradient_hook,
        )
        state, log2 = SH.solve_with_hooks(
            state, steps - 100, operator=op, setup_step=zero_gradient_hook, t0=100.0
        )
        rho, ux, uy = macroscopic(state["f"])
        q_in = float((rho[1:-1, 1] * ux[1:-1, 1]).sum())
        q_out = float((rho[1:-1, -2] * ux[1:-1, -2]).sum())
        umax = float(ux[1:-1, 1:-1].abs().max())
        # no-penetration is a statement about the fluid cell next to the wall
        wall_norm = float(uy[1, 1:-1].abs().max() / umax)
    wall_s = time.perf_counter() - t0
    return {
        "case": "composite_channel",
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "u_in": u_in,
        "mach": math.sqrt(3.0) * u_in,
        "Re": u_in * (ny - 2) / nu,
        "steps": steps,
        "q_in": q_in,
        "q_out": q_out,
        "mass_imbalance_pct": 100.0 * abs(q_out - q_in) / abs(q_in),
        "adjacent_normal_velocity_rel": wall_norm,
        "setup_calls": len(log.setup) + len(log2.setup),
        "f_sha256": f_digest(state["f"]),
        "finite": bool(torch.isfinite(state["f"]).all()),
        "wall_s": wall_s,
        "_ux": ux.numpy(),
        "_rho": rho.numpy(),
    }


# ------------------------------------------------------------------- figures
def make_figures(result: dict, fig_dir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    cou = result["couette"]["configs"]
    ref_ny = max(c["ny"] for c in cou)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm, style, label in (
        ("A_explicit", "o", "A explicit index BC"),
        ("B_hook_blend", "s", "B hook + mask blend (ported)"),
        ("D_upstream_lid", "x", "D upstream-faithful lid (velocity-overwrite hook)"),
    ):
        row = next((c for c in cou if c["ny"] == ref_ny and c["arm"] == arm), None)
        if row is None:
            continue
        y = np.arange(1, ref_ny - 1, dtype=float) - 0.5
        ax.plot(np.asarray(row["_profile"]), y, style, ms=4, mfc="none", label=label)
    y = np.arange(1, ref_ny - 1, dtype=float) - 0.5
    exact = np.asarray(result["couette"]["exact_profile"][ref_ny])
    ax.plot(exact, y, "-", color="k", lw=1.2, label="analytic u = U (y-0.5)/H")
    ax.set_xlabel("u_x (lattice)")
    ax.set_ylabel("y - 0.5")
    ax.set_title(f"Plane Couette, Ny={ref_ny}: moving lid carried by the mask blend")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "fig1_couette_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": p.name,
            "path": str(p),
            "caption": "Plane Couette velocity profiles against the analytic line (final step).",
        }
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key, style in (("couette", "o-"), ("poiseuille", "s-")):
        cfg = result[key]["configs"]
        nys = sorted({c["ny"] for c in cfg if c["arm"] == "B_hook_blend"})
        vals = [
            next(c for c in cfg if c["ny"] == ny and c["arm"] == "B_hook_blend")["profile_l2_pct"]
            for ny in nys
        ]
        ax.plot(nys, vals, style, label=f"{key} (arm B hook+blend)")
    ax.axhline(3.0, color="tab:green", ls=":", lw=1.2, label="3% internal target")
    ax.axhline(15.0, color="tab:red", ls="--", lw=1.2, label="15% owner gate")
    ax.set_xlabel("channel height Ny")
    ax.set_ylabel("profile L2 error (%)")
    ax.set_yscale("log")
    ax.set_title("Grid convergence of the ported formulation")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "fig2_grid_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": p.name,
            "path": str(p),
            "caption": "Profile L2 error against the analytic solutions versus channel height.",
        }
    )

    ups = result["upstream_lid"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["upstream-faithful lid\n(velocity-overwrite hook)", "port: hook + mask blend\n(wall-momentum term)"]
    vals = [ups["flow_pct_of_uwall"], ups["ported_flow_pct_of_uwall"]]
    ax.bar(labels, vals, color=["tab:red", "tab:green"])
    ax.axhline(100.0, color="k", ls=":", lw=1.0, label="imposed U_wall")
    ax.set_ylabel("measured peak |u_x| / U_wall (%)")
    ax.set_title("Why the port needs a wall-momentum term")
    for b, v in zip(ax.patches, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%", ha="center", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "fig3_upstream_lid_gap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": p.name,
            "path": str(p),
            "caption": "Flow driven by upstream's velocity-overwrite hook versus the momentum-carrying mask blend.",
        }
    )

    comp = result["composite_channel"]
    ux = np.asarray(comp["_ux"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(ux, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xlabel("x (lattice)")
    ax.set_ylabel("y (lattice)")
    ax.set_title("u_x in the composite-mask channel (walls + inlet + hook-driven outlet)")
    fig.colorbar(im, ax=ax, label="u_x")
    fig.tight_layout()
    p = fig_dir / "fig4_composite_channel_ux.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": p.name,
            "path": str(p),
            "caption": "Streamwise velocity field from four mask layers (two bounce walls, equilibrium inlet, hook-driven outlet).",
        }
    )

    poi = result["poiseuille"]["configs"]
    ny_ref = max(c["ny"] for c in poi)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for arm, style in (("A_explicit", "o"), ("B_hook_blend", "s")):
        row = next((c for c in poi if c["ny"] == ny_ref and c["arm"] == arm), None)
        if row is None:
            continue
        y = np.arange(1, ny_ref - 1, dtype=float) - 0.5
        ax.plot(np.asarray(row["_profile"]), y, style, ms=4, mfc="none", label=arm)
    y = np.arange(1, ny_ref - 1, dtype=float) - 0.5
    ax.plot(np.asarray(result["poiseuille"]["exact_profile"][ny_ref]), y, "-", color="k",
            lw=1.2, label="analytic g y (H-y) / 2nu")
    ax.set_xlabel("u_x (lattice)")
    ax.set_ylabel("y - 0.5")
    ax.set_title(f"Plane Poiseuille, Ny={ny_ref}: forced channel with two mask layers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "fig5_poiseuille_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": p.name,
            "path": str(p),
            "caption": "Plane Poiseuille profile: explicit BC arm, hook+mask-blend arm, analytic parabola.",
        }
    )
    return figs


# ---------------------------------------------------------------------- main
def steps_for_couette(H: float, nu: float) -> int:
    """12 transient e-folding times of the Couette start-up (residual < 1e-5).

    The start-up transient decays as ``exp(-pi^2 nu t / H^2)``; 4 e-folding
    times leave a ~1 % residual and masquerade as a discretisation error, so the
    step count is fixed from the analytic decay constant at 12 e-folding times.
    """
    return int(12.0 * H * H / (math.pi**2 * nu)) + 200


def steps_for_poiseuille(H: float, nu: float) -> int:
    return int(3.5 * H * H / nu)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return "unknown"


def build_checks(couette_rows, pois_rows, sens, invariance) -> list[dict]:
    def find(rows, ny, arm):
        return next(r for r in rows if r["ny"] == ny and r["arm"] == arm)

    cou_ref = max({r["ny"] for r in couette_rows if r["arm"] == "B_hook_blend"})
    poi_ref = max({r["ny"] for r in pois_rows if r["arm"] == "B_hook_blend"})
    b_cou = find(couette_rows, cou_ref, "B_hook_blend")
    a_cou = find(couette_rows, cou_ref, "A_explicit")
    b_poi = find(pois_rows, poi_ref, "B_hook_blend")
    a_poi = find(pois_rows, poi_ref, "A_explicit")
    checks = [
        {
            "kpi": f"couette_profile_l2_pct (arm B, Ny={cou_ref})",
            "value": b_cou["profile_l2_pct"],
            "gated": True,
        },
        {
            "kpi": f"couette_max_err_pct (arm B, Ny={cou_ref})",
            "value": b_cou["max_err_pct"],
            "gated": True,
        },
        {
            "kpi": f"couette_profile_l2_pct (arm A baseline, Ny={cou_ref})",
            "value": a_cou["profile_l2_pct"],
            "gated": True,
        },
        {
            "kpi": f"poiseuille_profile_l2_pct (arm B, Ny={poi_ref})",
            "value": b_poi["profile_l2_pct"],
            "gated": True,
        },
        {
            "kpi": f"poiseuille_peak_err_pct (arm B, Ny={poi_ref})",
            "value": b_poi["peak_err_pct"],
            "gated": True,
        },
        {
            "kpi": f"poiseuille_profile_l2_pct (arm A baseline, Ny={poi_ref})",
            "value": a_poi["profile_l2_pct"],
            "gated": True,
        },
        {
            "kpi": "explicit-vs-blend equivalence (max rel, blob+patch masks)",
            "value": 100.0 * invariance["equivalence_max_rel"],
            "gated": True,
        },
        {
            "kpi": "hook cadence neutrality (save_interval 50 vs 500, sha256 equal)",
            "value": 0.0 if sens["save_interval_sha_equal"] else 100.0,
            "gated": True,
        },
    ]
    for c in checks:
        c["pass_15"] = c["value"] < 15.0
        c["pass_3"] = c["value"] < 3.0
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="CI-sized subset")
    ap.add_argument("--oss-src", default=None, help="path to an upstream checkout")
    args = ap.parse_args(argv)

    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    t_start = time.perf_counter()

    ny_list_couette = (16, 24) if args.quick else (16, 24, 32)
    ny_list_pois = (16,) if args.quick else (16, 24)
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    u_wall = 0.02
    u_peak = 0.02
    nx = 8

    oss_src = Path(args.oss_src) if args.oss_src else find_oss_checkout()
    evidence = extract_oss_evidence(oss_src)

    couette_rows: list[dict] = []
    exact_profiles: dict[int, list[float]] = {}
    for ny in ny_list_couette:
        H = float(ny - 2)
        steps = steps_for_couette(H, nu)
        for arm in COUETTE_ARMS:
            row = run_couette(ny, nx, tau, u_wall, steps, arm)
            couette_rows.append(row)
            exact_profiles[ny] = row["_exact"].tolist()
            print(
                f"[couette] Ny={ny} arm={arm}: L2={row['profile_l2_pct']:.4f}% "
                f"max={row['max_err_pct']:.4f}% umax={row['interior_umax_over_uwall_pct']:.3f}%U "
                f"bits={row['hook_bits_changed']}",
                flush=True,
            )

    sens: dict = {}
    ny_s = 24
    steps_s = steps_for_couette(float(ny_s - 2), nu)
    sens["wall_convention_node"] = run_couette(
        ny_s, nx, tau, u_wall, steps_s, "B_hook_blend", wall_convention="node"
    )
    sens["save_interval_50"] = run_couette(
        ny_s, nx, tau, u_wall, steps_s, "B_hook_blend", save_interval=50
    )
    sens["save_interval_500"] = run_couette(
        ny_s, nx, tau, u_wall, steps_s, "B_hook_blend", save_interval=500
    )
    sens["ramp_1"] = run_couette(ny_s, nx, tau, u_wall, steps_s, "B_hook_blend", ramp_steps=1)
    sens["ramp_200"] = run_couette(ny_s, nx, tau, u_wall, steps_s, "B_hook_blend", ramp_steps=200)
    sens["save_interval_sha_equal"] = (
        sens["save_interval_50"]["f_sha256"] == sens["save_interval_500"]["f_sha256"]
    )
    sens["ramp_profile_delta_pct"] = abs(
        sens["ramp_1"]["profile_l2_pct"] - sens["ramp_200"]["profile_l2_pct"]
    )
    print("[couette] sensitivity done", flush=True)

    pois_rows: list[dict] = []
    exact_pois: dict[int, list[float]] = {}
    for ny in ny_list_pois:
        H = float(ny - 2)
        steps = steps_for_poiseuille(H, nu)
        for arm in POISEUILLE_ARMS:
            row = run_poiseuille(ny, nx, tau, u_peak, steps, arm)
            pois_rows.append(row)
            exact_pois[ny] = row["_exact"].tolist()
            print(
                f"[poiseuille] Ny={ny} arm={arm}: L2={row['profile_l2_pct']:.4f}% "
                f"peak={row['peak_err_pct']:.4f}%",
                flush=True,
            )

    invariance = run_mask_invariance()
    print("[invariance]", invariance["equivalence_max_rel"], invariance["mass_drift_rel"], flush=True)
    composite = run_composite_channel()
    print("[composite] imbalance", composite["mass_imbalance_pct"], flush=True)

    ups_ref = max((r for r in couette_rows if r["arm"] == "D_upstream_lid"), key=lambda r: r["ny"])
    b_ref = next(
        r for r in couette_rows if r["arm"] == "B_hook_blend" and r["ny"] == ups_ref["ny"]
    )
    upstream_lid = {
        "ny": ups_ref["ny"],
        "u_wall": ups_ref["u_wall"],
        "steps": ups_ref["steps"],
        "flow_pct_of_uwall": ups_ref["interior_umax_over_uwall_pct"],
        "profile_l2_pct": ups_ref["profile_l2_pct"],
        "hook_bits_changed": ups_ref["hook_bits_changed"],
        "hook_bits_total": int(ups_ref["ny"] * ups_ref["nx"] * 9),
        "ported_flow_pct_of_uwall": b_ref["interior_umax_over_uwall_pct"],
        "ported_profile_l2_pct": b_ref["profile_l2_pct"],
    }

    checks = build_checks(couette_rows, pois_rows, sens, invariance)
    gate15 = all(c["pass_15"] for c in checks if c["gated"])
    gate3 = all(c["pass_3"] for c in checks if c["gated"])

    # timings: hook machinery overhead, same grid and step count
    a_t = next(r for r in couette_rows if r["arm"] == "A_explicit" and r["ny"] == ups_ref["ny"])
    b_t = next(r for r in couette_rows if r["arm"] == "B_hook_blend" and r["ny"] == ups_ref["ny"])
    timing = {
        "explicit_ms_per_step": a_t["s_per_step_ms"],
        "hook_blend_ms_per_step": b_t["s_per_step_ms"],
        "hook_overhead_pct": 100.0 * (b_t["s_per_step_ms"] - a_t["s_per_step_ms"]) / a_t["s_per_step_ms"],
        "total_s": time.perf_counter() - t_start,
    }

    result: dict = {
        "anchor": ANCHOR,
        "mission": MISSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "dtype": "float64",
            "threads": THREADS,
            "cuda": bool(torch.cuda.is_available()),
        },
        "predeclared": json.loads((HERE / "predeclared.json").read_text()),
        "reference": {
            "oss_repo": OSS_REPO,
            "evidence": evidence,
            "formulas": {
                "couette": "u(y) = U_wall * (y - 0.5) / (Ny - 2), half-way walls at y=0.5 and y=Ny-1.5",
                "poiseuille": "u(y) = g * y * (H - y) / (2 nu), H = Ny - 2, g the imposed acceleration",
            },
        },
        "couette": {"configs": couette_rows, "exact_profile": exact_profiles},
        "poiseuille": {"configs": pois_rows, "exact_profile": exact_pois},
        "sensitivity": {k: v for k, v in sens.items() if k != "save_interval_sha_equal"}
        | {"save_interval_sha_equal": sens["save_interval_sha_equal"]},
        "invariance": invariance,
        "composite_channel": composite,
        "upstream_lid": upstream_lid,
        "timing": timing,
        "verdict": {
            "verdict": "PASS" if gate15 else "FAIL",
            "gate_15pct": bool(gate15),
            "gate_3pct": bool(gate3),
            "checks": checks,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = make_figures(result, OUT_DIR / "figures")
    result["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]

    # CSV of every arm x configuration
    rows = []
    for r in couette_rows + pois_rows:
        rows.append(
            {
                "case": r["case"],
                "arm": r["arm"],
                "ny": r["ny"],
                "tau": r["tau"],
                "nu": r["nu"],
                "steps": r["steps"],
                "profile_l2_pct": round(r["profile_l2_pct"], 6),
                "max_or_peak_err_pct": round(
                    r.get("max_err_pct", r.get("peak_err_pct", float("nan"))), 6
                ),
                "f_sha256": r["f_sha256"],
            }
        )
    with (OUT_DIR / "table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # strip heavy arrays before serialising the summary, keep a npz for figures
    np.savez_compressed(
        OUT_DIR / "profiles.npz",
        **{f"couette_{r['ny']}_{r['arm']}": np.asarray(r["_profile"]) for r in couette_rows},
        **{f"poiseuille_{r['ny']}_{r['arm']}": np.asarray(r["_profile"]) for r in pois_rows},
        composite_ux=np.asarray(composite["_ux"]),
        composite_rho=np.asarray(composite["_rho"]),
    )
    def strip(row: dict) -> dict:
        return {k: v for k, v in row.items() if not k.startswith("_")}

    slim = {k: v for k, v in result.items()}
    slim["couette"] = {"configs": [strip(r) for r in couette_rows], "exact_profile": exact_profiles}
    slim["poiseuille"] = {"configs": [strip(r) for r in pois_rows], "exact_profile": exact_pois}
    slim["sensitivity"] = {
        k: (strip(v) if isinstance(v, dict) and "profile_l2_pct" in v else v)
        for k, v in result["sensitivity"].items()
    }
    slim["composite_channel"] = strip(composite)
    (OUT_DIR / "result.json").write_text(json.dumps(slim, indent=1, sort_keys=False, default=float))
    (OUT_DIR / "oss_ref").mkdir(exist_ok=True)
    (OUT_DIR / "oss_ref" / "evidence.json").write_text(json.dumps(evidence, indent=1))

    sys.path.insert(0, str(HERE))
    from render_report import build_markdown  # noqa: E402  (same directory)

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(build_markdown(result))

    print(
        f"[verdict] {result['verdict']['verdict']} gate15={gate15} gate3={gate3} "
        f"total={timing['total_s']:.1f}s -> {OUT_DIR}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())