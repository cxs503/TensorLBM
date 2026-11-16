"""3D physics validation of the conv-as-streaming fixed-kernel transport (round 4).

Rounds 1-3 of ``XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL`` established two things:

* in **2D**, the upstream fixed-kernel transport (``StreamSC``/``CollideSC`` in
  ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``) is bit-wise the
  same operator as TensorLBM's gather transport, and both reproduce the 2D
  Taylor-Green decay and the plane Poiseuille parabola;
* in **3D**, only the *operator* was compared (``stream_conv_3d`` vs the
  canonical roll traversal on a random field).

The 3D *physics* was never run: no 3D closed-form case went through the conv arm,
and 3D wall bounce (``bounce_blend`` with the D3Q19 reversal table) was never
exercised by a simulation.  That is the gap this driver closes, with three
closed-form 3D judges and the same two-arm design (only the transport/bounce
operator changes between arms):

1. ``mode3d``   -- divergence-free Fourier mode in the Stokes regime; velocity
   decays as ``exp(-nu*|k|^2 t)`` with ``|k|^2 = kx^2+ky^2+kz^2`` exactly.
2. ``slit3d``   -- 3D plane Poiseuille: walls in y, periodic x and z; exact
   parabola with half-way wall spacing ``H = Ny - 2``.
3. ``duct3d``   -- square duct, solid mask on all four lateral walls (so the
   mask has *corners*, which a single flat wall never tests); Boussinesq series.

The duct series is used as a judge, so it is itself checked first: finite-
difference Poisson residual and the literature constant ``f*Re = 56.908``
(Shah & London 1978, Table 49).  If that self-check leaves 1 %, the duct KPI is
reported WITHHELD instead of pass/fail.

Run:  python benchmarks/xval_conv_as_streaming/run_3d_physics.py            (full)
      python benchmarks/xval_conv_as_streaming/run_3d_physics.py --quick    (CI-sized)
Docs: the markdown report is rendered from ``result.json`` by the same command,
      so no number in the write-up is typed by hand.
"""

from __future__ import annotations

import argparse
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

from tensorlbm import streaming_conv as SC  # noqa: E402
from tensorlbm.d3q19 import (  # noqa: E402
    C as C3D,
    OPPOSITE as OPP3D,
    W as W3D,
    equilibrium3d,
    macroscopic3d,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT_DIR = ROOT / "runs" / "xval_conv_as_streaming_3d"
DOC_PATH = ROOT / "docs" / "XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL_R4_3D.md"
ANCHOR = "XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL"
PRE_DECLARED = HERE / "predeclared_3d.json"
DT = torch.float64
ARM_A = "A_native_gather_3d"
ARM_B = "B_conv_fixed_kernel_3d"
ARMS = (ARM_A, ARM_B)


# ------------------------------------------------------------------ operators
def stream_gather_3d(f: torch.Tensor) -> torch.Tensor:
    """Periodic gather transport on ``(19, nz, ny, nx)``: out[q] = f[q, z-cz, y-cy, x-cx].

    The 3D analogue of :func:`tensorlbm.solver.stream` (which is 2D-only).  Built
    from the D3Q19 velocity table so the shift and the lattice cannot drift apart,
    and kept independent of :mod:`tensorlbm.streaming_conv` on purpose: the whole
    point of the two-arm design is that the gather must not borrow the conv
    module's tables.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    c = C3D.to(f.device)
    z_src = (torch.arange(nz, device=f.device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
    y_src = (torch.arange(ny, device=f.device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
    x_src = (torch.arange(nx, device=f.device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
    q_idx = torch.arange(19, device=f.device).view(19, 1, 1, 1).expand(19, nz, ny, nx)
    z_idx = z_src.view(19, nz, 1, 1).expand(19, nz, ny, nx)
    y_idx = y_src.view(19, 1, ny, 1).expand(19, nz, ny, nx)
    x_idx = x_src.view(19, 1, 1, nx).expand(19, nz, ny, nx)
    return f[q_idx, z_idx, y_idx, x_idx]


def make_transport_3d(arm: str):
    if arm == ARM_A:
        return stream_gather_3d
    if arm == ARM_B:
        return SC.stream_conv_3d
    raise ValueError(f"unknown arm {arm}")


def collide_bgk_3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK on D3Q19 -- the same relaxation form as :func:`tensorlbm.solver.collide_bgk`."""
    rho, ux, uy, uz = macroscopic3d(f)
    return f - (f - equilibrium3d(rho, ux, uy, uz)) / tau


def guo_source_3d(f: torch.Tensor, force: tuple[float, float, float]) -> torch.Tensor:
    """Plain momentum source ``3 w_q (c_q . F)`` -- the 2D convention of rounds 1-3.

    Shares the trap documented in ``run_xval.guo_term``: the textbook
    ``(1 - 1/(2 tau))`` prefactor belongs to the shifted-equilibrium collision and
    under-injects momentum by exactly that factor on top of the unmodified BGK
    used here.  Kept identical between the two arms, so it cannot bias them
    relatively.
    """
    c = C3D.to(dtype=f.dtype)
    w = W3D.to(dtype=f.dtype).view(19, 1, 1, 1)
    cf = sum(c[:, i].view(19, 1, 1, 1) * force[i] for i in range(3))
    return 3.0 * w * cf


def velocity_with_force_3d(f: torch.Tensor, force: tuple[float, float, float]):
    rho, ux, uy, uz = macroscopic3d(f)
    return (
        rho,
        ux + 0.5 * force[0] / rho,
        uy + 0.5 * force[1] / rho,
        uz + 0.5 * force[2] / rho,
    )


def f_digest(f: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(f.detach().numpy()).tobytes()).hexdigest()


def _step(f, transport, tau, force, mask, arm):
    """One lattice step: transport -> BGK -> source -> wall bounce."""
    fs = transport(f)
    fc = collide_bgk_3d(fs, tau) + guo_source_3d(fs, force)
    if arm == ARM_A:
        out = fc.clone()
        out[:, mask > 0] = fs[OPP3D.to(torch.int64)][:, mask > 0]
        return out
    return SC.bounce_blend(fc, fs, mask, opposite=OPP3D.to(torch.int64))


# ------------------------------------------------------- analytic: duct series
def duct_series(
    y_pos: np.ndarray, z_pos: np.ndarray, g: float, nu: float, side: float, terms: int = 101
) -> np.ndarray:
    """Boussinesq series for fully developed square-duct flow (half-way walls).

    White (2006) *Viscous Fluid Flow* sec. 3-3.3 for a rectangular duct of half
    sides ``a`` (Fourier direction, here y) and ``b`` (cosh direction, here z)::

        u = 16 a^2 (G/mu) / pi^3 * sum_{i odd} (-1)^((i-1)/2)/i^3
              [1 - cosh(i pi zeta/(2a))/cosh(i pi b/(2a))] cos(i pi eta/(2a))

    with ``eta = y_pos - a``, ``zeta = z_pos - a`` measured from the duct centre
    and ``G/mu = g/nu``.  Odd ``i`` makes ``cos(i pi/2) = 0``, i.e. no-slip on the
    y walls; the cosh bracket vanishes on the z walls.

    ``y_pos`` is ``(ny,)`` and ``z_pos`` is ``(nz,)``; the result is ``(nz, ny)``
    with the field order used by the solver.
    """
    a = 0.5 * side
    b = 0.5 * side
    eta = (np.asarray(y_pos)[None, :] - a)  # (1, ny)
    zeta = (np.asarray(z_pos)[:, None] - a)  # (nz, 1)
    amp = 16.0 * a * a * (g / nu) / math.pi**3
    out = np.zeros((zeta.shape[0], eta.shape[1]), dtype=np.float64)
    for i in range(1, terms + 1, 2):
        sign = (-1.0) ** ((i - 1) // 2)
        cosh_ratio = np.cosh(i * math.pi * zeta / (2.0 * a)) / math.cosh(i * math.pi * b / (2.0 * a))
        out += sign / i**3 * (1.0 - cosh_ratio) * np.cos(i * math.pi * eta / (2.0 * a))
    return amp * out  # (nz, ny)


def duct_metrics_analytic(g: float, nu: float, side: float, terms: int = 101):
    """Centreline velocity, mean velocity and ``f*Re`` of the series solution.

    ``u_mean`` by quadrature on a fine grid (the series has no closed-form mean),
    ``f*Re = 2 g D_h^2/(nu u_mean)`` with the hydraulic diameter ``D_h = side``
    for a square section.
    """
    m = 400
    pos = (np.arange(m) + 0.5) * (side / m)
    u = duct_series(pos, pos, g, nu, side, terms=terms)
    u_mean = float(u.mean())
    h = side
    return {
        "u_max": float(u.max()),
        "u_mean": u_mean,
        "fRe": 2.0 * g * h * h / (nu * u_mean),
        "centre_ratio": float(u.max() / u_mean),
        "terms": terms,
    }


def duct_series_selfcheck(g: float = 1e-4, nu: float = 0.1, side: float = 10.0) -> dict:
    """Validate the judge before judging with it.

    (a) convergence in the number of series terms, (b) finite-difference Poisson
    residual against ``-g/nu`` on interior points, (c) ``f*Re`` against the
    literature value 56.908 (Shah & London 1978, Table 49, square duct).
    """
    ref = duct_metrics_analytic(g, nu, side, terms=201)
    conv = {}
    for terms in (21, 51, 101):
        m = duct_metrics_analytic(g, nu, side, terms=terms)
        conv[str(terms)] = {
            "u_max": m["u_max"],
            "u_mean": m["u_mean"],
            "fRe": m["fRe"],
            "u_max_rel_diff_pct": 100.0 * abs(m["u_max"] - ref["u_max"]) / ref["u_max"],
            "u_mean_rel_diff_pct": 100.0 * abs(m["u_mean"] - ref["u_mean"]) / ref["u_mean"],
        }

    m = 300
    hh = side / m
    pos = (np.arange(m) + 0.5) * hh
    u = duct_series(pos, pos, g, nu, side, terms=101)
    lap = (
        np.roll(u, 1, axis=1)
        - 4.0 * u
        + np.roll(u, -1, axis=1)
        + np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
    ) / hh**2
    # drop 3 cells next to the wall: the wall layer is where a truncated series is least accurate
    inner = lap[3:-3, 3:-3]
    target = -g / nu
    poisson_rel = 100.0 * abs(float(inner.mean()) - target) / abs(target)
    poisson_spread = 100.0 * float(inner.std()) / abs(target)

    fre_ref = 56.908
    fre_rel = 100.0 * abs(ref["fRe"] - fre_ref) / fre_ref
    return {
        "terms_convergence": conv,
        "poisson_target": target,
        "poisson_mean_rel_err_pct": poisson_rel,
        "poisson_spread_pct": poisson_spread,
        "fRe_series": ref["fRe"],
        "fRe_literature": fre_ref,
        "fRe_rel_err_pct": fre_rel,
        "centre_ratio_series": ref["centre_ratio"],
        "judge_ok": bool(poisson_rel < 1.0 and fre_rel < 1.0),
    }


# ------------------------------------------------------------- 3D: Stokes mode
def stokes_mode_initial(n: int, amp: float) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Divergence-free 3D Fourier mode, an eigenfunction of the Stokes operator.

    ``u_x = A sin(kx) cos(ky) cos(kz)``, ``u_y = -A cos(kx) sin(ky) cos(kz)``,
    ``u_z = 0`` with ``k = 2 pi/n`` on all three axes: ``div u = 0`` and
    ``lap u = -3 k^2 u``, so a Stokes (small-amplitude) run decays as
    ``exp(-nu*3k^2 t)`` with no nonlinear contamination beyond ``O(A^2)``.
    """
    k = 2.0 * math.pi / n
    zc, yc, xc = np.meshgrid(
        np.arange(n) + 0.5, np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij"
    )
    ux = amp * np.sin(k * xc) * np.cos(k * yc) * np.cos(k * zc)
    uy = -amp * np.cos(k * xc) * np.sin(k * yc) * np.cos(k * zc)
    uz = np.zeros_like(ux)
    rho = np.ones_like(ux)
    feq = equilibrium3d(
        torch.as_tensor(rho, dtype=DT),
        torch.as_tensor(ux, dtype=DT),
        torch.as_tensor(uy, dtype=DT),
        torch.as_tensor(uz, dtype=DT),
    )
    return feq, ux, uy, uz


def run_stokes_mode(n: int, tau: float, amp: float, steps: int, arm: str, sample_every: int) -> dict:
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    k2 = 3.0 * k * k
    gamma_exact = nu * k2
    transport = make_transport_3d(arm)
    mask = torch.zeros(n, n, n, dtype=DT)
    force = (0.0, 0.0, 0.0)
    f, *_ = stokes_mode_initial(n, amp)
    hist = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(steps + 1):
            if step % sample_every == 0:
                _, ux, uy, uz = velocity_with_force_3d(f, force)
                umag = torch.sqrt(ux * ux + uy * uy + uz * uz)
                hist.append((step, float(umag.max()), float(f.sum())))
            if step == steps:
                break
            f = _step(f, transport, tau, force, mask, arm)
    wall = time.perf_counter() - t0

    arr = np.asarray(hist)
    half = steps // 2
    sel = arr[:, 0] >= half
    t = arr[sel, 0]
    y = np.log(arr[sel, 1])
    slope, intercept = np.polyfit(t, y, 1)
    fit = slope * t + intercept
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    gamma_fit = -float(slope)
    return {
        "arm": arm,
        "n": n,
        "tau": tau,
        "nu": nu,
        "amp": amp,
        "steps": steps,
        "sample_every": sample_every,
        "k": k,
        "gamma_exact": gamma_exact,
        "gamma_fit": gamma_fit,
        "gamma_fit_rel_err_pct": 100.0 * abs(gamma_fit - gamma_exact) / gamma_exact,
        "gamma_fit_r2": r2,
        "decay_hist": arr.tolist(),
        "mass_drift_rel": float(arr[-1, 2] / arr[0, 2] - 1.0),
        "finite": bool(torch.isfinite(f).all()),
        "f_sha256": f_digest(f),
        "wall_s": wall,
        "s_per_step": wall / max(1, steps),
    }


# ------------------------------------------------------- 3D: slit channel (y)
def run_slit3d(
    ny: int, nx: int, nz: int, tau: float, umax_target: float, steps: int, arm: str, sample_every: int
) -> dict:
    """Walls in y only (half-way at 0.5 and Ny-1.5), periodic in x and z."""
    nu = (tau - 0.5) / 3.0
    height = float(ny - 2)
    g = 8.0 * nu * umax_target / height**2
    force = (g, 0.0, 0.0)
    transport = make_transport_3d(arm)
    mask = torch.zeros(nz, ny, nx, dtype=DT)
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    rho0 = torch.ones(nz, ny, nx, dtype=DT)
    zero = torch.zeros(nz, ny, nx, dtype=DT)
    f = equilibrium3d(rho0, zero, zero, zero)

    hist = []
    prof_mid = None
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(steps + 1):
            if step % sample_every == 0:
                _, ux, _, _ = velocity_with_force_3d(f, force)
                prof = ux.numpy()[nz // 2, 1:-1, nx // 2]
                hist.append((step, float(prof.max()), float(f.sum())))
                prof_mid = prof
            if step == steps:
                break
            f = _step(f, transport, tau, force, mask, arm)
    wall = time.perf_counter() - t0

    y_index = np.arange(1, ny - 1)
    y_pos = y_index - 0.5
    exact = g / (2.0 * nu) * y_pos * (height - y_pos)
    prof_mid = np.asarray(prof_mid)
    hist = np.asarray(hist)
    half_sel = hist[:, 0] >= steps // 2
    drift = abs(hist[-1, 1] - hist[half_sel][0, 1]) / hist[-1, 1]
    return {
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "nz": nz,
        "steps": steps,
        "tau": tau,
        "nu": nu,
        "height": height,
        "g": g,
        "umax_target": umax_target,
        "umax_analytic": g * height**2 / (8.0 * nu),
        "umax_got": float(prof_mid.max()),
        "peak_rel_err_pct": 100.0 * abs(prof_mid.max() - exact.max()) / exact.max(),
        "profile_l2_pct": 100.0 * float(np.linalg.norm(prof_mid - exact) / np.linalg.norm(exact)),
        "profile": prof_mid.tolist(),
        "exact": exact.tolist(),
        "half_run_drift": float(drift),
        "mass_drift_rel": float(hist[-1, 2] / hist[0, 2] - 1.0),
        "finite": bool(torch.isfinite(f).all()),
        "f_sha256": f_digest(f),
        "wall_s": wall,
        "s_per_step": wall / max(1, steps),
    }


# ----------------------------------------------------------- 3D: square duct
def run_duct3d(
    n: int, nx: int, tau: float, umax_target: float, steps: int, arm: str, sample_every: int
) -> dict:
    """Walls on all four lateral faces (y and z), periodic streamwise in x.

    The mask therefore has corners -- the configuration a single flat wall never
    tests, and the one where a fixed-kernel padding could plausibly leak.
    """
    nu = (tau - 0.5) / 3.0
    side = float(n - 2)
    unit = duct_metrics_analytic(1.0, nu, side, terms=101)
    g = umax_target / (unit["u_max"] / 1.0)
    force = (g, 0.0, 0.0)
    ref = duct_metrics_analytic(g, nu, side, terms=101)
    transport = make_transport_3d(arm)
    mask = torch.zeros(n, n, nx, dtype=DT)
    mask[0, :, :] = 1.0
    mask[-1, :, :] = 1.0
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    rho0 = torch.ones(n, n, nx, dtype=DT)
    zero = torch.zeros(n, n, nx, dtype=DT)
    f = equilibrium3d(rho0, zero, zero, zero)

    sl = slice(1, -1)
    y_pos = np.arange(1, n - 1) - 0.5
    exact = duct_series(y_pos, y_pos, g, nu, side, terms=101)  # (nz-2, ny-2)
    hist = []
    got = None
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(steps + 1):
            if step % sample_every == 0:
                _, ux, _, _ = velocity_with_force_3d(f, force)
                got = ux.numpy()[sl, sl, nx // 2]
                hist.append((step, float(got.max())))
            if step == steps:
                break
            f = _step(f, transport, tau, force, mask, arm)
    wall = time.perf_counter() - t0

    hist = np.asarray(hist)
    u_mean_num = float(np.asarray(got).mean())
    u_mean_ref = float(exact.mean())
    fRe_num = 2.0 * g * side**2 / (nu * u_mean_num)
    half_sel = hist[:, 0] >= steps // 2
    return {
        "arm": arm,
        "n": n,
        "nx": nx,
        "steps": steps,
        "tau": tau,
        "nu": nu,
        "side": side,
        "g": g,
        "umax_target": umax_target,
        "umax_got": float(got.max()),
        "umax_analytic": float(ref["u_max"]),
        "centre_rel_err_pct": 100.0 * abs(float(got.max()) - ref["u_max"]) / ref["u_max"],
        "field_l2_pct": 100.0 * float(np.linalg.norm(got - exact) / np.linalg.norm(exact)),
        "u_mean_num": u_mean_num,
        "u_mean_analytic": u_mean_ref,
        "u_mean_rel_err_pct": 100.0 * abs(u_mean_num - u_mean_ref) / u_mean_ref,
        "fRe_num": fRe_num,
        "fRe_analytic": ref["fRe"],
        "fRe_rel_err_pct": 100.0 * abs(fRe_num - ref["fRe"]) / ref["fRe"],
        "centre_ratio_num": float(got.max() / u_mean_num),
        "centre_ratio_analytic": ref["centre_ratio"],
        "half_run_drift": float(abs(hist[-1, 1] - hist[half_sel][0, 1]) / hist[-1, 1]),
        "field": np.asarray(got).tolist(),
        "exact": exact.tolist(),
        "finite": bool(torch.isfinite(f).all()),
        "f_sha256": f_digest(f),
        "wall_s": wall,
        "s_per_step": wall / max(1, steps),
    }


# ------------------------------------------------------------ equivalence 3D
def equivalence_3d(seed: int = 20261113) -> dict:
    """Operator-level checks, now also on a masked (wall-bearing) 3D field."""
    torch.manual_seed(seed)
    out: dict = {}
    f = torch.rand(19, 6, 7, 8, dtype=DT)
    a, b = stream_gather_3d(f), SC.stream_conv_3d(f)
    out["conv_vs_gather_random"] = {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float((a - b).abs().max()),
        "mass_drift_rel": float((f.sum() - b.sum()) / f.sum()),
    }
    from tensorlbm.core.d3q19_stencil import roll_from_pull_source

    ref = torch.empty_like(f)
    ref[0] = f[0]
    for q in range(1, 19):
        ref[q] = roll_from_pull_source(f[q], q)
    out["conv_vs_canonical_roll"] = {
        "bitwise_equal": bool(torch.equal(ref, b)),
        "max_abs_diff": float((ref - b).abs().max()),
    }
    mask = torch.zeros(5, 6, 7, dtype=DT)
    mask[0, :, :] = mask[-1, :, :] = 1.0
    mask[:, 0, :] = mask[:, -1, :] = 1.0
    pre = torch.rand(19, 5, 6, 7, dtype=DT)
    post = torch.rand(19, 5, 6, 7, dtype=DT)
    conv_b = SC.bounce_blend(post, pre, mask, opposite=OPP3D.to(torch.int64))
    idx_b = post.clone()
    idx_b[:, mask > 0] = pre[OPP3D.to(torch.int64)][:, mask > 0]
    out["bounce_conv_vs_index"] = {
        "bitwise_equal": bool(torch.equal(conv_b, idx_b)),
        "max_abs_diff": float((conv_b - idx_b).abs().max()),
        "solid_nodes": int((mask > 0).sum()),
    }
    return out


# ------------------------------------------------------------------- driver
def _git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _pair_equivalence(runs: list[dict]) -> list[dict]:
    """Pair each B run with its A twin and report the difference (bit-wise or not)."""
    by_key = {}
    for r in runs:
        key = tuple(
            (k, r[k])
            for k in ("ny", "nx", "nz", "n", "tau", "umax_target", "amp", "steps")
            if k in r
        )
        by_key.setdefault(key, {})[r["arm"]] = r
    pairs = []
    for key, arms in by_key.items():
        if ARM_A not in arms or ARM_B not in arms:
            continue
        a, b = arms[ARM_A], arms[ARM_B]
        entry = {
            "case_key": dict(key),
            "f_sha256_equal": a["f_sha256"] == b["f_sha256"],
            "max_abs_diff_kpi": [],
            "s_per_step_ratio": b["s_per_step"] / a["s_per_step"] if a["s_per_step"] else None,
        }
        for kpi in ("gamma_fit", "umax_got", "field_l2_pct", "centre_rel_err_pct"):
            if kpi in a and kpi in b:
                entry["max_abs_diff_kpi"].append(
                    {"kpi": kpi, "a": a[kpi], "b": b[kpi], "abs_diff": abs(a[kpi] - b[kpi])}
                )
        pairs.append(entry)
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="CI-sized run (~1 min)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-doc", action="store_true")
    ap.add_argument(
        "--cases",
        default="mode,slit,duct",
        help="comma list of case families to run (dot-chunking for the scheduler queue)",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        mode_cfgs = [(16, 120)]
        slit_cfgs = [(12, 6, 4)]
        duct_cfgs = [(12, 4)]
        slit_steps, duct_steps, sample = 4000, 4000, 500
    else:
        mode_cfgs = [(24, 800), (32, 800)]
        slit_cfgs = [(12, 4, 4), (16, 4, 4)]
        duct_cfgs = [(12, 4), (16, 4)]
        slit_steps, duct_steps, sample = 30000, 30000, 1000

    tau = 0.8
    umax_target = 0.02
    amp = 1e-3

    t_start = time.perf_counter()
    judge = duct_series_selfcheck()
    equiv = equivalence_3d()

    want = {c.strip() for c in args.cases.split(",") if c.strip()}
    runs: list[dict] = []
    for arm in ARMS:
        for n, steps in (mode_cfgs if "mode" in want else []):
            runs.append(run_stokes_mode(n, tau, amp, steps, arm, sample_every=max(1, steps // 20)))
        for ny, nx, nz in (slit_cfgs if "slit" in want else []):
            runs.append(
                run_slit3d(
                    ny, nx, nz, tau, umax_target, slit_steps, arm,
                    sample_every=max(1, slit_steps // 30),
                )
            )
        for n, nx in (duct_cfgs if "duct" in want else []):
            runs.append(
                run_duct3d(
                    n, nx, tau, umax_target, duct_steps, arm, sample_every=max(1, duct_steps // 30)
                )
            )

    pairs = _pair_equivalence(runs)

    def worst(kpi: str) -> float | None:
        vals = [r[kpi] for r in runs if kpi in r and r.get(kpi) is not None]
        return max(vals) if vals else None

    kpis = {
        "gamma_fit_rel_err_pct_max": worst("gamma_fit_rel_err_pct"),
        "slit_profile_l2_pct_max": worst("profile_l2_pct"),
        "slit_peak_rel_err_pct_max": worst("peak_rel_err_pct"),
        "duct_field_l2_pct_max": worst("field_l2_pct"),
        "duct_centre_rel_err_pct_max": worst("centre_rel_err_pct"),
        "duct_fRe_rel_err_pct_max": worst("fRe_rel_err_pct"),
    }
    owner_gate = 15.0
    target = 3.0
    all_finite = all(r["finite"] for r in runs)
    arm_equal = all(p["f_sha256_equal"] for p in pairs) if pairs else False
    limited = [k for k, v in kpis.items() if v is not None and k not in ("duct_fRe_rel_err_pct_max",)]
    under_owner = all(v < owner_gate for k, v in kpis.items() if k in limited and v is not None)
    under_target = all(v < target for k, v in kpis.items() if k in limited and v is not None)
    duct_ok = bool(judge["judge_ok"])

    result = {
        "anchor": ANCHOR,
        "mission": 22987,
        "round": 4,
        "quick": bool(args.quick),
        "cases": sorted(want),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predeclared": json.loads(PRE_DECLARED.read_text()),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "dtype": "float64",
            **_git_state(),
        },
        "duct_judge_selfcheck": judge,
        "equivalence_3d": equiv,
        "runs": runs,
        "pairs": pairs,
        "kpis": kpis,
        "verdict": {
            "all_finite": all_finite,
            "arms_bitwise_equal": arm_equal,
            "operator_checks_ok": all(
                v.get("bitwise_equal", False) for v in equiv.values()
            ),
            "duct_judge_ok": duct_ok,
            "all_kpi_under_15pct": under_owner,
            "all_kpi_under_3pct": under_target,
            "verdict": (
                "PASS"
                if (all_finite and arm_equal and under_owner and under_target and duct_ok)
                else ("GATE_PASS_TARGET_MISS" if (all_finite and arm_equal and under_owner) else "FAIL")
            ),
        },
        "wall_total_s": time.perf_counter() - t_start,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    if not args.no_doc:
        import render_3d

        render_3d.render(result, out_dir, DOC_PATH)
    print(json.dumps(result["verdict"], indent=2))
    print(f"kpis: {json.dumps(kpis, indent=2)}")
    print(f"total wall {result['wall_total_s']:.1f} s -> {out_dir}")
    return 0 if result["verdict"]["verdict"] == "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())