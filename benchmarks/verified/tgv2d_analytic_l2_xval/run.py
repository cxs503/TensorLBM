#!/usr/bin/env python
"""XVAL: TensorLBM vs the OSS ``analytic-solution-relative-L2-harness`` practice.

Anchor: XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS (round 1, cross-validation).

Reference implementation (zero hand-copying -- read from source)
--------------------------------------------------------------
``github.com/lbm-principles-practice/code`` (MIT, code accompanying Krüger et al.,
*The Lattice Boltzmann Method: Principles and Practice*, Springer 2016):

* ``chapter13/matlab/vortexdecay.m`` -- 2-D Taylor-Green decay driver.  Lattice
  ``NX = 32*scale``, ``tau = 1`` (=> ``nu = (2 tau - 1)/6 = 1/6``),
  ``u_max = 0.04/scale``, ``NSTEPS = 200*scale^2``, sampling every
  ``NMSG = 50*scale^2`` steps, population initialised from the analytic field at
  ``t = 0``; each sample prints ``t, E, L2rho, L2ux, L2uy``.
* ``chapter13/matlab/taylorgreen.m`` -- the closed-form field, incl.
  ``td = 1/(nu (kx^2+ky^2))`` and ``rho = rho0 + 3P``.
* ``chapter13/matlab/equilibrium.m`` -- the D2Q9 equilibrium used for the IC.

Two arms are run at *identical* settings:

* **A -- OSS port**: a line-by-line NumPy translation of ``vortexdecay.m``
  (circshift streaming, same collision algebra, same diagnostic formulas).  It is
  the closest executable stand-in for the MATLAB script.
* **B -- TensorLBM**: this repository's own D2Q9 primitives
  (``tensorlbm.solver.stream`` / ``collide_bgk`` / ``d2q9.equilibrium`` /
  ``d2q9.macroscopic``, float64), with the diagnostics taken from the new harness
  ``tensorlbm.verification.analytic_l2``.

Cross-arm agreement is the XVAL measurement: if the two implementations agree to
round-off the *convention* has been transferred correctly.  Agreement of either
arm with the analytic field (``< 15 %``, owner accuracy standard; ``< 3 %`` is the
in-repo target at usable resolution) is the physical check.

Additional arms
---------------
* OSS parameter ladder (``scale = 1, 2, 4``) reproduced verbatim -> the table the
  MATLAB script would print.
* Grid-refinement study at fixed ``nu``, ``u_max``, ``t/td`` -> observed order.
* Sensitivity: ``tau`` (viscosity), ``u_max`` (Mach / compressibility), and
  sampling time ``t/td``.
* Negative controls ("gate can fail" witnesses): the same TensorLBM field graded
  against (i) the analytic solution of a *wrong* viscosity and (ii) a sampling
  time shifted by one lattice step.  A harness that cannot fail is not a test.

Usage
-----
    python run.py --out result.json --csv table.csv [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.verification.analytic_l2 import (  # noqa: E402
    kinetic_energy,
    relative_l2,
    tgv2d_analytic,
    tgv2d_decay_time,
)

# --------------------------------------------------------------------------- #
# Arm A: faithful NumPy port of chapter13/matlab/vortexdecay.m
# --------------------------------------------------------------------------- #
# Direction order of the MATLAB code: 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1)
# 6:(-1,1) 7:(-1,-1) 8:(1,-1) 9:(0,0)  (index 0 below is MATLAB index 1).
_OSS_CX = np.array([1, 0, -1, 0, 1, -1, -1, 1, 0])
_OSS_CY = np.array([0, 1, 0, -1, 1, 1, -1, -1, 0])
_OSS_W = np.array([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 4 / 9])


def _oss_equilibrium(rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``chapter13/matlab/equilibrium.m``, vectorised over directions."""
    subexp1 = 1.0 - 1.5 * (u * u + v * v)
    cu = _OSS_CX[:, None, None] * u[None] + _OSS_CY[:, None, None] * v[None]
    feq = _OSS_W[:, None, None] * rho[None] * (subexp1[None] + 3.0 * cu + 4.5 * cu * cu)
    return feq


def _oss_stream(f: np.ndarray) -> np.ndarray:
    """``circshift`` streaming of ``vortexdecay.m`` (periodic, F.9 stays put)."""
    fprop = np.empty_like(f)
    for d in range(9):
        fprop[d] = np.roll(f[d], shift=(-_OSS_CY[d], -_OSS_CX[d]), axis=(0, 1))
    return fprop


def _oss_macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    jx = f[0] + f[4] + f[7] - f[2] - f[5] - f[6]
    jy = f[1] + f[4] + f[5] - f[3] - f[6] - f[7]
    return rho, jx / rho, jy / rho


@dataclass
class ArmResult:
    name: str
    nx: int
    tau: float
    u_max: float
    nu: float
    td: float
    nsteps: int
    nmsg: int
    samples: list[dict] = field(default_factory=list)
    final_fields: dict[str, np.ndarray] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "nx": self.nx,
            "tau": self.tau,
            "u_max": self.u_max,
            "nu": self.nu,
            "td": self.td,
            "nsteps": self.nsteps,
            "nmsg": self.nmsg,
            "samples": self.samples,
            "wall_time_s": self.wall_time_s,
        }


def run_oss_port(
    nx: int,
    *,
    tau: float = 1.0,
    u_max: float = 0.02,
    nsteps: int = 800,
    nmsg: int = 200,
    rho0: float = 1.0,
) -> ArmResult:
    """Line-by-line NumPy port of ``chapter13/matlab/vortexdecay.m``."""
    ny = nx
    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, ny, nu)
    t0 = time.perf_counter()

    ana = tgv2d_analytic(0.0, nx, ny, nu, rho0=rho0, u_max=u_max)
    f = _oss_equilibrium(ana.rho, ana.ux, ana.uy)

    out = ArmResult("oss_port", nx, tau, u_max, nu, td, nsteps, nmsg)
    for t in range(1, nsteps + 1):
        fprop = _oss_stream(f)
        rho, u, v = _oss_macroscopic(fprop)
        feq = _oss_equilibrium(rho, u, v)
        f = (1.0 - 1.0 / tau) * fprop + (1.0 / tau) * feq
        if t % nmsg == 0:
            ref = tgv2d_analytic(t, nx, ny, nu, rho0=rho0, u_max=u_max)
            out.samples.append(
                {
                    "t": t,
                    "t_over_td": t / td,
                    "E": kinetic_energy(rho, u, v),
                    "L2rho": relative_l2(rho, ref.rho, ref_base=rho0).rel_l2,
                    # same error, plain normalisation (denominator ||rho_a|| instead of
                    # the O(Ma^2) fluctuation) -- shows the absolute size of the
                    # density error, which the OSS normalisation inflates.
                    "L2rho_plain": relative_l2(rho, ref.rho, ref_base=0.0).rel_l2,
                    "L2ux": relative_l2(u, ref.ux).rel_l2,
                    "L2uy": relative_l2(v, ref.uy).rel_l2,
                }
            )
    out.final_fields = {"rho": rho, "ux": u, "uy": v}
    out.wall_time_s = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- #
# Arm B: TensorLBM primitives
# --------------------------------------------------------------------------- #
def run_tensorlbm(
    nx: int,
    *,
    tau: float = 1.0,
    u_max: float = 0.02,
    nsteps: int = 800,
    nmsg: int = 200,
    rho0: float = 1.0,
    dtype: str = "float64",
    collide: str = "bgk",
) -> ArmResult:
    """Same case on this repository's torch D2Q9 path.

    Uses ``tensorlbm.d2q9.equilibrium/macroscopic`` and ``tensorlbm.solver.stream``
    + ``collide_bgk`` (or ``collide_trt``); diagnostics come from the new harness
    ``tensorlbm.verification.analytic_l2`` so the two arms share one evaluator API
    but not one implementation.
    """
    import torch

    from tensorlbm import solver
    from tensorlbm.d2q9 import equilibrium, macroscopic

    torch_dtype = getattr(torch, dtype)
    torch.set_num_threads(1)

    ny = nx
    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, ny, nu)
    t0 = time.perf_counter()

    ana0 = tgv2d_analytic(0.0, nx, ny, nu, rho0=rho0, u_max=u_max, as_torch=True, dtype=torch_dtype)
    f = equilibrium(ana0.rho, ana0.ux, ana0.uy)

    out = ArmResult(f"tensorlbm_{collide}_{dtype}", nx, tau, u_max, nu, td, nsteps, nmsg)
    for t in range(1, nsteps + 1):
        f = solver.stream(f)
        rho, u, v = macroscopic(f)
        f = solver.collide_bgk(f, tau) if collide == "bgk" else solver.collide_trt(f, tau)
        if t % nmsg == 0:
            ref = tgv2d_analytic(t, nx, ny, nu, rho0=rho0, u_max=u_max)
            rho_np, u_np, v_np = (
                rho.detach().cpu().numpy(),
                u.detach().cpu().numpy(),
                v.detach().cpu().numpy(),
            )
            out.samples.append(
                {
                    "t": t,
                    "t_over_td": t / td,
                    "E": kinetic_energy(rho_np, u_np, v_np),
                    "L2rho": relative_l2(rho_np, ref.rho, ref_base=rho0).rel_l2,
                    "L2rho_plain": relative_l2(rho_np, ref.rho, ref_base=0.0).rel_l2,
                    "L2ux": relative_l2(u_np, ref.ux).rel_l2,
                    "L2uy": relative_l2(v_np, ref.uy).rel_l2,
                }
            )
    out.final_fields = {"rho": rho_np, "ux": u_np, "uy": v_np}
    out.wall_time_s = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- #
# pressure-harmonic diagnostics (what the L2rho residual actually is)
# --------------------------------------------------------------------------- #
def tgv_harmonic_amplitude(rho: np.ndarray, nx: int) -> float:
    """Least-squares amplitude of the ``cos(2 k_x x)`` pressure harmonic.

    The analytic solution's only density structure is ``3P``, whose square-box
    form is ``-0.75 u_max^2 [cos(2 k x) + cos(2 k y)] exp(-2t/td)``.  Projecting
    the numerical field onto that one basis function therefore yields a single
    number whose ratio to the analytic amplitude is immune to the L2 norm's
    denominator problem and to any *symmetric* transient.
    """
    from tensorlbm.verification.analytic_l2 import cell_centred_coordinates

    xx, _ = cell_centred_coordinates(nx, nx)
    k = 2.0 * math.pi / nx
    basis = np.cos(2.0 * k * xx)
    return float(np.sum((rho - 1.0) * basis) / np.sum(basis * basis))


def harmonic_scan(nx: int, *, tau: float = 1.0, u_max: float = 0.04, tstar_max: float = 2.2, every: int = 4) -> dict:
    """Time series of the pressure-harmonic amplitude on one TensorLBM run.

    Used to identify the *acoustic transient* excited by the OSS initial
    condition (``f = f_eq`` of the analytic field, which omits the O(du) and
    O(d rho/dt) non-equilibrium corrections): the harmonic amplitude oscillates
    around the analytic decay with the sound period of the wave vector
    ``(2 k, 0)``, ``T = 2 pi / (c_s * 2 k)``.
    """
    import torch

    from tensorlbm import solver
    from tensorlbm.d2q9 import equilibrium, macroscopic

    torch.set_num_threads(1)
    ny = nx
    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, ny, nu)
    nsteps = max(2, int(round(tstar_max * td)))
    ana0 = tgv2d_analytic(0.0, nx, ny, nu, u_max=u_max, as_torch=True, dtype=torch.float64)
    f = equilibrium(ana0.rho, ana0.ux, ana0.uy)

    rows = []
    for t in range(1, nsteps + 1):
        f = solver.stream(f)
        rho, ux, uy = macroscopic(f)
        f = solver.collide_bgk(f, tau)
        if t % every == 0:
            rho_np = rho.detach().cpu().numpy()
            amp_num = tgv_harmonic_amplitude(rho_np, nx)
            amp_ana = -0.75 * u_max * u_max * math.exp(-2.0 * t / td)
            rows.append({"t": t, "t_over_td": t / td, "amp_num": amp_num, "amp_ana": amp_ana, "ratio": amp_num / amp_ana})

    k = 2.0 * math.pi / nx
    cs = 1.0 / math.sqrt(3.0)
    period = 2.0 * math.pi / (cs * 2.0 * k)
    # dominant oscillation period of the *deviation* from the smooth decay
    ts = np.array([r["t"] for r in rows], float)
    dev = np.array([r["amp_num"] - r["amp_ana"] * 0.0 for r in rows])  # raw amplitude
    detrended = dev - np.array([r["amp_ana"] for r in rows])
    windowed = detrended * np.hanning(len(detrended))
    n_pad = 64 * len(windowed)  # zero padding: period resolution ~ 1/64 sample
    spectrum = np.abs(np.fft.rfft(windowed, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=float(ts[1] - ts[0]))
    peak = int(np.argmax(spectrum[1:]) + 1)
    measured_period = 1.0 / freqs[peak] if freqs[peak] > 0 else float("inf")
    return {
        "nx": nx,
        "nu": nu,
        "td": td,
        "every": every,
        "samples": rows,
        "predicted_acoustic_period": period,
        "measured_deviation_period": measured_period,
    }


def transient_averaged_metrics(nx_list: list[int], *, tau: float = 1.0, u_max: float = 0.04, tstar: float = 0.5, nper: float = 2.0, every: int = 2) -> list[dict]:
    """Transient-robust amplitude of the pressure harmonic, and its convergence.

    The pressure harmonic of the numerical field is measured as a *complex*
    coefficient ``c(t) = <(rho-1) exp(-i 2 k x)>`` (for ``A cos(2kx)`` this gives
    ``A/2``); multiplying by ``exp(+2t/td)`` de-trends the viscous decay, and
    averaging the real part over ``nper`` sound periods of the wave vector
    ``(2k, 0)`` (``T = 2 pi/(c_s 2k)``, the analytic value -- not fitted) removes
    the acoustic transient launched by the equilibrium initial condition.

    ``amp_ratio`` = measured / analytic ``-0.75 u_max^2`` at ``t = 0``.  A
    diffusion-accurate scheme must drive this ratio to 1 under refinement; a
    first-order scheme would not.
    """
    import torch

    from tensorlbm import solver
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.verification.analytic_l2 import cell_centred_coordinates

    torch.set_num_threads(1)
    nu = (2.0 * tau - 1.0) / 6.0
    rows = []
    for nx in nx_list:
        td = tgv2d_decay_time(nx, nx, nu)
        nsteps = max(4, int(round(tstar * td)))
        k = 2.0 * math.pi / nx
        period = 2.0 * math.pi / ((1.0 / math.sqrt(3.0)) * 2.0 * k)
        nwin = max(4, int(round(nper * period)))
        xx, _ = cell_centred_coordinates(nx, nx)
        basis = np.exp(-1j * 2.0 * k * xx)
        denom = np.sum(np.abs(basis) ** 2)

        ana0 = tgv2d_analytic(0.0, nx, nx, nu, u_max=u_max, as_torch=True, dtype=torch.float64)
        f = equilibrium(ana0.rho, ana0.ux, ana0.uy)
        amps, l2rho, l2ux = [], [], []
        for t in range(1, nsteps + 1):
            f = solver.stream(f)
            rho, ux, uy = macroscopic(f)
            f = solver.collide_bgk(f, tau)
            if t % every == 0 and t > nsteps - nwin:
                rho_np = rho.detach().cpu().numpy()
                c = np.sum((rho_np - 1.0) * basis) / denom
                amps.append(2.0 * float(c.real) * math.exp(2.0 * t / td))
                ref = tgv2d_analytic(t, nx, nx, nu, u_max=u_max)
                l2rho.append(relative_l2(rho_np, ref.rho, ref_base=1.0).rel_l2)
                l2ux.append(relative_l2(ux.detach().cpu().numpy(), ref.ux).rel_l2)

        amp_ana = -0.75 * u_max * u_max
        amp = float(np.mean(amps))
        rows.append(
            {
                "nx": nx,
                "nsteps": nsteps,
                "t_over_td": nsteps / td,
                "acoustic_period": period,
                "window_steps": nwin,
                "amp_num": amp,
                "amp_ana": amp_ana,
                "amp_ratio": amp / amp_ana,
                "amp_rel_err_pct": 100.0 * (amp / amp_ana - 1.0),
                "L2rho_period_mean": float(np.mean(l2rho)),
                "L2ux_period_mean": float(np.mean(l2ux)),
            }
        )
    for i in range(1, len(rows)):
        r0, r1 = rows[i - 1], rows[i]
        r0[f"order_amp_to_{r1['nx']}"] = math.log(abs(r0["amp_rel_err_pct"]) / abs(r1["amp_rel_err_pct"])) / math.log(
            r1["nx"] / r0["nx"]
        )
    return rows


def harmonic_convergence(nx_list: list[int], *, tau: float = 1.0, u_max: float = 0.04, tstar: float = 0.5) -> list[dict]:
    """Pressure-harmonic amplitude ratio vs resolution at fixed ``t/td``."""
    import torch

    from tensorlbm import solver
    from tensorlbm.d2q9 import equilibrium, macroscopic

    torch.set_num_threads(1)
    nu = (2.0 * tau - 1.0) / 6.0
    rows = []
    for nx in nx_list:
        td = tgv2d_decay_time(nx, nx, nu)
        nsteps = max(1, int(round(tstar * td)))
        ana0 = tgv2d_analytic(0.0, nx, nx, nu, u_max=u_max, as_torch=True, dtype=torch.float64)
        f = equilibrium(ana0.rho, ana0.ux, ana0.uy)
        for _ in range(nsteps):
            f = solver.stream(f)
            f = solver.collide_bgk(f, tau)
        rho, ux, uy = macroscopic(f)
        amp_num = tgv_harmonic_amplitude(rho.detach().cpu().numpy(), nx)
        amp_ana = -0.75 * u_max * u_max * math.exp(-2.0 * nsteps / td)
        rows.append(
            {
                "nx": nx,
                "nsteps": nsteps,
                "amp_num": amp_num,
                "amp_ana": amp_ana,
                "ratio": amp_num / amp_ana,
                "rel_err_pct": 100.0 * (amp_num / amp_ana - 1.0),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# comparisons / studies
# --------------------------------------------------------------------------- #
def compare_arms(a: ArmResult, b: ArmResult) -> list[dict]:
    """Relative difference of each diagnostic between the two arms."""
    if len(a.samples) != len(b.samples):
        raise ValueError("arms sampled different numbers of times")
    rows = []
    for sa, sb in zip(a.samples, b.samples):
        if sa["t"] != sb["t"]:
            raise ValueError(f"sample times differ: {sa['t']} vs {sb['t']}")
        row = {"t": sa["t"], "t_over_td": sa["t_over_td"]}
        for key in ("E", "L2rho", "L2ux", "L2uy"):
            denom = max(abs(sa[key]), abs(sb[key]), 1e-300)
            row[f"d_{key}_rel"] = abs(sa[key] - sb[key]) / denom
        rows.append(row)
    return rows


def scaled_case(scale: int) -> dict:
    """The OSS ``vortexdecay.m`` parameter ladder, verbatim.

    ``NX = 32*scale``, ``u_max = 0.04/scale``, ``NSTEPS = 200*scale^2``,
    ``NMSG = 50*scale^2``, ``tau = 1``.
    """
    return {
        "scale": scale,
        "nx": 32 * scale,
        "tau": 1.0,
        "u_max": 0.04 / scale,
        "nsteps": 200 * scale * scale,
        "nmsg": 50 * scale * scale,
    }


def convergence_study(nx_list: list[int], *, tau: float = 1.0, u_max: float = 0.04, tstar: float = 0.5):
    """Fixed-viscosity refinement at a fixed fraction of the decay time."""
    nu = (2.0 * tau - 1.0) / 6.0
    rows = []
    for nx in nx_list:
        td = tgv2d_decay_time(nx, nx, nu)
        nsteps = max(1, int(round(tstar * td)))
        res = run_tensorlbm(nx, tau=tau, u_max=u_max, nsteps=nsteps, nmsg=nsteps)
        s = res.samples[-1]
        rows.append(
            {
                "nx": nx,
                "nsteps": nsteps,
                "t_over_td": s["t_over_td"],
                "L2rho": s["L2rho"],
                "L2ux": s["L2ux"],
                "L2uy": s["L2uy"],
            }
        )
    for i in range(1, len(rows)):
        r0, r1 = rows[i - 1], rows[i]
        r0[f"order_L2ux_to_{r1['nx']}"] = math.log(r0["L2ux"] / r1["L2ux"]) / math.log(r1["nx"] / r0["nx"])
        r0[f"order_L2uy_to_{r1['nx']}"] = math.log(r0["L2uy"] / r1["L2uy"]) / math.log(r1["nx"] / r0["nx"])
        r0[f"order_L2rho_to_{r1['nx']}"] = math.log(r0["L2rho"] / r1["L2rho"]) / math.log(r1["nx"] / r0["nx"])
    return rows


def tau_sensitivity(nx: int, taus: list[float], *, u_max: float = 0.04, tstar: float = 0.5):
    rows = []
    for tau in taus:
        nu = (2.0 * tau - 1.0) / 6.0
        td = tgv2d_decay_time(nx, nx, nu)
        nsteps = max(1, int(round(tstar * td)))
        res = run_tensorlbm(nx, tau=tau, u_max=u_max, nsteps=nsteps, nmsg=nsteps)
        s = res.samples[-1]
        rows.append(
            {
                "tau": tau,
                "nu": nu,
                "nsteps": nsteps,
                "L2rho": s["L2rho"],
                "L2ux": s["L2ux"],
                "L2uy": s["L2uy"],
            }
        )
    return rows


def mach_sensitivity(nx: int, u_max_list: list[float], *, tau: float = 1.0, tstar: float = 0.5):
    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, nx, nu)
    nsteps = max(1, int(round(tstar * td)))
    rows = []
    for u_max in u_max_list:
        res = run_tensorlbm(nx, tau=tau, u_max=u_max, nsteps=nsteps, nmsg=nsteps)
        s = res.samples[-1]
        cs = 1.0 / math.sqrt(3.0)
        rows.append(
            {
                "u_max": u_max,
                "Ma": math.sqrt(2.0) * u_max / cs,
                "L2rho": s["L2rho"],
                "L2ux": s["L2ux"],
                "L2uy": s["L2uy"],
            }
        )
    return rows


def sampling_time_scan(nx: int, *, tau: float = 1.0, u_max: float = 0.04, tstar_max: float = 2.0, n_samples: int = 8):
    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, nx, nu)
    nsteps = max(1, int(round(tstar_max * td)))
    nmsg = max(1, nsteps // n_samples)
    res = run_tensorlbm(nx, tau=tau, u_max=u_max, nsteps=nsteps, nmsg=nmsg)
    return res.samples


def negative_controls(nx: int, *, tau_field: float = 1.0, tau_ref: float = 1.2, u_max: float = 0.04, tstar: float = 0.5):
    """Two deliberately wrong gradings -- the harness must reject them."""
    nu_field = (2.0 * tau_field - 1.0) / 6.0
    nu_ref = (2.0 * tau_ref - 1.0) / 6.0
    td = tgv2d_decay_time(nx, nx, nu_field)
    nsteps = max(1, int(round(tstar * td)))
    res = run_tensorlbm(nx, tau=tau_field, u_max=u_max, nsteps=nsteps, nmsg=nsteps)
    t = res.samples[-1]["t"]
    u = res.final_fields["ux"]
    v = res.final_fields["uy"]
    rho = res.final_fields["rho"]

    honest = tgv2d_analytic(t, nx, nx, nu_field, u_max=u_max)
    wrong_nu = tgv2d_analytic(t, nx, nx, nu_ref, u_max=u_max)
    shifted = tgv2d_analytic(t + 1, nx, nx, nu_field, u_max=u_max)

    return {
        "t": t,
        "t_over_td": t / td,
        "honest": {
            "L2rho": relative_l2(rho, honest.rho, ref_base=1.0).rel_l2,
            "L2ux": relative_l2(u, honest.ux).rel_l2,
            "L2uy": relative_l2(v, honest.uy).rel_l2,
        },
        "wrong_viscosity": {
            "tau_field": tau_field,
            "tau_ref": tau_ref,
            "L2rho": relative_l2(rho, wrong_nu.rho, ref_base=1.0).rel_l2,
            "L2ux": relative_l2(u, wrong_nu.ux).rel_l2,
            "L2uy": relative_l2(v, wrong_nu.uy).rel_l2,
        },
        "one_step_shift": {
            "L2rho": relative_l2(rho, shifted.rho, ref_base=1.0).rel_l2,
            "L2ux": relative_l2(u, shifted.ux).rel_l2,
            "L2uy": relative_l2(v, shifted.uy).rel_l2,
        },
    }


def fluid_node_convention_demo(nx: int = 32, *, tau: float = 1.0, u_max: float = 0.04, tstar: float = 0.2):
    """Why the OSS walled benchmarks exclude wall rows from the norm.

    The Taylor-Green field is fully periodic, so there are no wall nodes to drop.
    The demonstration therefore *manufactures* the situation: two rows of the
    saved velocity field are overwritten with garbage (a stand-in for a wall /
    wet-node layer that does not satisfy the interior analytic solution) and the
    norm is evaluated with and without the fluid-node mask.  Without the mask the
    interior accuracy is invisible; with it the interior value is recovered.
    """
    from tensorlbm.verification.analytic_l2 import fluid_mask

    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, nx, nu)
    nsteps = max(1, int(round(tstar * td)))
    res = run_tensorlbm(nx, tau=tau, u_max=u_max, nsteps=nsteps, nmsg=nsteps)
    t = res.samples[-1]["t"]
    ref = tgv2d_analytic(t, nx, nx, nu, u_max=u_max)

    ux = res.final_fields["ux"].copy()
    interior = relative_l2(ux, ref.ux).rel_l2
    ux[0, :] = 0.05 * u_max  # "wall" rows: constant, far from the interior field
    ux[-1, :] = 0.0
    polluted = relative_l2(ux, ref.ux).rel_l2
    mask = fluid_mask(np.zeros_like(ux), extra_excluded_rows=1)
    masked = relative_l2(ux, ref.ux, mask=mask).rel_l2
    return {
        "nx": nx,
        "n_nodes_full": int(ux.size),
        "n_nodes_masked": int(mask.sum()),
        "L2ux_interior": interior,
        "L2ux_with_wall_rows": polluted,
        "L2ux_wall_rows_excluded": masked,
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "result.json"))
    ap.add_argument("--csv", default=str(HERE / "table.csv"))
    ap.add_argument("--quick", action="store_true", help="smaller sweeps for CI")
    args = ap.parse_args(argv)

    import torch

    t_start = time.perf_counter()
    report: dict = {
        "anchor": "XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS",
        "oss_source": "lbm-principles-practice/code chapter13/matlab/{vortexdecay,taylorgreen,equilibrium}.m",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": 1,
        },
    }

    # 1) OSS parameter ladder, both arms, identical settings.
    ladder = []
    for scale in (1, 2) if args.quick else (1, 2, 4):
        case = scaled_case(scale)
        kw = {k: v for k, v in case.items() if k != 'scale'}
        a = run_oss_port(**kw)
        b = run_tensorlbm(**kw)
        ladder.append(
            {
                "case": case,
                "oss_port": a.as_dict(),
                "tensorlbm": b.as_dict(),
                "cross_arm": compare_arms(a, b),
            }
        )
    report["oss_ladder"] = ladder

    # 2) grid refinement at fixed physics
    report["convergence"] = convergence_study([32, 64] if args.quick else [32, 64, 128])

    # 3) sensitivity sweeps
    report["tau_sensitivity"] = tau_sensitivity(64, [0.6, 0.8, 1.0, 1.2] if not args.quick else [0.8, 1.0])
    report["mach_sensitivity"] = mach_sensitivity(64, [0.01, 0.02, 0.04, 0.08])
    report["time_scan"] = sampling_time_scan(64)

    # 4) what the L2rho residual is: acoustic transient from the equilibrium IC
    report["harmonic_transient"] = harmonic_scan(64 if not args.quick else 32)
    report["harmonic_convergence"] = harmonic_convergence([32, 64] if args.quick else [32, 64, 128])
    report["transient_averaged"] = transient_averaged_metrics([16, 32, 64] if args.quick else [16, 32, 64, 128])

    # 5) negative controls + fluid-node convention
    report["negative_controls"] = negative_controls(64)
    report["fluid_node_convention"] = fluid_node_convention_demo(64)

    # 6) verdicts against the owner accuracy standard
    cross_ok = max(
        row[f"d_{k}_rel"]
        for case in report["oss_ladder"]
        for row in case["cross_arm"]
        for k in ("E", "L2rho", "L2ux", "L2uy")
    )
    phys = {
        f"scale{case['case']['scale']}": max(
            max(s["L2rho"], s["L2ux"], s["L2uy"]) for s in case["tensorlbm"]["samples"]
        )
        for case in report["oss_ladder"]
    }
    vel = {
        f"scale{case['case']['scale']}": max(
            max(s["L2ux"], s["L2uy"]) for s in case["tensorlbm"]["samples"]
        )
        for case in report["oss_ladder"]
    }
    den_plain = max(s["L2rho_plain"] for case in report["oss_ladder"] for s in case["tensorlbm"]["samples"])
    conv_orders = [
        v for row in report["convergence"] for k, v in row.items() if k.startswith("order_L2ux")
    ]
    hc = report["harmonic_convergence"]
    report["verdict"] = {
        "cross_arm_max_rel_diff": cross_ok,
        "cross_arm_threshold": 1e-8,
        "cross_arm_pass": bool(cross_ok < 1e-8),
        "max_error_vs_analytic": phys,
        "max_velocity_error_vs_analytic": vel,
        "max_density_plain_normalised_error": den_plain,
        "owner_threshold_15pct_pass": bool(max(phys.values()) < 0.15),
        "target_3pct_velocity_pass": bool(max(vel.values()) < 0.03),
        "target_3pct_all_fields_pass": bool(max(phys.values()) < 0.03),
        "velocity_order_observed": conv_orders,
        "harmonic_ratio_max_abserr_pct": max(abs(r["rel_err_pct"]) for r in hc),
        "transient_averaged_amp_rel_err_pct": {
            f"nx{r['nx']}": r["amp_rel_err_pct"] for r in report["transient_averaged"]
        },
        "transient_averaged_L2rho_at_finest_pct": 100.0 * report["transient_averaged"][-1]["L2rho_period_mean"],
        "wall_time_s": time.perf_counter() - t_start,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # flat CSV of the OSS ladder (the table the MATLAB driver prints)
    csv_path = Path(args.csv)
    lines = ["scale,nx,t,t_over_td,arm,E,L2rho,L2rho_plain,L2ux,L2uy"]
    for case in report["oss_ladder"]:
        sc = case["case"]["scale"]
        for arm in ("oss_port", "tensorlbm"):
            for s in case[arm]["samples"]:
                lines.append(
                    f"{sc},{case['case']['nx']},{s['t']},{s['t_over_td']:.6g},{arm},"
                    f"{s['E']:.12g},{s['L2rho']:.12g},{s['L2rho_plain']:.12g},{s['L2ux']:.12g},{s['L2uy']:.12g}"
                )
    csv_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {out_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())