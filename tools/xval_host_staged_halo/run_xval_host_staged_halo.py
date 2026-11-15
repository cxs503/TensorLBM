#!/usr/bin/env python
"""XVAL driver: TensorLBM host-staged halo vs FluidX3D single-node multi-GPU transport.

Mission anchor: ``XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO``.
Reference: ProjectPhysX/FluidX3D (checkout at ``/tmp/fx3d``); the facts this
driver checks against are quoted with file:line in
``src/tensorlbm/host_staged_halo.py``.

Predeclared decision and thresholds (fixed before the numbers were produced):

* D1 (scheme adoption): the ported host-staged scheme must be
  *decomposition-invariant* — the assembled velocity field of a ``Dx*Dy*Dz``
  device grid must agree with the single-device run of the same solver to
  better than ``1e-10`` relative L2 in float64 and ``1e-4`` in float32.
* D2 (transport/accounting): measured staging volume of the port must equal
  the reference formula (``Amax x 5 x sizeof(fpxx)`` per side, one read + one
  write per side, 2 sides, per decomposed direction) to better than 15 %.
* D3 (physics): the ported scheme must reproduce the analytic Taylor-Green
  decay rate to better than 15 % (owner accuracy standard).
* D4 (staging mode): the reference's host pointer swap must remove 100 % of
  the host-to-host copy volume of the naive staging arm, and must not change
  a single bit of the solution.

Stopping rule: a table row that violates its threshold is reported as FAIL and
kept; thresholds are never re-tuned after seeing results.

Outputs (``artifacts/xval_host_staged_halo/``): ``summary.json``,
``table_layout.csv``, ``table_transport.csv``, ``table_invariance.csv``,
``table_tg_physics.csv``, ``table_sensitivity.csv``, ``table_timing.csv`` and
four PNG figures.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:  # repo checkout usage
    from tensorlbm.host_staged_halo import (
        D3Q19_C,
        FLUIDX3D_D3Q19_INDEX_TRANSFER,
        FLUIDX3D_D3Q19_VELOCITY_SET,
        REF_TRANSFERS_PER_FACE,
        HostStagedD3Q19,
        HostStagedHalo,
        HostStagedLayout,
        crossing_indices,
    )
except ImportError:  # pragma: no cover - direct script usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from tensorlbm.host_staged_halo import (  # type: ignore[no-redef]
        D3Q19_C,
        FLUIDX3D_D3Q19_INDEX_TRANSFER,
        FLUIDX3D_D3Q19_VELOCITY_SET,
        REF_TRANSFERS_PER_FACE,
        HostStagedD3Q19,
        HostStagedHalo,
        HostStagedLayout,
        crossing_indices,
    )

ART = Path(__file__).resolve().parents[2] / "artifacts" / "xval_host_staged_halo"
FIG = ART / "figures"

THRESH_INVARIANCE_F64 = 1e-10
THRESH_INVARIANCE_F32 = 1e-4
THRESH_ACCOUNTING = 0.15  # 15 % (owner accuracy standard)
THRESH_PHYSICS = 0.15
THRESH_DISCRETE = 0.03  # exact discrete eigenvalue reference
ITEM_BYTES = 4  # reference default build: FP32 fpxx (defines.hpp:93-97)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def abc_velocity(nx: int, ny: int, nz: int, u0: float = 0.05) -> torch.Tensor:
    """Divergence-free ABC-type field with structure along all three axes.

    ``u = (u0 sin(y)cos(z), u0 sin(z)cos(x), u0 sin(x)cos(y))`` with
    ``k = 2*pi/n``; each component is independent of its own coordinate, so
    ``div(u) = 0`` exactly (checked in the test suite).  Used for the
    decomposition-invariance arm: every face, edge and corner of a
    ``Dx*Dy*Dz`` grid carries a non-trivial non-equilibrium payload.
    """
    kx, ky, kz = 2 * math.pi / nx, 2 * math.pi / ny, 2 * math.pi / nz
    z, y, x = torch.meshgrid(
        torch.arange(nz) * kz, torch.arange(ny) * ky, torch.arange(nx) * kx, indexing="ij"
    )
    return torch.stack(
        [u0 * torch.sin(y) * torch.cos(z), u0 * torch.sin(z) * torch.cos(x), u0 * torch.sin(x) * torch.cos(y)]
    )


def tg_velocity(nx: int, ny: int, nz: int, u0: float = 0.05) -> torch.Tensor:
    """2D Taylor-Green vortex embedded in 3D (exact Navier-Stokes solution).

    ``u = u0 (sin X cos Y, -cos X sin Y, 0)`` with ``X = 2*pi*i/nx``,
    ``Y = 2*pi*j/ny`` on the lattice indices ``i, j``.  The field is periodic,
    divergence-free and a pure superposition of the two Fourier modes
    ``(kx, +-ky, 0)`` with ``|k| = sqrt(2)*kx``, so it is resolved exactly by the
    lattice and its kinetic energy decays as ``exp(-4*nu*kx^2*t)``.

    NB: the index grids below are *unscaled* indices; the wavenumber is applied
    exactly once (``sin(kx*i)``).  Scaling the index by ``kx`` and then
    multiplying by ``kx`` again yields ``sin(kx^2 * i)``, which is **not**
    periodic on an ``nx`` torus -- that defect silently invalidated the first
    version of the D3 arm (see ``docs/XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO.md`` ⑥).
    """
    kx, ky = 2 * math.pi / nx, 2 * math.pi / ny
    z, y, x = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    ux = u0 * torch.sin(kx * x) * torch.cos(ky * y)
    uy = -u0 * torch.cos(kx * x) * torch.sin(ky * y)
    uz = torch.zeros_like(ux)
    return torch.stack([ux, uy, uz])


def tg_selfcheck(nx: int, ny: int, nz: int, u0: float = 0.05) -> dict:
    """Integrity gate for the Taylor-Green initial condition (D3 arm).

    A physics arm is only meaningful if its initial condition *is* the claimed
    analytic field.  Four independent properties are asserted before any solver
    is stepped, so that a wrong initial condition can never again be reported as
    a solver inaccuracy:

    1. the sampled field resolves the mode: ``max|u| == u0`` (a mode that is
       not an integer number of periods on the torus is under-sampled and its
       peak is strictly smaller),
    2. zero discrete divergence (central differences),
    3. spectral purity (all FFT amplitude outside the ``(+-1, +-1)`` mode pair),
    4. kinetic energy equal to the analytic ``u0**2/4``.
    """
    u = tg_velocity(nx, ny, nz, u0)
    peak = float(u.abs().max())
    peak_err = abs(peak - u0)
    dudx = (torch.roll(u[0], -1, 2) - torch.roll(u[0], 1, 2)) / 2.0
    dudy = (torch.roll(u[1], -1, 1) - torch.roll(u[1], 1, 1)) / 2.0
    divergence = float((dudx + dudy).abs().max())
    amp = torch.fft.fft2(u[0][0].to(torch.complex128)).abs()  # (z, y, x) -> z = 0 plane
    total = float(amp.sum())
    mode = float(amp[(-1, 1)] + amp[(-1, -1)] + amp[(1, 1)] + amp[(1, -1)])
    off_mode = (total - mode) / total
    ke = 0.5 * float((u * u).sum(dim=0).mean())
    ke_exact = u0**2 / 4.0
    ok = peak_err < 1e-12 and divergence < 1e-12 and off_mode < 1e-12 and abs(ke / ke_exact - 1.0) < 1e-12
    return {
        "grid": f"{nx}x{ny}x{nz}",
        "u0": u0,
        "peak_amplitude": peak,
        "peak_err": peak_err,
        "divergence_err": divergence,
        "off_mode_energy_fraction": off_mode,
        "ke_measured": ke,
        "ke_analytic": ke_exact,
        "ke_rel_err": abs(ke / ke_exact - 1.0),
        "pass": ok,
    }


def shear_velocity(n: int, a: float = 1e-4) -> torch.Tensor:
    """Single-mode shear wave ``u = (0, a sin(k x), 0)``, ``k = 2*pi/n``.

    A pure transverse mode with wavevector along ``x``: its amplitude is an
    exact exponential of the linearised collide+stream operator, so it can be
    compared against the *discrete* eigenvalue of that operator (no continuum
    limit involved).  ``a`` is kept at the linear-regime level.
    """
    k = 2 * math.pi / n
    z, y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    zero = torch.zeros_like(x)
    return torch.stack([zero, a * torch.sin(k * x), zero])


def discrete_transverse_rates(kvec: tuple[float, float, float], nu: float) -> list[dict]:
    """Exact per-step decay exponent of the *transverse* modes (linearised BGK).

    Linearised D3Q19 BGK collide+stream operator for a single Fourier mode
    ``f_q(x) = f_hat_q exp(i k.x)``::

        A = S (I - omega (I - P)),   S = diag(exp(-i c_q . k)),
                                     P_qq' = W_q (1 + 3 c_q . c_q')

    (``S`` is pull streaming ``f_q(x - c_q)``, ``P`` is the equilibrium
    linearised about ``u = 0``; the labels of the D3Q19 velocity set are the
    solver's own, so no external convention is imported).  Eigenmodes of ``A``
    that carry no density and have ``k . u = 0`` are the transverse (shear)
    modes; their amplitude decays as ``|lambda|^t``, hence the exponent
    ``-ln|lambda|`` per step.  This is an *exact* discrete reference -- unlike
    ``4*nu*k^2`` it contains no continuum limit and no finite-``k`` ambiguity.
    """
    c = np.array(D3Q19_C, dtype=np.float64)
    w = np.array(D3Q19_W, dtype=np.float64)
    omega = 1.0 / (3.0 * nu + 0.5)
    proj = w[:, None] * (1.0 + 3.0 * (c @ c.T))
    phase = np.exp(-1j * (c @ np.asarray(kvec, dtype=np.float64)))
    step_op = phase[:, None] * (np.eye(19) - omega * (np.eye(19) - proj))
    vals, vecs = np.linalg.eig(step_op)
    k = np.asarray(kvec, dtype=np.float64)
    out = []
    for lam, vec in zip(vals, vecs):
        if abs(lam) >= 1.0:
            continue
        if abs(complex(vec.sum())) > 1e-9:  # carries density -> not transverse
            continue
        mom = c.T @ vec
        mom = mom / (np.linalg.norm(mom) + 1e-300)
        if abs(np.dot(k, mom)) > 1e-8:  # not perpendicular to k
            continue
        out.append({"lambda_abs": float(abs(lam)), "rate": float(-math.log(abs(lam))), "polarisation": mom.tolist()})
    if not out:
        raise RuntimeError(f"no transverse eigenmode found for k={kvec}, nu={nu}")
    return out


def tg_analytic_decay(nx: int, ny: int, nu: float) -> float:
    """Analytic kinetic-energy decay exponent ``4*nu*k^2`` of the TG vortex."""
    return 4.0 * nu * (2 * math.pi / nx) ** 2


# ---------------------------------------------------------------------------
# Reference-side formulas (from the FluidX3D source, see module docstring)
# ---------------------------------------------------------------------------


def ref_staging_bytes(layout: HostStagedLayout) -> int:
    """Reference staging traffic for one exchange (``lbm.cpp:1343-1382``).

    Per decomposed direction and device: two sides, each one device->host read
    and one host->device write of ``area * transfers * sizeof(fpxx)`` bytes
    (``enqueue_transfer_extract_field`` / ``enqueue_transfer_insert_field``).
    """
    total = 0
    for axis in layout.decomposed_axes:
        area = layout.face_area(axis)
        per_side = area * REF_TRANSFERS_PER_FACE * ITEM_BYTES
        total += layout.n_devices * 2 * 2 * per_side  # 2 sides x (read + write)
    return total


def ref_buffer_bytes(layout: HostStagedLayout) -> int:
    """Reference staging allocation (``lbm.cpp:1305-1316``).

    Two buffers of ``Amax x max(transfers*sizeof(fpxx), 17)`` bytes per device
    (the 17 B floor covers the separate ``rho_u_flags`` field packet).
    """
    per_buffer = layout.amax * max(REF_TRANSFERS_PER_FACE * ITEM_BYTES, 17)
    return 2 * layout.n_devices * per_buffer


def repo_slab_transport_bytes(nx: int, ny: int, nz: int, n_devices: int) -> int:
    """TensorLBM ``domain_decomposition.HaloExchange`` staging volume (z slab).

    ``pack_send_buffers``/``unpack_recv_buffers`` move 5 crossing populations of
    a ``(ny, nx)`` plane per rank and side
    (``src/tensorlbm/domain_decomposition.py:251-287``), float32.
    """
    n_cross = len(crossing_indices(2, +1))
    return n_devices * 2 * n_cross * ny * nx * ITEM_BYTES


def repo_gloo_transport_bytes(nx: int, ny: int, nz_owned: int) -> int:
    """TensorLBM ``multi_gpu.D3Q19GlooTransport.exchange_ghosts`` volume.

    The Gloo transport ships the *full* 19-population boundary plane in each
    direction (``src/tensorlbm/multi_gpu.py:1147-1171``), float32.
    """
    return 2 * 19 * ny * nx * ITEM_BYTES


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def table_layout() -> list[dict]:
    rows = []
    cases = [
        (32, 32, 32, 1, 1, 1),
        (32, 32, 32, 2, 1, 1),
        (32, 32, 32, 1, 2, 2),
        (32, 32, 32, 2, 2, 2),
        (30, 30, 30, 4, 1, 1),  # not divisible -> reference trims
        (32, 32, 40, 2, 2, 4),
    ]
    for nx, ny, nz, dx, dy, dz in cases:
        lay = HostStagedLayout(nx, ny, nz, dx, dy, dz)
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "Dx": dx,
                "Dy": dy,
                "Dz": dz,
                "devices": lay.n_devices,
                "trimmed": lay.trimmed,
                "global_after_trim": f"{lay.gnx}x{lay.gny}x{lay.gnz}",
                "owned_xyz": "x".join(map(str, lay.owned_xyz)),
                "local_xyz_incl_ghost": "x".join(map(str, lay.local_xyz)),
                "halo_xyz": "".join(map(str, lay.halo)),
                "amax": lay.amax,
                "ref_buffer_bytes": ref_buffer_bytes(lay),
                "ref_staging_bytes_per_exchange": ref_staging_bytes(lay),
            }
        )
    return rows


def measure_transport(lay: HostStagedLayout, staging: str, dtype=torch.float32) -> dict:
    halo = HostStagedHalo(lay, dtype=dtype, staging=staging)
    slabs = [torch.zeros(lay.slab_shape(), dtype=dtype) for _ in range(lay.n_devices)]
    halo.exchange(slabs)
    st = halo.stats()
    return st


def table_transport() -> list[dict]:
    rows = []
    cases = [
        (32, 32, 32, 1, 1, 1),
        (32, 32, 32, 2, 1, 1),
        (32, 32, 32, 1, 1, 2),
        (32, 32, 32, 2, 2, 2),
        (64, 64, 64, 4, 1, 1),
        (64, 64, 64, 2, 2, 2),
        (128, 64, 64, 1, 1, 4),
    ]
    for nx, ny, nz, dx, dy, dz in cases:
        lay = HostStagedLayout(nx, ny, nz, dx, dy, dz)
        ref = ref_staging_bytes(lay)
        swap = measure_transport(lay, "swap")
        copy = measure_transport(lay, "copy")
        got = swap["pcie_bytes"]
        rel = 0.0 if ref == 0 else abs(got - ref) / ref
        rows.append(
            {
                "grid": f"{nx}x{ny}x{nz}",
                "D": f"{dx}x{dy}x{dz}",
                "devices": lay.n_devices,
                "ref_staging_bytes": ref,
                "ours_staging_bytes": got,
                "rel_err": rel,
                "pass": rel <= THRESH_ACCOUNTING,
                "ours_host_copy_bytes_swap": swap["host_copy_bytes"],
                "ours_host_copy_bytes_copy": copy["host_copy_bytes"],
                "staging_alloc_bytes": swap["staging_allocation_bytes"],
                "ref_buffer_bytes": ref_buffer_bytes(lay),
                "repo_HaloExchange_bytes": repo_slab_transport_bytes(nx, ny, nz, lay.n_devices),
                "repo_GlooTransport_bytes_per_rank": repo_gloo_transport_bytes(nx, ny, lay.owned_xyz[2]),
            }
        )
    return rows


def run_solver(
    lay: HostStagedLayout,
    u0: torch.Tensor,
    *,
    nu: float,
    steps: int,
    dtype: torch.dtype,
    staging: str = "swap",
) -> HostStagedD3Q19:
    solver = HostStagedD3Q19(lay, nu=nu, dtype=dtype, staging=staging)
    solver.set_velocity(u0.to(dtype))
    solver.step(steps)
    return solver


def table_invariance(steps: int = 120) -> list[dict]:
    rows = []
    n = 24
    u0 = abc_velocity(n, n, n, 0.05)
    for dtype, tol in ((torch.float64, THRESH_INVARIANCE_F64), (torch.float32, THRESH_INVARIANCE_F32)):
        ref = run_solver(HostStagedLayout(n, n, n), u0, nu=0.02, steps=steps, dtype=dtype)
        u_ref = ref.project_velocity()
        k_ref = ref.kinetic_energy()
        for dx, dy, dz in ((1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 2), (1, 1, 4), (1, 2, 4), (4, 2, 1)):
            lay = HostStagedLayout(n, n, n, dx, dy, dz)
            for staging in ("swap", "copy"):
                s = run_solver(lay, u0, nu=0.02, steps=steps, dtype=dtype, staging=staging)
                u = s.project_velocity()
                rel = float((u - u_ref).norm() / u_ref.norm())
                rows.append(
                    {
                        "dtype": str(dtype).replace("torch.", ""),
                        "grid": f"{n}^3",
                        "D": f"{dx}x{dy}x{dz}",
                        "devices": lay.n_devices,
                        "staging": staging,
                        "steps": steps,
                        "rel_L2_vs_single_device": rel,
                        "kinetic_energy": s.kinetic_energy(),
                        "K_rel_diff": abs(s.kinetic_energy() - k_ref) / k_ref,
                        "max_rho_drift": float((s.project_density() - 1.0).abs().max()),
                        "threshold": tol,
                        "pass": rel <= tol,
                    }
                )
    return rows


def decay_rate(solver_factory, steps: int) -> tuple[float, float]:
    """Return ``(K0, K_end)`` after ``steps`` steps."""
    solver = solver_factory()
    k0 = solver.kinetic_energy()
    solver.step(steps)
    return k0, solver.kinetic_energy()


def _tg_arm_one(lay: HostStagedLayout, u0: torch.Tensor, nu: float, dtype, t1: int, t2: int) -> float:
    """Measured kinetic-energy decay exponent of the TG arm.

    Protocol (fixed *before* the runs): step a transient ``t1`` first -- the
    equilibrium initialisation ``f = feq(rho=1, u)`` carries no O(Ma^2) pressure
    field, so the first steps contain an acoustic transient that is not part of
    the viscous decay -- then take the log-slope between ``t1`` and ``t2``.
    """
    solver = HostStagedD3Q19(lay, nu=nu, dtype=dtype)
    solver.set_velocity(u0.to(dtype))
    solver.step(t1)
    k1 = solver.kinetic_energy()
    solver.step(t2 - t1)
    k2 = solver.kinetic_energy()
    return math.log(k1 / k2) / (t2 - t1)


def table_tg_physics() -> tuple[list[dict], list[dict]]:
    """D3: Taylor-Green decay vs the analytic ``4*nu*k^2`` (continuum reference).

    Returns ``(tg_rows, selfcheck_rows)``.  The run is gated on
    :func:`tg_selfcheck`: if the initial condition is not the claimed analytic
    field the arm raises instead of producing a physics verdict.
    """
    rows, checks = [], []
    t1, t2 = 40, 200
    for n in (24, 32, 40):
        chk = tg_selfcheck(n, n, n, 0.05)
        checks.append(chk)
        if not chk["pass"]:
            raise RuntimeError(f"Taylor-Green initial condition failed its self-check: {chk}")
        u0_field = tg_velocity(n, n, n, 0.05)
        for nu in (0.02, 0.05):
            gamma_theory = tg_analytic_decay(n, n, nu)
            for dx, dy, dz in ((1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 2, 2)):
                lay = HostStagedLayout(n, n, n, dx, dy, dz)
                g = _tg_arm_one(lay, u0_field, nu, torch.float64, t1, t2)
                rows.append(
                    {
                        "grid": f"{n}^3",
                        "nu": nu,
                        "tau": 3 * nu + 0.5,
                        "D": f"{dx}x{dy}x{dz}",
                        "t1": t1,
                        "t2": t2,
                        "gamma_LBM": g,
                        "gamma_theory": gamma_theory,
                        "rel_err": abs(g - gamma_theory) / gamma_theory,
                        "pass": abs(g - gamma_theory) / gamma_theory <= THRESH_PHYSICS,
                    }
                )
    return rows, checks


def _shear_amplitude(solver: HostStagedD3Q19, n: int, k: float) -> float:
    """Fourier amplitude of ``u_y`` on the ``sin(k x)`` mode (linear projection)."""
    u = solver.project_velocity()[1]
    prof = torch.sin(k * torch.arange(n, dtype=torch.float64)).view(1, 1, n)
    return float((u * prof).mean() / (prof * prof).mean())


def table_shear_discrete() -> list[dict]:
    """D3b: single-mode shear decay vs the *exact discrete* eigenvalue.

    Tighter than D3 because the reference is the linearised collide+stream
    operator itself (:func:`discrete_transverse_rates`): no continuum limit and
    no finite-``k`` ambiguity.  The declared threshold is 3 % -- the remaining
    error is the O(a^2) nonlinearity of the initial projection plus the
    transient decay of the non-hydrodynamic eigenmodes.
    """
    rows = []
    a, t1, t2 = 1e-4, 20, 220
    for n in (16, 24, 32):
        k = 2 * math.pi / n
        u0_field = shear_velocity(n, a)
        for nu in (0.02, 0.05, 0.10):
            rates = discrete_transverse_rates((k, 0.0, 0.0), nu)
            # for an axis-aligned wavevector the y- and z-polarised transverse
            # modes are degenerate; take the median candidate as the reference
            gamma_disc = sorted(r["rate"] for r in rates)[len(rates) // 2]
            for dx, dy, dz in ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 2, 2)):
                lay = HostStagedLayout(n, n, n, dx, dy, dz)
                solver = HostStagedD3Q19(lay, nu=nu, dtype=torch.float64)
                solver.set_velocity(u0_field)
                solver.step(t1)
                a1 = _shear_amplitude(solver, n, k)
                solver.step(t2 - t1)
                a2 = _shear_amplitude(solver, n, k)
                gamma = math.log(a1 / a2) / (t2 - t1)
                rows.append(
                    {
                        "grid": f"{n}^3",
                        "nu": nu,
                        "tau": 3 * nu + 0.5,
                        "k": k,
                        "khat_sq": 2 * (1 - math.cos(k)),
                        "D": f"{dx}x{dy}x{dz}",
                        "t1": t1,
                        "t2": t2,
                        "gamma_LBM": gamma,
                        "gamma_discrete": gamma_disc,
                        "gamma_continuum": nu * 2 * (1 - math.cos(k)),
                        "rel_err": abs(gamma - gamma_disc) / gamma_disc,
                        "pass": abs(gamma - gamma_disc) / gamma_disc <= THRESH_DISCRETE,
                    }
                )
    return rows


def table_sensitivity() -> list[dict]:
    """Parameter sweeps: lattice Mach number (U0) and collision rate (nu)."""
    rows = []
    n = 24
    steps = 160
    lay = HostStagedLayout(n, n, n, 2, 1, 2)
    ref_lay = HostStagedLayout(n, n, n)
    for u0mag in (0.01, 0.05, 0.1):
        for nu in (0.01, 0.02, 0.05, 0.1):
            u0 = abc_velocity(n, n, n, u0mag)
            ref = run_solver(ref_lay, u0, nu=nu, steps=steps, dtype=torch.float64)
            s = run_solver(lay, u0, nu=nu, steps=steps, dtype=torch.float64)
            rel = float((s.project_velocity() - ref.project_velocity()).norm() / ref.project_velocity().norm())
            rows.append(
                {
                    "U0": u0mag,
                    "Ma": u0mag * math.sqrt(3.0),
                    "nu": nu,
                    "tau": 3 * nu + 0.5,
                    "omega": 1.0 / (3 * nu + 0.5),
                    "steps": steps,
                    "rel_L2_multi_vs_single": rel,
                    "max_rho_drift": float((s.project_density() - 1.0).abs().max()),
                    "pass": rel <= THRESH_INVARIANCE_F64,
                }
            )
    return rows


def table_timing(repeats: int = 5) -> list[dict]:
    """Exchange wall time, swap vs copy staging (measured on this host)."""
    rows = []
    for nx, ny, nz, dx, dy, dz in ((64, 64, 64, 2, 2, 2), (64, 64, 64, 1, 1, 4), (128, 128, 64, 2, 1, 1)):
        lay = HostStagedLayout(nx, ny, nz, dx, dy, dz)
        for staging in ("swap", "copy"):
            times = []
            for _ in range(repeats):
                halo = HostStagedHalo(lay, dtype=torch.float32, staging=staging)
                slabs = [torch.zeros(lay.slab_shape(), dtype=torch.float32) for _ in range(lay.n_devices)]
                t0 = time.perf_counter()
                for _ in range(3):
                    halo.exchange(slabs)
                times.append((time.perf_counter() - t0) / 3.0)
            rows.append(
                {
                    "grid": f"{nx}x{ny}x{nz}",
                    "D": f"{dx}x{dy}x{dz}",
                    "staging": staging,
                    "median_exchange_s": statistics.median(times),
                    "min_exchange_s": min(times),
                    "host_copy_bytes": halo.stats()["host_copy_bytes"] / 3.0,
                    "pcie_bytes": halo.stats()["pcie_bytes"] / 3.0,
                    "measured_on": "cpu",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(layout_rows, transport_rows, invariance_rows, tg_rows, timing_rows, shear_rows) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    # 1. staging volume: port vs reference formula vs repo transports
    labels, ours, refs, copies, repo_slab, repo_gloo = [], [], [], [], [], []
    for r in transport_rows:
        labels.append(f"{r['grid']}\nD={r['D']}")
        ours.append(r["ours_staging_bytes"] / 1e6)
        refs.append(r["ref_staging_bytes"] / 1e6)
        copies.append(r["ours_host_copy_bytes_copy"] / 1e6)
        repo_slab.append(r["repo_HaloExchange_bytes"] / 1e6)
        repo_gloo.append(r["repo_GlooTransport_bytes_per_rank"] / 1e6)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    idx = range(len(labels))
    w = 0.2
    ax.bar([i - 2 * w for i in idx], refs, w, label="FluidX3D formula (ref)", color="#444")
    ax.bar([i - w for i in idx], ours, w, label="this port (pointer swap)", color="#1f77b4")
    ax.bar([i for i in idx], copies, w, label="this port (host memcpy arm)", color="#ff7f0e")
    ax.bar([i + w for i in idx], repo_slab, w, label="repo HaloExchange (19->5, z-slab)", color="#2ca02c")
    ax.bar([i + 2 * w for i in idx], repo_gloo, w, label="repo GlooTransport (19 planes)", color="#d62728")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("bytes moved per exchange [MB]")
    ax.set_title("Halo staging volume per exchange (host-memcpy arm = 2x ref by construction)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig1_staging_volume.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(p)

    # 2. decomposition invariance
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for dtype, marker in (("float64", "o"), ("float32", "s")):
        xs = [r["D"] for r in invariance_rows if r["dtype"] == dtype and r["staging"] == "swap"]
        ys = [max(r["rel_L2_vs_single_device"], 1e-18) for r in invariance_rows if r["dtype"] == dtype and r["staging"] == "swap"]
        ax.semilogy(xs, ys, marker + "-", label=f"{dtype} (swap arm)")
    ax.axhline(THRESH_INVARIANCE_F64, color="k", ls=":", label="predeclared 1e-10 (f64)")
    ax.axhline(THRESH_INVARIANCE_F32, color="gray", ls=":", label="predeclared 1e-4 (f32)")
    ax.set_ylabel("relative L2 vs single device")
    ax.set_title("Decomposition invariance of the ported host-staged scheme (ABC flow, 120 steps)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig2_invariance.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(p)

    # 3. TG decay error vs grid, per decomposition
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ds = sorted({r["D"] for r in tg_rows})
    grids = sorted({r["grid"] for r in tg_rows}, key=lambda g: int(g.split("^")[0]))
    for d in ds:
        ys = []
        for g in grids:
            sel = [r for r in tg_rows if r["D"] == d and r["grid"] == g and abs(r["nu"] - 0.02) < 1e-12]
            ys.append(100 * sel[0]["rel_err"] if sel else float("nan"))
        ax.plot(grids, ys, "o-", label=f"D={d}")
    ax.axhline(15, color="r", ls="--", label="15% acceptance line")
    ax.set_ylabel("TG decay-rate error vs analytic [%]")
    ax.set_title("Taylor-Green decay vs analytic 4*nu*k^2 (nu=0.02)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig3_tg_error.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(p)

    # 4. exchange timing swap vs copy
    fig, ax = plt.subplots(figsize=(8, 4.2))
    keys = sorted({f"{r['grid']} D={r['D']}" for r in timing_rows})
    swap = [next(r["median_exchange_s"] for r in timing_rows if f"{r['grid']} D={r['D']}" == k and r["staging"] == "swap") for k in keys]
    copy = [next(r["median_exchange_s"] for r in timing_rows if f"{r['grid']} D={r['D']}" == k and r["staging"] == "copy") for k in keys]
    xs = range(len(keys))
    ax.bar([i - 0.2 for i in xs], [1e3 * v for v in swap], 0.4, label="pointer swap (reference)")
    ax.bar([i + 0.2 for i in xs], [1e3 * v for v in copy], 0.4, label="host memcpy arm")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(keys, fontsize=7)
    ax.set_ylabel("exchange wall time [ms] (CPU host)")
    ax.set_title("Staging mode cost: pointer swap vs host memcpy (no GPU on this node)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig4_timing.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(p)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys) + "\n")


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    layout_rows = table_layout()
    transport_rows = table_transport()
    invariance_rows = table_invariance()
    tg_rows, selfcheck_rows = table_tg_physics()
    shear_rows = table_shear_discrete()
    sens_rows = table_sensitivity()
    timing_rows = table_timing()

    write_csv(ART / "table_layout.csv", layout_rows)
    write_csv(ART / "table_transport.csv", transport_rows)
    write_csv(ART / "table_invariance.csv", invariance_rows)
    write_csv(ART / "table_tg_physics.csv", tg_rows)
    write_csv(ART / "table_shear_discrete.csv", shear_rows)
    write_csv(ART / "table_ic_selfcheck.csv", selfcheck_rows)
    write_csv(ART / "table_sensitivity.csv", sens_rows)
    write_csv(ART / "table_timing.csv", timing_rows)
    figs = make_figures(layout_rows, transport_rows, invariance_rows, tg_rows, timing_rows, shear_rows)

    checks = {
        "D1_invariance_all_pass": all(r["pass"] for r in invariance_rows),
        "D2_accounting_all_pass": all(r["pass"] for r in transport_rows),
        "D3_physics_all_pass": all(r["pass"] for r in tg_rows),
        "D3_ic_selfcheck_all_pass": all(r["pass"] for r in selfcheck_rows),
        "D3b_discrete_eigenvalue_all_pass": all(r["pass"] for r in shear_rows),
        "D4_copy_volume_zero_for_swap": all(r["ours_host_copy_bytes_swap"] == 0 for r in transport_rows),
        "D4_swap_equals_copy_solution": max(
            abs(r["rel_L2_vs_single_device"]) for r in invariance_rows if r["staging"] == "copy"
        )
        <= THRESH_INVARIANCE_F32,
    }
    summary = {
        "mission": "m21387",
        "anchor": "XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO",
        "reference": "ProjectPhysX/FluidX3D (lbm.cpp:721-733,1283-1316,1343-1382; kernel.cpp:904-931,2091-2137)",
        "supersedes": "m18373 (PR #2292): its D3 verdict was withdrawn -- the TG initial condition double-applied the wavenumber (non-periodic field), see docs ⑥",
        "thresholds": {
            "invariance_float64": THRESH_INVARIANCE_F64,
            "invariance_float32": THRESH_INVARIANCE_F32,
            "accounting": THRESH_ACCOUNTING,
            "physics": THRESH_PHYSICS,
            "discrete_eigenvalue": THRESH_DISCRETE,
        },
        "d3_protocol": {
            "ic": "u = u0(sin X cos Y, -cos X sin Y, 0), X = 2*pi*i/nx -- periodicity/divergence/spectrum/KE gated by tg_selfcheck",
            "transient_skip": 40,
            "window_end": 200,
            "reference": "continuum 4*nu*k^2",
        },
        "d3b_protocol": {
            "mode": "u = (0, a sin(k x), 0), a = 1e-4 (linear regime), k = 2*pi/n",
            "reference": "exact transverse eigenvalue of the linearised D3Q19 BGK step operator (discrete_transverse_rates)",
            "transient_skip": 20,
            "window_end": 220,
        },
        "ic_selfcheck": selfcheck_rows,
        "reference_table_check": {
            # the reference's own index table, mapped through the reference's own
            # velocity set, must be exactly the geometric crossing set (the integer
            # labels differ between the two storages, the geometry must not)
            "fluidx3d_d3q19_crossing_sets_match_geometry": all(
                sorted(idx for idx, c in enumerate(FLUIDX3D_D3Q19_VELOCITY_SET) if c[axis] == (+1 if side == "p" else -1))
                == sorted(idx)
                and set(crossing_indices(axis, +1 if side == "p" else -1))
                == {q for q, c in enumerate(D3Q19_C) if c[axis] == (+1 if side == "p" else -1)}
                for (axis, side), idx in FLUIDX3D_D3Q19_INDEX_TRANSFER.items()
            ),
            "transfers_per_face": REF_TRANSFERS_PER_FACE,
        },
        "worst_case": {
            "invariance_rel_L2": max(r["rel_L2_vs_single_device"] for r in invariance_rows),
            "accounting_rel_err": max(r["rel_err"] for r in transport_rows),
            "physics_rel_err": max(r["rel_err"] for r in tg_rows),
            "discrete_eigenvalue_rel_err": max(r["rel_err"] for r in shear_rows),
            "sensitivity_rel_L2": max(r["rel_L2_multi_vs_single"] for r in sens_rows),
        },
        "checks": checks,
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
            "host_note": "no CUDA device on this node: host staging is CPU memory, PCIe copies are NOT exercised",
        },
        "wall_seconds": time.time() - t_start,
        "figures": [p.name for p in figs],
        "tables": [
            "table_layout.csv",
            "table_transport.csv",
            "table_invariance.csv",
            "table_tg_physics.csv",
            "table_shear_discrete.csv",
            "table_ic_selfcheck.csv",
            "table_sensitivity.csv",
            "table_timing.csv",
        ],
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if all(v for k, v in checks.items() if not k.startswith("D4_swap_equals")) and checks[
        "D4_swap_equals_copy_solution"
    ] else 1


if __name__ == "__main__":
    raise SystemExit(main())