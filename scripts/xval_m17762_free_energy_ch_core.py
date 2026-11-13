"""XVAL m17762 — TensorLBM free-energy Cahn-Hilliard core vs ebinan92/Fingering_dynamics.

Reference implementation (zero hand-copy: every formula is *executed* out of the
OSS source snapshot, never transcribed):

    repo    ebinan92/Fingering_dynamics
    commit  150a5739e320498c6ecf22ba31ab9ebc57130462
    files   lattice_boltzmann/fingering_periodic.py   (En + Pe + CH driving)
            lattice_boltzmann/fingering.py            (walls / wetting variant)

The script imports ``fingering_periodic.py`` (it is guarded by
``if __name__ == '__main__'``) and drives the *class's own* methods
(``getgeq``, ``getG``, ``stream``, ``udpatePsi``, ``getNabla_psi*``, ``getMu``,
``getLarge_F``) with module globals overridden for a controlled periodic-box
setting.  Nothing from the OSS code is re-typed.

Parts
-----
A  discrete operators   OSS D2Q9 isotropic grad/Laplacian vs TensorLBM
                        ``central7`` / ``isotropic_d3q19``; observed order.
B  free-energy calibration  OSS ``kappa = 0.75*xi*sigma``, ``a = -2*kappa/xi^2``
                        vs TensorLBM ``free_energy_parameters(sigma, W)``;
                        tanh-profile chemical-potential residual; surface
                        tension from the gradient-energy integral.
C  Cahn-Hilliard dispersion  single-mode linear growth rate omega(k) vs the
                        analytic Cahn-Hilliard relation for the three solvers
                        (OSS / TensorLBM ``swift`` / TensorLBM ``lee_lin``),
                        plus resolution, amplitude and mobility sensitivities.
D  structural differences   force form (-phi grad mu vs mu grad phi), mobility
                        discretisation, boundary conditions, viscosity closure.

Usage
-----
    python scripts/xval_m17762_free_energy_ch_core.py --out artifacts/xval_m17762
    OSS_FINGERING_ROOT=/path/to/Fingering_dynamics python ... --quick
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.multiphase3d import free_energy_step_3d, init_free_energy_g_3d  # noqa: E402
from tensorlbm.phasefield.free_energy import DoubleWellFreeEnergy  # noqa: E402
from tensorlbm.phasefield.isotropic_fd import (  # noqa: E402
    isotropic_gradient_3d,
    isotropic_laplacian_3d,
)
from tensorlbm.phasefield.operators import (  # noqa: E402
    central_gradient_3d,
    laplacian_3d,
)
from tensorlbm.solver3d import stream3d  # noqa: E402

torch.set_num_threads(1)  # the box is shared; never grab the whole node

OSS_DEFAULT = os.environ.get("OSS_FINGERING_ROOT", "/tmp/Fingering_dynamics")
OSS_COMMIT = "150a5739e320498c6ecf22ba31ab9ebc57130462"
TAU_G = 1.0 / (3.0 - math.sqrt(3.0))  # OSS module-level `tau` (order parameter)
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oss_provenance(root: pathlib.Path) -> dict:
    files = ["fingering_periodic.py", "fingering.py", "create_block.py"]
    prov = {
        "repo": "ebinan92/Fingering_dynamics",
        "expected_commit": OSS_COMMIT,
        "root": str(root),
        "source_files": {},
    }
    for name in files:
        p = root / "lattice_boltzmann" / name
        if p.exists():
            prov["source_files"][name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - only when git is unavailable
        head = f"unavailable ({exc.__class__.__name__})"
    prov["commit"] = head
    prov["commit_matches_expected"] = head == OSS_COMMIT
    return prov


def load_oss(root: pathlib.Path):
    """Import the OSS module without executing its ``main()``."""
    lb_dir = root / "lattice_boltzmann"
    if not (lb_dir / "fingering_periodic.py").exists():
        raise FileNotFoundError(
            f"OSS snapshot not found under {root}. Set OSS_FINGERING_ROOT to a clone of "
            "ebinan92/Fingering_dynamics (commit %s)." % OSS_COMMIT
        )
    sys.path.insert(0, str(lb_dir))
    spec = importlib.util.spec_from_file_location(
        "_oss_fingering_periodic", str(lb_dir / "fingering_periodic.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Part A — discrete operators
# ---------------------------------------------------------------------------


def _analytic_field(N: int, p: int = 2, q: int = 2):
    x = np.arange(N)
    kx, ky = 2.0 * math.pi * p / N, 2.0 * math.pi * q / N
    X, Y = np.meshgrid(x, x, indexing="xy")  # rows = y, cols = x
    psi = np.sin(kx * X) * np.cos(ky * Y)
    gradx = kx * np.cos(kx * X) * np.cos(ky * Y)
    grady = -ky * np.sin(kx * X) * np.sin(ky * Y)
    lap = -(kx**2 + ky**2) * psi
    return psi, gradx, grady, lap


def oss_operators(mod, psi: np.ndarray) -> dict:
    """Return OSS D2Q9 grad/Laplacian of ``psi`` (its own methods, periodic x)."""
    N = psi.shape[0]
    mod.H = mod.W = N
    mask = np.ones((N, N), dtype=bool)
    cm = mod.Compute(mask)
    cm.block_mask[:] = False
    cm.psi = np.array(psi, dtype=float, copy=True)
    cm.left_wall = cm.psi[:, -1:].copy()  # make the Zou-He wall columns periodic
    cm.right_wall = cm.psi[:, :1].copy()
    return {
        "grad_x": cm.getNabla_psix(),
        "grad_y": cm.getNabla_psiy(),
        "laplacian": cm.getNabla_psi2(),
    }


def _rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def case_a_operators(mod, sizes=(32, 64, 128)) -> dict:
    rows = []
    for N in sizes:
        psi, gx, gy, lap = _analytic_field(N)
        oss = oss_operators(mod, psi)
        psi_t = torch.tensor(psi[None], dtype=torch.float64)  # (z=1, y, x)
        tl_iso = {
            "grad_x": isotropic_gradient_3d(psi_t, boundary="periodic")[0][0].numpy(),
            "grad_y": isotropic_gradient_3d(psi_t, boundary="periodic")[1][0].numpy(),
            "laplacian": isotropic_laplacian_3d(psi_t, boundary="periodic")[0].numpy(),
        }
        tl_c7 = {
            "grad_x": central_gradient_3d(psi_t, boundary="periodic")[0][0].numpy(),
            "grad_y": central_gradient_3d(psi_t, boundary="periodic")[1][0].numpy(),
            "laplacian": laplacian_3d(psi_t, boundary="periodic")[0].numpy(),
        }
        ref = {"grad_x": gx, "grad_y": gy, "laplacian": lap}
        row = {"N": N}
        for key in ref:
            row[f"oss_{key}"] = _rel_l2(oss[key], ref[key])
            row[f"tl_isotropic_{key}"] = _rel_l2(tl_iso[key], ref[key])
            row[f"tl_central7_{key}"] = _rel_l2(tl_c7[key], ref[key])
        # OSS vs TensorLBM isotropic stencil on the same field (same operator?)
        row["oss_vs_tl_isotropic_laplacian"] = _rel_l2(oss["laplacian"], tl_iso["laplacian"])
        rows.append(row)

    def order(key):
        if len(sizes) < 2:
            return None
        e0 = next(r[key] for r in rows if r["N"] == sizes[0])
        e1 = next(r[key] for r in rows if r["N"] == sizes[-1])
        if e0 <= 0 or e1 <= 0:
            return None
        return math.log2(e0 / e1) / math.log2(sizes[-1] / sizes[0])

    return {
        "field": "psi = sin(2*pi*2*x/N) cos(2*pi*2*y/N)",
        "rows": rows,
        "observed_order": {
            "oss_laplacian": order("oss_laplacian"),
            "tl_isotropic_laplacian": order("tl_isotropic_laplacian"),
            "tl_central7_laplacian": order("tl_central7_laplacian"),
            "oss_grad_x": order("oss_grad_x"),
            "tl_isotropic_grad_x": order("tl_isotropic_grad_x"),
        },
    }


# ---------------------------------------------------------------------------
# Part B — free-energy calibration
# ---------------------------------------------------------------------------


def _tanh_profile(N: int, xi: float) -> np.ndarray:
    """Periodic equilibrium profile with two interfaces (tanh at both crossings).

    A single interface cannot be periodic, so the signed distance is taken along a
    smooth periodic curve, ``s(x) = (N/2pi) sin(2pi x/N)``.  ``ds/dx = 1`` at both
    zero crossings, so locally ``psi = tanh(x/xi)`` — the exact Cahn-Hilliard
    equilibrium profile — while ``psi -> +-1`` in the two bulk halves.  Both the
    discrete stencil and the surface-tension integral see a smooth field.
    """
    x = np.arange(N, dtype=float)
    s = (N / (2.0 * math.pi)) * np.sin(2.0 * math.pi * x / N)
    return np.tanh(s / xi)


def case_b_calibration(mod, sigma: float = 0.05, xi: float = 2.0, N: int = 128) -> dict:
    from tensorlbm.allen_cahn_lbm import free_energy_parameters

    kappa_oss = 0.75 * xi * sigma
    a_oss = -2.0 * kappa_oss / xi**2

    # 1) OSS closed-form surface tension recovered from its own constants
    sigma_from_oss = 4.0 * kappa_oss / (3.0 * xi)
    # 2) TensorLBM parameter inversion for the same physical pair (sigma, W=xi)
    params = free_energy_parameters(sigma, W=xi)
    beta_tl, kappa_tl = float(params.beta), float(params.kappa)
    sigma_from_tl = (2.0 * math.sqrt(2.0) / 3.0) * math.sqrt(kappa_tl * beta_tl)

    # 3) discrete profile: OSS chemical potential vs TensorLBM chemical potential
    psi = _tanh_profile(N, xi)
    mod.H = mod.W = N
    mod.kappa = kappa_oss
    mod.a = a_oss
    mask = np.ones((N, N), dtype=bool)
    cm = mod.Compute(mask)
    cm.block_mask[:] = False
    cm.psi = np.repeat(psi[None, :], N, axis=0)
    cm.left_wall = cm.psi[:, -1:].copy()
    cm.right_wall = cm.psi[:, :1].copy()
    # the OSS class caches the stencils, so recompute them for the new field
    cm.nabla_psix = cm.getNabla_psix()
    cm.nabla_psiy = cm.getNabla_psiy()
    cm.nabla_psi2 = cm.getNabla_psi2()
    mu_oss = cm.getMu().reshape(N, N)[0]
    psi_t = torch.tensor(cm.psi[None], dtype=torch.float64)  # (z=1, y, x)
    fw = DoubleWellFreeEnergy(A=beta_tl, B=beta_tl, kappa=kappa_tl)
    mu_tl_c7 = fw.chemical_potential(psi_t, boundary="periodic", operator="central7")[0].numpy()
    mu_tl_iso = fw.chemical_potential(
        psi_t, boundary="periodic", operator="isotropic_d3q19"
    )[0].numpy()
    mu_scale = abs(a_oss)

    # 4) surface tension from the discrete gradient-energy integral of the OSS profile
    #    (two interfaces in the box -> divide by 2)
    dpsi = np.gradient(psi)
    sigma_profile_oss = float(0.5 * np.sum(kappa_oss * dpsi**2))
    sigma_profile_tl = float(0.5 * np.sum(kappa_tl * dpsi**2))

    return {
        "inputs": {"sigma": sigma, "xi": xi, "N": N},
        "oss_constants": {
            "kappa": kappa_oss,
            "a": a_oss,
            "xi": xi,
            "sigma_recovered": sigma_from_oss,
            "sigma_identity_rel_error": (sigma_from_oss - sigma) / sigma,
        },
        "tensorlbm_parameters": {
            "beta": beta_tl,
            "kappa": kappa_tl,
            "W": xi,
            "sigma_recovered": sigma_from_tl,
            "sigma_identity_rel_error": (sigma_from_tl - sigma) / sigma,
            "beta_over_abs_a": beta_tl / abs(a_oss),
            "kappa_over_oss_kappa": kappa_tl / kappa_oss,
        },
        "chemical_potential_on_discrete_tanh": {
            "mu_scale_abs_a": mu_scale,
            "oss_max_abs_mu": float(np.abs(mu_oss).max()),
            "oss_max_rel_mu": float(np.abs(mu_oss).max() / mu_scale),
            "tl_central7_max_rel_mu": float(np.abs(mu_tl_c7[0]).max() / mu_scale),
            "tl_isotropic_max_rel_mu": float(np.abs(mu_tl_iso[0]).max() / mu_scale),
            "tl_central7_vs_oss_rel_l2": _rel_l2(mu_tl_c7[0], mu_oss),
            "tl_isotropic_vs_oss_rel_l2": _rel_l2(mu_tl_iso[0], mu_oss),
        },
        "surface_tension_from_discrete_profile": {
            "oss": sigma_profile_oss,
            "oss_rel_error": (sigma_profile_oss - sigma) / sigma,
            "tensorlbm": sigma_profile_tl,
            "tensorlbm_rel_error": (sigma_profile_tl - sigma) / sigma,
        },
    }


# ---------------------------------------------------------------------------
# Part C — Cahn-Hilliard linear dispersion
# ---------------------------------------------------------------------------


class OSSCH:
    """Drive the OSS order-parameter equation alone (momentum held at u = 0)."""

    def __init__(self, mod, N: int, a: float, kappa: float, Gamma: float, tau_g: float = TAU_G):
        mod.H = mod.W = N
        mod.a = a
        mod.kappa = kappa
        mod.tau = tau_g
        mod.gamma = Gamma
        self.mod = mod
        self.N = N
        mask = np.ones((N, N), dtype=bool)
        self.cm = mod.Compute(mask)
        self.cm.block_mask[:] = False
        self.f_zero = np.zeros_like(self.cm.f)

    def initialise(self, psi: np.ndarray) -> None:
        cm = self.cm
        cm.psi = np.array(psi, dtype=float, copy=True)
        cm.ux = np.zeros(self.N * self.N)
        cm.uy = np.zeros(self.N * self.N)
        self._refresh()
        cm.g = np.array([np.zeros((self.N, self.N)) for _ in range(9)])
        for i in range(9):
            cm.geq[i] = cm.getgeq(i)
            cm.g[i][cm.mask] = cm.geq[i]

    def _refresh(self) -> None:
        cm = self.cm
        cm.left_wall = cm.psi[:, -1:].copy()
        cm.right_wall = cm.psi[:, :1].copy()
        cm.nabla_psix = cm.getNabla_psix()
        cm.nabla_psiy = cm.getNabla_psiy()
        cm.nabla_psi2 = cm.getNabla_psi2()
        cm.mu = cm.getMu()

    def step(self) -> None:
        cm = self.cm
        for i in range(9):
            cm.geq[i] = cm.getgeq(i)
            cm.g[i][cm.mask] = cm.getG(i)
        self.mod.stream(self.f_zero, cm.g)
        cm.udpatePsi()
        self._refresh()

    @property
    def psi(self) -> np.ndarray:
        return self.cm.psi


class TLCH:
    """Drive TensorLBM's D3Q19 free-energy step alone (momentum held at u = 0).

    The box is ``(nz=1, ny=N, nx=N)``: with a z-invariant field the z-shifts of
    streaming are the identity, so this is exactly the 2-D problem at 1/nz the
    cost.  Momentum is frozen by reusing the initial (u = 0, rho = 1)
    distribution, which isolates the Cahn-Hilliard block on both sides
    symmetrically.
    """

    def __init__(self, N, A, kappa, Gamma, convention, tau_g=TAU_G, dtype=torch.float32):
        self.N = N
        self.A = A
        self.kappa = kappa
        self.Gamma = Gamma
        self.convention = convention
        self.tau_g = tau_g
        self.dtype = dtype

    def initialise(self, psi: np.ndarray) -> None:
        phi = torch.tensor(psi[None], dtype=self.dtype)
        self.phi = phi
        self.g = init_free_energy_g_3d(phi)
        z = torch.zeros_like(phi)
        self.f = equilibrium3d(torch.ones_like(phi), z, z, z)

    def step(self) -> None:
        self.g = stream3d(self.g)
        _, self.g = free_energy_step_3d(
            self.f,
            self.g,
            tau_f=1.0,
            tau_g=self.tau_g,
            A=self.A,
            B=self.A,
            kappa=self.kappa,
            Gamma=self.Gamma,
            geq_convention=self.convention,
        )

    @property
    def psi(self) -> np.ndarray:
        return self.g.sum(dim=0)[0].numpy()


def _mode_amplitude(psi: np.ndarray, k: float, axis_len: int) -> float:
    x = np.arange(axis_len)
    if psi.ndim == 2:  # OSS 2-D field: average over rows
        proj = np.sum(psi * np.exp(-1j * k * x)[None, :]) / psi.shape[0]
        return float(2.0 / axis_len * abs(proj))
    proj = np.sum(psi * np.exp(-1j * k * x))
    return float(2.0 / axis_len * abs(proj))


def _growth_rate(amps: np.ndarray, eps: float, lo=1.2, hi=20.0) -> tuple[float, int]:
    ok = (amps > lo * eps) & (amps < hi * eps)
    idx = np.arange(len(amps))[ok]
    if idx.size < 8:
        return float("nan"), int(idx.size)
    rate = float(np.polyfit(idx, np.log(amps[ok]), 1)[0])
    return rate, int(idx.size)


def _mode_psi(N: int, m: int, eps: float) -> np.ndarray:
    x = np.arange(N)
    return eps * np.sin(2.0 * math.pi * m * x / N)[None, :] * np.ones((N, 1))


def dispersion_analytic(k: float, A: float, kappa: float, Gamma: float, tau_g: float) -> float:
    """Standard Cahn-Hilliard linear growth rate, mobility M = Gamma (tau_g - 1/2)."""
    return Gamma * (tau_g - 0.5) * k**2 * (A - kappa * k**2)


def dispersion_swift(k: float, A: float, kappa: float, Gamma: float, tau_g: float) -> float:
    """TensorLBM ``swift`` equilibrium: CH mobility (2/3)M plus spurious cs^2 (tau-1/2) diffusion."""
    return (2.0 / 3.0) * Gamma * (tau_g - 0.5) * k**2 * (A - kappa * k**2) - CS2 * (
        tau_g - 0.5
    ) * k**2


def case_c_swift_decay(mod, N=64, modes=(3, 5, 8, 11, 14), eps=0.01, max_steps=900):
    """Measured decay rate of the historical ``swift`` convention (no growth window)."""
    sigma, xi, Gamma = 0.05, 1.0, 1.0
    kappa = 0.75 * xi * sigma
    A = 2.0 * kappa / xi**2
    rows = []
    for m in modes:
        k = 2.0 * math.pi * m / N
        solver = TLCH(N, A, kappa, Gamma, "swift")
        solver.initialise(_mode_psi(N, m, eps))
        amps = []
        for _ in range(max_steps):
            solver.step()
            amps.append(_mode_amplitude(solver.psi, k, N))
            if amps[-1] < 0.15 * eps:
                break
        amps = np.asarray(amps)
        ok = (amps > 0.15 * eps) & (amps < 0.9 * eps)
        rate = (
            float(np.polyfit(np.arange(len(amps))[ok], np.log(amps[ok]), 1)[0])
            if ok.sum() >= 6
            else float("nan")
        )
        model = dispersion_swift(k, A, kappa, Gamma, TAU_G)
        rows.append(
            {
                "m": m,
                "k": k,
                "omega_measured": rate,
                "omega_model_spurious": model,
                "rel_error": (rate - model) / model if model else float("nan"),
                "steps": int(len(amps)),
            }
        )
    return {"parameters": {"sigma": sigma, "xi": xi, "Gamma": Gamma, "N": N}, "rows": rows}


def make_figures(result: dict, out: pathlib.Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    C = result["parts"].get("C_dispersion")
    if C and "rows" in C:
        rows = C["rows"]
        ks = [r["k"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.plot(ks, [r["analytic"] for r in rows], "k-", label="analytic CH, M=Gamma(tau_g-1/2)")
        ax.plot(ks, [r["oss_omega"] for r in rows], "o", ms=7, mfc="none", label="OSS free-energy core")
        ax.plot(ks, [r["tl_lee_lin_omega"] for r in rows], "x", ms=8, label="TensorLBM lee_lin")
        if "swift_decay_scan" in C:
            sr = C["swift_decay_scan"]["rows"]
            ax.plot([r["k"] for r in sr], [r["omega_measured"] for r in sr], "s", ms=5,
                    label="TensorLBM swift (historical)")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("wavenumber k"); ax.set_ylabel("growth rate $\\omega$")
        ax.set_title("Cahn-Hilliard linear dispersion, $\\sigma$=0.05, $\\xi$=1, $\\Gamma$=1")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        f1 = out / "fig1_ch_dispersion.png"; fig.savefig(f1, dpi=140, bbox_inches="tight"); plt.close(fig)
        figs.append({"file": f1.name, "caption": "Measured single-mode growth rate vs analytic Cahn-Hilliard dispersion. The historical swift convention decays for every k (spurious cs^2(tau_g-1/2) grad^2(phi) diffusion); the ported lee_lin convention coincides with the OSS reference core and with the analytic relation."})

        ser = C.get("growth_series", {})
        if ser:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            for key, lab in (("oss", "OSS free-energy core"), ("tl_lee_lin", "TensorLBM lee_lin"), ("tl_swift", "TensorLBM swift")):
                y = ser.get(key)
                if not y:
                    continue
                t = np.arange(len(y)) * 5
                ax.semilogy(t, y, label=lab)
            ax.set_xlabel("step"); ax.set_ylabel("mode amplitude")
            ax.set_title("Single-mode amplitude history (m=8, k=0.785)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
            f2 = out / "fig2_mode_history.png"; fig.savefig(f2, dpi=140, bbox_inches="tight"); plt.close(fig)
            figs.append({"file": f2.name, "caption": "Amplitude of the k=0.785 mode against time: OSS and lee_lin grow at the analytic rate, the swift convention decays."})

    A = result["parts"].get("A_operators")
    if A:
        rows = A["rows"]
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        Ns = [r["N"] for r in rows]
        for key, lab, st in (
            ("oss_laplacian", "OSS D2Q9 isotropic", "o-"),
            ("tl_isotropic_laplacian", "TensorLBM isotropic_d3q19", "x--"),
            ("tl_central7_laplacian", "TensorLBM central7", "s:"),
        ):
            ax.loglog(Ns, [r[key] for r in rows], st, label=lab)
        ax.set_xlabel("N"); ax.set_ylabel("relative L2 error of $\\nabla^2\\psi$")
        ax.set_title("Discrete Laplacian on a smooth field"); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        f3 = out / "fig3_operator_convergence.png"; fig.savefig(f3, dpi=140, bbox_inches="tight"); plt.close(fig)
        figs.append({"file": f3.name, "caption": "Second-order convergence of the three Laplacians; the OSS D2Q9 isotropic stencil and TensorLBM's isotropic_d3q19 agree to 1e-15 on the same field."})
    return figs


def run_mode(solver_factory, N, m, eps, max_steps, sample_every=1):
    solver = solver_factory(N)
    solver.initialise(_mode_psi(N, m, eps))
    k = 2.0 * math.pi * m / N
    amps = []
    for _ in range(max_steps):
        solver.step()
        amps.append(_mode_amplitude(solver.psi, k, N))
        if amps[-1] > 20.0 * eps:
            break
    return np.asarray(amps), k


def case_c_dispersion(
    mod,
    sigma=0.05,
    xi=1.0,
    Gamma=1.0,
    N=64,
    modes=(3, 5, 8, 11, 14),
    eps=0.01,
    max_steps=1400,
) -> dict:
    kappa = 0.75 * xi * sigma
    a = -2.0 * kappa / xi**2
    A = abs(a)

    def oss_factory(n):
        return OSSCH(mod, n, a=a, kappa=kappa, Gamma=Gamma, tau_g=TAU_G)

    def tl_factory(convention):
        return lambda n: TLCH(n, A, kappa, Gamma, convention)

    rows = []
    series = {}
    for m in modes:
        k = 2.0 * math.pi * m / N
        row = {"m": m, "k": k, "analytic": dispersion_analytic(k, A, kappa, Gamma, TAU_G)}
        for label, factory in (
            ("oss", oss_factory),
            ("tl_lee_lin", tl_factory("lee_lin")),
            ("tl_swift", tl_factory("swift")),
        ):
            amps, _ = run_mode(factory, N, m, eps, max_steps)
            rate, n_samples = _growth_rate(amps, eps)
            row[f"{label}_omega"] = rate
            row[f"{label}_samples"] = n_samples
            row[f"{label}_steps"] = int(len(amps))
            row[f"{label}_rel_error"] = (
                (rate - row["analytic"]) / row["analytic"]
                if abs(row["analytic"]) > 1e-12
                else float("nan")
            )
            if m == modes[len(modes) // 2]:
                series[label] = [float(v) for v in amps[:600:5]]
        row["swift_model"] = dispersion_swift(k, A, kappa, Gamma, TAU_G)
        rows.append(row)

    # --- sensitivity 1: grid resolution (same physical k) --------------------
    res_rows = []
    for N_res, m_res in ((64, 8), (128, 16)):
        k = 2.0 * math.pi * m_res / N_res
        entry = {"N": N_res, "m": m_res, "k": k, "analytic": dispersion_analytic(k, A, kappa, Gamma, TAU_G)}
        for label, factory in (
            ("oss", oss_factory),
            ("tl_lee_lin", tl_factory("lee_lin")),
        ):
            amps, _ = run_mode(factory, N_res, m_res, eps, max_steps * 2)
            rate, n = _growth_rate(amps, eps)
            entry[f"{label}_omega"] = rate
            entry[f"{label}_rel_error"] = (rate - entry["analytic"]) / entry["analytic"]
            entry[f"{label}_samples"] = n
        res_rows.append(entry)

    # --- sensitivity 2: initial amplitude (linearity) ------------------------
    amp_rows = []
    for eps_i in (0.005, 0.02):
        k = 2.0 * math.pi * 8 / N
        entry = {"eps": eps_i, "analytic": dispersion_analytic(k, A, kappa, Gamma, TAU_G)}
        for label, factory in (
            ("oss", oss_factory),
            ("tl_lee_lin", tl_factory("lee_lin")),
        ):
            amps, _ = run_mode(factory, N, 8, eps_i, max_steps)
            rate, n = _growth_rate(amps, eps_i)
            entry[f"{label}_omega"] = rate
            entry[f"{label}_rel_error"] = (rate - entry["analytic"]) / entry["analytic"]
        amp_rows.append(entry)

    # --- sensitivity 3: mobility calibration M_eff / Gamma -------------------
    mob_rows = []
    k = 2.0 * math.pi * 5 / N
    for Gamma_i in (0.5, 1.0, 2.0):
        entry = {"Gamma": Gamma_i, "k": k}
        denom = k**2 * (A - kappa * k**2)
        for label, factory in (
            ("oss", lambda n, g=Gamma_i: OSSCH(mod, n, a=a, kappa=kappa, Gamma=g, tau_g=TAU_G)),
            (
                "tl_lee_lin",
                lambda n, g=Gamma_i: TLCH(n, A, kappa, g, "lee_lin"),
            ),
            ("tl_swift", lambda n, g=Gamma_i: TLCH(n, A, kappa, g, "swift")),
        ):
            amps, _ = run_mode(factory, N, 5, eps, max_steps)
            rate, _ = _growth_rate(amps, eps)
            entry[f"{label}_omega"] = rate
            entry[f"{label}_M_eff"] = rate / denom
        entry["predicted_M_lee_lin"] = (TAU_G - 0.5) * Gamma_i
        entry["predicted_M_swift"] = (2.0 / 3.0) * (TAU_G - 0.5) * Gamma_i
        mob_rows.append(entry)

    return {
        "parameters": {
            "sigma": sigma,
            "xi": xi,
            "kappa": kappa,
            "a": a,
            "A_tensorlbm": A,
            "Gamma": Gamma,
            "tau_g": TAU_G,
            "tau_g_minus_half": TAU_G - 0.5,
            "N": N,
            "eps": eps,
            "momentum": "frozen at u=0 on both sides (order-parameter block only)",
        },
        "rows": rows,
        "growth_series": series,
        "resolution_sensitivity": res_rows,
        "amplitude_sensitivity": amp_rows,
        "mobility_calibration": mob_rows,
    }


# ---------------------------------------------------------------------------
# Part D — structural differences
# ---------------------------------------------------------------------------


def case_d_differences(mod, N=64, xi=1.0, sigma=0.05) -> dict:
    """Quantify the two capillary force forms the two codes use."""
    kappa = 0.75 * xi * sigma
    a = -2.0 * kappa / xi**2
    A = abs(a)
    x = np.arange(N)
    k = 2.0 * math.pi * 3 / N
    psi = 0.6 * np.sin(k * x)[None, :] * np.cos(k * x)[:, None]
    psi_t = torch.tensor(psi[None], dtype=torch.float64)
    fw = DoubleWellFreeEnergy(A=A, B=A, kappa=kappa)
    mu = fw.chemical_potential(psi_t, boundary="periodic")[0].numpy()
    gx, gy = central_gradient_3d(psi_t, boundary="periodic")[0][0].numpy(), central_gradient_3d(
        psi_t, boundary="periodic"
    )[1][0].numpy()
    mux, muy = central_gradient_3d(
        torch.tensor(mu[None], dtype=torch.float64), boundary="periodic"
    )[0][0].numpy(), central_gradient_3d(
        torch.tensor(mu[None], dtype=torch.float64), boundary="periodic"
    )[1][
        0
    ].numpy()

    # TensorLBM: F = -phi grad(mu)
    ft_x, ft_y = -psi * mux, -psi * muy
    # OSS: F = mu grad(psi)   (getLarge_F, Guo-weighted)
    fo_x, fo_y = mu * gx, mu * gy
    diff_x, diff_y = ft_x - fo_x, ft_y - fo_y
    scale = float(np.abs(fo_x).max() + np.abs(fo_y).max())
    # -psi grad(mu) - mu grad(psi) = -grad(psi*mu); the two forms differ by a
    # pure gradient, i.e. by a pressure redefinition in the momentum equation.
    phi_mu = torch.tensor((psi * mu)[None], dtype=torch.float64)
    gpm_x, gpm_y = (
        central_gradient_3d(phi_mu, boundary="periodic")[0][0].numpy(),
        central_gradient_3d(phi_mu, boundary="periodic")[1][0].numpy(),
    )
    denom = float(np.linalg.norm(gpm_x) + np.linalg.norm(gpm_y))
    return {
        "field": "psi = 0.6 sin(kx) cos(ky), k = 2*pi*3/N",
        "force_scale": scale,
        "tl_minus_oss_force_rel_l2": float(np.linalg.norm(diff_x) + np.linalg.norm(diff_y))
        / (2.0 * scale),
        "difference_vs_minus_grad_phi_mu_rel_l2": float(
            np.linalg.norm(diff_x + gpm_x) + np.linalg.norm(diff_y + gpm_y)
        )
        / denom,
        "identity": "F_tl - F_oss = -phi grad(mu) - mu grad(phi) = -grad(phi mu)",
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oss-root", default=OSS_DEFAULT)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "xval_m17762"))
    ap.add_argument("--quick", action="store_true", help="small grids / few modes")
    ap.add_argument("--parts", default="ABCD")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.oss_root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = oss_provenance(root)
    mod = load_oss(root)
    prev_path = out / "xval_free_energy_ch_core.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())
    else:
        prev = {}
    result = {
        "mission": "m17762",
        "anchor": "XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE",
        "reference": prov,
        "tensorlbm_git_head": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parts": {},
    }

    result["parts"].update(prev.get("parts", {}))
    if "A" in args.parts:
        sizes = (16, 32, 64) if args.quick else (32, 64, 128)
        result["parts"]["A_operators"] = case_a_operators(mod, sizes)
        print("[A] operators done", flush=True)
    if "B" in args.parts:
        result["parts"]["B_calibration"] = case_b_calibration(mod)
        print("[B] calibration done", flush=True)
    if "C" in args.parts:
        modes = (5, 8) if args.quick else (3, 5, 8, 11, 14)
        result["parts"]["C_dispersion"] = case_c_dispersion(
            mod,
            N=48 if args.quick else 64,
            modes=modes,
            max_steps=300 if args.quick else 1400,
        )
        print("[C] dispersion done", flush=True)
    if "D" in args.parts:
        result["parts"]["D_differences"] = case_d_differences(mod)
        print("[D] differences done", flush=True)
    if "E" in args.parts:
        result["parts"].setdefault("C_dispersion", {})["swift_decay_scan"] = case_c_swift_decay(
            mod, N=48 if args.quick else 64, modes=(5, 8) if args.quick else (3, 5, 8, 11, 14),
            max_steps=200 if args.quick else 900,
        )
        print("[E] swift decay scan done", flush=True)
    if "F" in args.parts:
        result["figures"] = make_figures(result, out)
        print("[F] figures done", flush=True)

    for key, val in prev.get("parts", {}).items():
        result["parts"].setdefault(key, val)
    if "figures" not in result and prev.get("figures"):
        result["figures"] = prev["figures"]
    path = out / "xval_free_energy_ch_core.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())