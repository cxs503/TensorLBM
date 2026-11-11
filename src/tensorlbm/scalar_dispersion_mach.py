"""Lattice-Mach error scaling of passive-scalar shear dispersion (Taylor–Aris slit).

Why this module exists
----------------------
``tensorlbm.taylor_aris`` / ``benchmarks/verified/taylor_aris_dispersion`` validate
Taylor–Aris dispersion with a **second-moment** meter (``sigma_x^2`` growth) in a
periodic box that must be long enough for ``sigma_x <= L/4``.  That meter is
unusable at high Péclet number — the plume spreads as ``sigma_x ~ H Pe sqrt(t)``
so ``L`` would have to grow like ``Pe`` — and the cheapest alternative, the
**modal decay** meter (track the amplitude of one streamwise Fourier mode of the
cross-sectionally averaged field, ``rate = k^2 D_eff``), carries a truncation
error of its own: the Taylor–Aris coefficient is a ``k -> 0`` (long-wave)
quantity, and the meter returns ``D_eff(k)`` which differs from ``D_eff(0)`` by
``O((kH)^2)`` — non-perturbatively large once ``kH`` approaches one.

Measured on the shipped D2Q5 kernel at ``Pe = 40`` (``H`` fluid rows, periodic
``L = span*H``):

===========  ============  ============
``span``     ``kH``        error in ``D_eff``
===========  ============  ============
7.5          0.84          **+7.6 %**
15           0.42          −1.0 %
30           0.21          −2.8 %
60           0.105         −3.2 %
===========  ============  ============

i.e. the whole "+8.7 %" that a single ``span = L/H = 7.5`` run reports is a
finite-``k`` artefact of the meter, not a Mach-number effect of the solver.  The
``k -> 0`` residual (≈ −3.2 % here) *is* a lattice error: it scales with the
local Mach number and is removed by keeping ``u <= 0.05`` (or by the optional
second-order equilibrium ablation, see :func:`equilibrium_scalar_2nd`).

What this module provides
-------------------------
* :func:`slit_velocity` — the sampled parabolic / plug profile across the slit,
  with the *sampled* mean velocity reported (the Péclet number is always formed
  from the sampled mean, never from the nominal one).
* :func:`modal_decay_case` — a complete modal-meter run: D2Q5 advection–diffusion
  with the library kernels (``thermal.temperature_*``) plus half-way zero-flux
  scalar walls, fixed analytic velocity field, one streamwise Fourier mode.
* :func:`extrapolate_k0` / :func:`richardson_order` — the ``(kH)^2 -> 0``
  Richardson limit and its observed convergence order.
* :func:`mach_law_fit` — the OLS law ``error = a + b u^2`` with ``R^2``, i.e. the
  lattice-Mach scaling of the residual.
* :func:`equilibrium_scalar_2nd` — the third-order (~u^2) Hermite scalar
  equilibrium, kept as an ablation: it changes the second moment of the
  populations and therefore the numerical diffusivity.

Definitions used throughout
---------------------------
``Pe = u_sampled * H / D``, ``D = c_s^2 (tau - 1/2) = (tau - 1/2)/3``,
``D_eff/D = 1 + Pe^2/210`` for the plane slit (Taylor 1953, Aris 1956) with the
slit height ``H`` and the cross-section mean velocity ``u_sampled``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .taylor_aris import aris_dispersion_coefficient, taylor_aris_channel
from .thermal import (
    C5,
    W5,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)

__all__ = [
    "slit_velocity",
    "sampled_mean",
    "theory_deff_slit",
    "discrete_aris_deff_slit",
    "equilibrium_scalar_2nd",
    "temperature_collision_2nd",
    "scalar_wall_reflect",
    "modal_decay_case",
    "k0_extrapolation",
    "richardson_order",
    "mach_law_fit",
    "lattice_mach",
]

CS2 = 1.0 / 3.0


# ── geometry / velocity field ────────────────────────────────────────────────
def slit_velocity(
    height: int,
    u_mean: float,
    kind: str = "parabola",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Streamwise velocity of a plane-slit profile, sampled on the ``height`` fluid rows.

    Returns a ``(height,)`` tensor; row ``j`` (0-based inside the fluid) sits at
    ``y = j + 1`` with the zero-flux / no-slip planes half a cell outside at
    ``y = 0.5`` and ``y = height + 0.5``.  ``kind="parabola"`` is the plane
    Poiseuille profile ``1.5 u_mean (1 - ((y-y_c)/h)^2)`` with ``h = height/2``;
    ``kind="plug"`` is the uniform-flow negative control (no shear => no Taylor
    term).
    """
    if kind not in ("parabola", "plug"):
        raise ValueError(f"unknown profile kind: {kind}")
    y = torch.arange(height, dtype=dtype) + 1.0
    if kind == "plug":
        return torch.full((height,), float(u_mean), dtype=dtype)
    y_c = (height + 1.0) / 2.0
    h = height / 2.0
    return 1.5 * u_mean * (1.0 - ((y - y_c) / h) ** 2)


def sampled_mean(profile: torch.Tensor | np.ndarray) -> float:
    """Arithmetic mean of a sampled profile (the only admissible mean for ``Pe``)."""
    p = profile.detach().cpu().numpy() if isinstance(profile, torch.Tensor) else np.asarray(profile)
    return float(p.mean())


def theory_deff_slit(diffusivity: float, height: int, u_sampled: float) -> float:
    """Closed-form Taylor–Aris slit coefficient ``D + (2/105) h^2 u^2 / D``."""
    return taylor_aris_channel(diffusivity, height / 2.0, u_sampled)


def _thomas_aris(u_prime: np.ndarray, diffusivity: float) -> np.ndarray:
    """LAPACK-free solve of the discrete Aris problem ``D laplacian chi = u'``.

    Pure Neumann tridiagonal operator (``chi_{-1} = chi_0``, ``chi_n = chi_{n-1}``);
    the constant null space is removed by pinning ``chi_0 = 0``, which leaves
    ``D_eff = D + <u' chi>`` unchanged (``<u'> = 0``).  Deliberately independent of
    ``numpy.linalg.solve``: m8228 measured garbage solves from the cluster
    loong64 LAPACK for this pinned system at ``n >= 64``.
    """
    n = u_prime.size
    main = np.full(n, -2.0)
    main[0] = -1.0
    main[-1] = -1.0
    lower = np.ones(n)  # lower[0] unused
    upper = np.ones(n)  # upper[-1] unused
    # pin chi_0 = 0
    main[0] = 1.0
    upper[0] = 0.0
    b = -u_prime / diffusivity
    b[0] = 0.0
    # forward elimination
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    diag0 = main[0]
    if diag0 == 0.0:  # pragma: no cover
        raise ZeroDivisionError("singular Thomas pivot")
    cp[0] = upper[0] / diag0
    dp[0] = b[0] / diag0
    for i in range(1, n):
        m = main[i] - lower[i] * cp[i - 1]
        if m == 0.0:  # pragma: no cover
            raise ZeroDivisionError("singular Thomas pivot")
        if i < n - 1:
            cp[i] = upper[i] / m
        dp[i] = (b[i] - lower[i] * dp[i - 1]) / m
    chi = np.zeros(n)
    chi[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        chi[i] = dp[i] - cp[i] * chi[i + 1]
    return chi - chi.mean()


def discrete_aris_deff_slit(
    diffusivity: float, height: int, u_sampled: float, kind: str = "parabola"
) -> dict[str, Any]:
    """Discrete Aris auxiliary problem on the *sampled* profile (no time stepping).

    Removes the "sampled parabola vs continuous parabola" bookkeeping from the
    comparison, leaving the transport discretisation error alone.  The shipped
    library entry :func:`tensorlbm.taylor_aris.aris_dispersion_coefficient` is
    cross-checked against a LAPACK-free Thomas solve; a disagreement larger than
    1e-6 relative flags the platform trap documented in m8228.
    """
    u = slit_velocity(height, u_sampled, kind=kind).numpy()
    out = aris_dispersion_coefficient(u, diffusivity, spacing=1.0)
    up = u - u.mean()
    chi = _thomas_aris(up, diffusivity)
    d_thomas = diffusivity + float(np.mean(up * chi))
    d_lib = float(out["d_eff"])
    out["d_eff_thomas"] = d_thomas
    out["library_vs_thomas_rel"] = abs(d_lib - d_thomas) / d_thomas
    out["solver_trustworthy"] = bool(
        out["library_vs_thomas_rel"] < 1e-6 and float(out.get("residual", 0.0)) < 1e-8
    )
    return out


# ── optional u^2 equilibrium (ablation) ───────────────────────────────────────
def equilibrium_scalar_2nd(T: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q5 scalar equilibrium with the ``O(u^2)`` Hermite terms.

    ``g_eq = w T (1 + 3 c.u + (9/2)(c.u)^2 - (3/2) u^2)``.  The extra terms leave
    the zeroth and first moments untouched (``sum_i w_i c_i = 0``,
    ``sum_i w_i c_ia c_ib (c.u)^2`` is the fourth moment) but shift the second
    moment, hence they act as a controlled perturbation of the numerical
    diffusivity.  Kept for the ablation in the verification driver; the shipped
    kernel remains :func:`tensorlbm.thermal.temperature_equilibrium`.
    """
    c = C5.to(device=T.device, dtype=T.dtype).view(5, 2, 1, 1)
    w = W5.to(device=T.device, dtype=T.dtype).view(5, 1, 1)
    cu = c[:, 0, 0, 0].view(5, 1, 1) * ux + c[:, 1, 0, 0].view(5, 1, 1) * uy
    uu = ux * ux + uy * uy
    return w * T.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)


def temperature_collision_2nd(
    g: torch.Tensor, tau: float, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """BGK collision with :func:`equilibrium_scalar_2nd` (ablation only)."""
    T = g.sum(dim=0)
    geq = equilibrium_scalar_2nd(T, ux, uy)
    return g - (g - geq) / tau


# ── zero-flux scalar walls ───────────────────────────────────────────────────
def scalar_wall_reflect(g: torch.Tensor) -> torch.Tensor:
    """Half-way zero-flux scalar walls on rows ``0`` / ``-1`` (dead wrap pads).

    After the periodic streaming, this wall's own outgoing populations sit on the
    dead rows; reflect them into the first/last fluid row.  ``sum(g)`` is
    conserved bit-for-bit (pure relocation).  Rows ``0`` and ``-1`` are kept at
    zero so the scalar collision (``g_eq(T=0) = 0``) cannot manufacture mass
    there.
    """
    esc_b = g[4, 0, :].clone()
    esc_t = g[3, -1, :].clone()
    g[:, 0, :] = 0.0
    g[:, -1, :] = 0.0
    g[3, 1, :] = esc_b
    g[4, -2, :] = esc_t
    return g


def lattice_mach(u: float) -> float:
    """``Ma = u / c_s`` with ``c_s^2 = 1/3``."""
    return float(abs(u) / math.sqrt(CS2))


# ── the modal meter ─────────────────────────────────────────────────────────
def modal_decay_case(
    pe: float,
    height: int,
    span: float,
    diffusivity: float,
    kind: str = "parabola",
    t_end_factor: float = 2.0,
    settle_relax: float = 6.0,
    amplitude: float = 1.0e-2,
    n_samples: int = 200,
    second_order_equilibrium: bool = False,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one modal-decay dispersion case and return the measured ``D_eff``.

    Parameters
    ----------
    pe :
        Target Péclet number ``u_mean H / D``; the *sampled* mean velocity is
        ``pe * D / height``, and the sampled mean is what enters the data.
    height :
        Number of fluid rows across the slit (walls half a cell outside).
    span :
        Domain length in slit heights, ``L = span * height`` (``nx`` rows).
    diffusivity :
        Scalar diffusivity ``D`` in lattice units; ``tau = 3 D + 1/2``.
    t_end_factor :
        Simulated time in units of ``H^2 / D`` (clipped below by
        ``12 / (pi^2) * H^2/D`` so that at least a dozen transverse relaxation
        times are resolved).
    settle_relax :
        Fit window start, in transverse relaxation times ``H^2/(pi^2 D)``.

    Returns
    -------
    dict with ``deff``, ``deff_theory``, ``err_pct``, ``u_sampled``, ``pe_sampled``,
    ``ma``, ``k``, ``kH``, ``rate``, ``r2``, the decay samples and metadata.
    """
    if height < 4:
        raise ValueError("height must be >= 4 fluid rows")
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    nx = int(round(span * height))
    ny = height + 2
    tau = 3.0 * diffusivity + 0.5
    dtype = torch.float64
    device = torch.device("cpu")

    u_rows = slit_velocity(height, pe * diffusivity / height, kind=kind)
    u_s = sampled_mean(u_rows)
    ux = torch.zeros((ny, nx), dtype=dtype)
    ux[1:-1, :] = u_rows.view(height, 1)
    uy = torch.zeros_like(ux)

    k = 2.0 * math.pi / nx
    x = torch.arange(nx, dtype=dtype)
    T0 = torch.zeros((ny, nx), dtype=dtype)
    # phi = 1 + A cos(kx): a nonzero background keeps the mass audit meaningful.
    T0[1:-1, :] = (1.0 + amplitude * torch.cos(k * x)).view(1, nx).expand(height, nx)
    g = (
        equilibrium_scalar_2nd(T0, ux, uy)
        if second_order_equilibrium
        else temperature_equilibrium(T0, ux, uy)
    )
    mass0 = float(g.sum())

    t_relax = height * height / (math.pi**2 * diffusivity)
    n_steps = int(max(t_end_factor * height * height / diffusivity, 12.0 * t_relax))
    t_settle = int(min(settle_relax * t_relax, 0.5 * n_steps))
    dt = max(1, (n_steps - t_settle) // n_samples)

    collide = temperature_collision_2nd if second_order_equilibrium else temperature_collision
    ts: list[float] = []
    amps: list[float] = []
    z = torch.exp(-1j * k * x.double())
    for step in range(1, n_steps + 1):
        g = collide(g, tau, ux, uy)
        g = temperature_stream(g)
        g = scalar_wall_reflect(g)
        if step >= t_settle and (step - t_settle) % dt == 0:
            field = g.sum(dim=0)
            cbar = field[1:-1, :].mean(dim=0)
            amp = float(torch.abs((cbar.to(torch.complex128) * z).sum()) * 2.0 / nx)
            ts.append(float(step))
            amps.append(amp)

    t = np.asarray(ts, dtype=np.float64)
    a = np.asarray(amps, dtype=np.float64)
    log_a = np.log(a)
    slope, icpt = np.polyfit(t, log_a, 1)
    rate = -float(slope)
    resid = log_a - (slope * t + icpt)
    ss_res = float((resid**2).sum())
    ss_tot = float(((log_a - log_a.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    deff = rate / k**2
    # A uniform (plug) flow has no shear, hence no Taylor enhancement: D_eff == D.
    d_theory = diffusivity if kind == "plug" else theory_deff_slit(diffusivity, height, u_s)
    aris = discrete_aris_deff_slit(diffusivity, height, u_s, kind=kind)

    return {
        "pe_nominal": float(pe),
        "height": int(height),
        "span": float(span),
        "nx": int(nx),
        "ny": int(ny),
        "diffusivity": float(diffusivity),
        "tau": float(tau),
        "kind": kind,
        "second_order_equilibrium": bool(second_order_equilibrium),
        "u_nominal": float(pe * diffusivity / height),
        "u_sampled": u_s,
        "u_max": float(u_rows.abs().max()),
        "pe_sampled": float(u_s * height / diffusivity),
        "ma_sampled": lattice_mach(u_s),
        "ma_max": lattice_mach(float(u_rows.abs().max())),
        "uu_mean": float((u_rows**2).mean()),
        "k": k,
        "kH": float(k * height),
        "n_steps": n_steps,
        "t_settle": t_settle,
        "rate": rate,
        "r2": r2,
        "deff": deff,
        "deff_theory": d_theory,
        "deff_aris_discrete": float(aris["d_eff"]),
        "err_pct": 100.0 * (deff / d_theory - 1.0),
        "err_vs_discrete_aris_pct": 100.0 * (deff / float(aris["d_eff"]) - 1.0),
        "mass_drift_rel": abs(float(g.sum()) - mass0) / abs(mass0),
        "samples": {"t": ts, "amp": amps},
        "profile": profile or {},
    }


# ── analysis helpers ────────────────────────────────────────────────────────
def k0_extrapolation(rows: Sequence[dict[str, Any]], key: str = "err_pct") -> dict[str, Any]:
    """Extrapolate the modal-meter result to ``kH -> 0``.

    The Taylor–Aris coefficient is the long-wave (``k -> 0``) value, while the
    meter returns ``D_eff(k)`` whose truncation is a quadratic function of
    ``(kH)^2`` *inside the asymptotic range*.  The coarsest domain lengths are
    outside that range (measured: at ``Pe = 40`` the ``kH = 0.84`` point deviates
    by ~30 % from the quadratic through ``kH <= 0.42``), so the fit uses the
    **three smallest** ``kH`` values and reports the residuals of every span as
    a consistency diagnostic.  With exactly two spans a straight line in
    ``(kH)^2`` is used (flagged ``linear=True``).
    """
    rows = [r for r in rows if "kH" in r]
    if len(rows) < 2:
        raise ValueError("need at least two spans for a (kH)^2 extrapolation")
    rows = sorted(rows, key=lambda r: float(r["kH"]))
    used = rows[:3]
    x = np.array([float(r["kH"]) ** 2 for r in used])
    y = np.array([float(r[key]) for r in used])
    deg = 2 if len(used) >= 3 else 1
    coef = np.polyfit(x, y, deg)
    intercept = float(coef[-1])
    all_x = np.array([float(r["kH"]) ** 2 for r in rows])
    all_y = np.array([float(r[key]) for r in rows])
    pred = np.polyval(coef, all_x)
    resid = all_y - pred
    return {
        "intercept_k0": intercept,
        "linear": float(coef[-2]) if deg >= 1 else float("nan"),
        "quadratic": float(coef[0]) if deg >= 2 else 0.0,
        "linear_fit": bool(deg == 1),
        "n_used": int(len(used)),
        "kH_used": [float(r["kH"]) for r in used],
        "values_used": [float(v) for v in y],
        "kH_all": [float(r["kH"]) for r in rows],
        "values_all": [float(v) for v in all_y],
        "residuals_all": [float(v) for v in resid],
        "rms": float(np.sqrt(np.mean((y - np.polyval(coef, x)) ** 2))),
        "max_abs_resid_all": float(np.max(np.abs(resid))),
        "excluded_kH": [float(r["kH"]) for r in rows[3:]],
    }


def richardson_order(e_coarse: float, e_mid: float, e_fine: float) -> dict[str, Any]:
    """Observed convergence order from three refinement levels (ratio 2).

    ``p = log2 |(e_coarse - e_mid) / (e_mid - e_fine)|``.  Used here with the
    refinement parameter ``h = kH`` (domain length doubling), where the discrete
    meter error is a quadratic function of ``h^2`` and hence ``p -> 2``.
    """
    num = e_coarse - e_mid
    den = e_mid - e_fine
    if den == 0.0:
        return {"order": float("inf"), "ratio": float("inf"), "defined": False}
    ratio = abs(num / den)
    return {
        "order": float(math.log2(ratio)) if ratio > 0 else float("-inf"),
        "ratio": float(ratio),
        "defined": True,
        "limit_estimate": float(e_fine + (e_fine - e_mid) / (2.0**2 - 1.0)),
    }


def mach_law_fit(rows: Sequence[dict[str, Any]], key: str = "err_pct") -> dict[str, Any]:
    """OLS of ``error`` against ``u_sampled^2`` (the lattice-Mach scaling law).

    Returns the slope, intercept, ``R^2`` and per-point deviation from the fitted
    law.  A lattice-truncation error of the advection–diffusion scheme must give
    ``|intercept| << |slope| u^2`` and ``R^2 -> 1``.
    """
    if len(rows) < 3:
        raise ValueError("need at least three points for the Mach law")
    x = np.array([float(r["u_sampled"]) ** 2 for r in rows])
    y = np.array([float(r[key]) for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    ma2 = np.array([float(r["ma_sampled"]) ** 2 for r in rows])
    slope_ma, intercept_ma = np.polyfit(ma2, y, 1)
    return {
        "slope_per_u2": float(slope),
        "intercept_pct": float(intercept),
        "r2": float(r2),
        "slope_per_ma2": float(slope_ma),
        "intercept_ma_pct": float(intercept_ma),
        "n_points": int(len(rows)),
        "points": [
            {"u_sampled": float(r["u_sampled"]), "ma_sampled": float(r["ma_sampled"]),
             "err_pct": float(v), "fit_pct": float(p)}
            for r, v, p in zip(rows, y, pred)
        ],
        "rms_resid_pct": float(np.sqrt(np.mean((y - pred) ** 2))),
    }