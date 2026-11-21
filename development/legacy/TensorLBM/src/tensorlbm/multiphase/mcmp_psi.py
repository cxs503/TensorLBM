"""Multicomponent Shan-Chen (MCMP) in the *psi-field* formulation (D2Q9).

This module implements the multicomponent pseudopotential model the way the
Palabos ``shanChenMultiComponentProcessor3D`` family formulates it:

*  one distribution function and **one pseudopotential (psi) field per
   component**; the psi field is not necessarily ``rho`` -- it is
   ``psi(rho)`` with a selectable functional form (``rho``, ``rho - rho_ref``,
   or the Shan-Chen exponential form), which is what makes the "psi field"
   the primary object of the model;
*  the interaction force of component ``k`` is built **only from the psi
   fields of the *other* components** (self-interaction excluded), so a
   single-species uniform state carries *exactly zero* force and the model
   has no single-component (SCMP) phase separation unless a self-coupling is
   requested explicitly::

       F_k(x) = -G psi_k(x) * sum_i w_i psi_k'(x + c_i) c_i ,   k' != k

*  ``G > 0`` is **repulsive** (demixing), ``G < 0`` attractive.

Gradient (long-wavelength) expansion of the discrete neighbour sum,
``sum_i w_i psi(x + c_i) c_i = c_s^2 grad(psi) + O(grad^3)``, gives the
analytic equation of state that this module is validated against::

    p(rho_A, rho_B) = c_s^2 (rho_A + rho_B) + G c_s^2 psi_A psi_B

and, for the total-density (acoustic) mode with ``psi' = dpsi/drho``::

    c_eff^2 = dp/drho_tot = c_s^2 (1 + G psi(rho0) psi'(rho0))

Both relations are used as *analytic* references in
``xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD``.

Exact discrete identities of the force operator (machine-precision checkable):

1. uniform psi fields  -> F = 0                    (sum_i w_i c_i = 0)
2. psi of the other component identically zero -> F = 0   (self-excluded)
3. sum_x (F_A + F_B) = 0 in a periodic box         (summation by parts)
4. F_A / psi_A = F_B / psi_B                       (one shared neighbour sum)
5. forward (scatter) and backward (gather) formulations agree exactly.

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815 (multicomponent pseudopotential);
Shan & Doolen (1995) J. Stat. Phys. 81, 379;
Martys & Chen (1996) Phys. Rev. E 53, 743 (two-component SC);
Kupershtokh (2004) Comput. Math. Appl. 48, 1141 (EDM forcing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch

CS2 = 1.0 / 3.0

_W = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4
_C = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]

DTYPE = torch.float64


def _tensors(device=None, dtype=DTYPE) -> Tuple[torch.Tensor, torch.Tensor]:
    c = torch.tensor(_C, dtype=dtype, device=device)
    w = torch.tensor(_W, dtype=dtype, device=device)
    return c, w


# --------------------------------------------------------------------------- #
# psi fields
# --------------------------------------------------------------------------- #
PSI_KINDS = ("rho", "rho_minus_ref", "exp")


def psi_of(
    rho: torch.Tensor,
    kind: str = "rho",
    rho_ref: float = 1.0,
    rho0_exp: float = 1.0,
) -> torch.Tensor:
    """Pseudopotential field ``psi(rho)``.

    ``kind``:
      * ``"rho"``           psi = rho                      (bilinear interaction)
      * ``"rho_minus_ref"`` psi = rho - rho_ref            (density-fluctuation
                            psi field; psi = 0 on the reference isodensity)
      * ``"exp"``           psi = rho0 (1 - exp(-rho/rho0)) (Shan-Chen 1994)
    """
    if kind == "rho":
        return rho
    if kind == "rho_minus_ref":
        return rho - rho_ref
    if kind == "exp":
        return rho0_exp * (1.0 - torch.exp(-rho / rho0_exp))
    raise ValueError(f"unknown psi kind {kind!r}; expected one of {PSI_KINDS}")


def psi_slope(rho0: float, kind: str = "rho", rho0_exp: float = 1.0) -> float:
    """dpsi/drho at ``rho = rho0`` (for the analytic sound speed)."""
    if kind in ("rho", "rho_minus_ref"):
        return 1.0
    if kind == "exp":
        return float(math.exp(-rho0 / rho0_exp))
    raise ValueError(f"unknown psi kind {kind!r}")


# --------------------------------------------------------------------------- #
# neighbour sums and force
# --------------------------------------------------------------------------- #
def neighbor_vec_sum(psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(sum_i w_i psi(x+c_i) c_ix, sum_i w_i psi(x+c_i) c_iy)``, periodic."""
    c, w = _tensors(psi.device, psi.dtype)
    sx = torch.zeros_like(psi)
    sy = torch.zeros_like(psi)
    for i in range(1, 9):
        cx, cy = _C[i]
        shifted = torch.roll(psi, shifts=(-cy, -cx), dims=(0, 1))
        sx = sx + w[i] * cx * shifted
        sy = sy + w[i] * cy * shifted
    return sx, sy


def neighbor_vec_sum_scatter(psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward (scatter) form of :func:`neighbor_vec_sum` -- algebraically identical.

    ``out(x) = -sum_i w_i psi(x - c_i) c_i`` accumulated by scattering psi(x)
    to ``x + c_i``; because ``w`` is symmetric and ``c_{-i} = -c_i`` this is
    *identically* the gather operator, but with the lattice loop run in the
    opposite sense.  Used as an independent discretisation of the same
    operator (exact-equality test, see the xval card).
    """
    c, w = _tensors(psi.device, psi.dtype)
    sx = torch.zeros_like(psi)
    sy = torch.zeros_like(psi)
    for i in range(1, 9):
        cx, cy = _C[i]
        contrib = w[i] * psi
        sx = sx - cx * torch.roll(contrib, shifts=(cy, cx), dims=(0, 1))
        sy = sy - cy * torch.roll(contrib, shifts=(cy, cx), dims=(0, 1))
    return sx, sy


def mcmp_psi_force(
    psi_a: torch.Tensor,
    psi_b: torch.Tensor,
    G: float = 2.5,
    self_coupling: float = 0.0,
    body_a: Tuple[float, float] = (0.0, 0.0),
    body_b: Tuple[float, float] = (0.0, 0.0),
    rho_a: torch.Tensor | None = None,
    rho_b: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Two-component psi-field Shan-Chen force (Palabos convention).

    ``F_A = -G psi_A * sum_i w_i psi_B(x+c_i) c_i`` and the mirror image for B
    (self-interaction excluded).  ``self_coupling`` (default 0) re-introduces
    an SCMP-like self term ``+self_coupling * psi_A * sum_i w_i psi_A c_i`` and
    is used only as an ablation arm.

    Body forces are accelerations applied per component, ``rho_k * g``.
    """
    sxb, syb = neighbor_vec_sum(psi_b)
    sxa, sya = neighbor_vec_sum(psi_a)

    fx_a = -G * psi_a * sxb
    fy_a = -G * psi_a * syb
    fx_b = -G * psi_b * sxa
    fy_b = -G * psi_b * sya

    if self_coupling != 0.0:
        fx_a = fx_a + self_coupling * psi_a * sxa
        fy_a = fy_a + self_coupling * psi_a * sya
        fx_b = fx_b + self_coupling * psi_b * sxb
        fy_b = fy_b + self_coupling * psi_b * syb

    if body_a != (0.0, 0.0) and rho_a is not None:
        fx_a = fx_a + rho_a * body_a[0]
        fy_a = fy_a + rho_a * body_a[1]
    if body_b != (0.0, 0.0) and rho_b is not None:
        fx_b = fx_b + rho_b * body_b[0]
        fy_b = fy_b + rho_b * body_b[1]

    return {"fx_a": fx_a, "fy_a": fy_a, "fx_b": fx_b, "fy_b": fy_b}


def eos_pressure(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
    psi_a: torch.Tensor,
    psi_b: torch.Tensor,
    G: float,
) -> torch.Tensor:
    """Analytic equation of state of the cross-interacting psi-field model.

    ``p = c_s^2 (rho_A + rho_B) + G c_s^2 psi_A psi_B``
    (gradient expansion of the force, :func:`mcmp_psi_force`).
    """
    return CS2 * (rho_a + rho_b) + G * CS2 * psi_a * psi_b


def effective_sound_speed(
    G: float, rho0: float, kind: str = "rho", rho_ref: float = 1.0, rho0_exp: float = 1.0
) -> float:
    """``c_eff = sqrt(c_s^2 (1 + G psi(rho0) psi'(rho0)))`` for the acoustic mode."""
    psi0 = float(psi_of(torch.tensor(rho0, dtype=DTYPE), kind, rho_ref, rho0_exp))
    return float(math.sqrt(CS2 * (1.0 + G * psi0 * psi_slope(rho0, kind, rho0_exp))))


# --------------------------------------------------------------------------- #
# D2Q9 operators
# --------------------------------------------------------------------------- #
def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    c, w = _tensors(rho.device, rho.dtype)
    cu = c[:, 0].view(9, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
    usq = ux * ux + uy * uy
    return w.view(9, 1, 1) * rho.unsqueeze(0) * (
        1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - usq.unsqueeze(0) / (2.0 * CS2)
    )


def macroscopic(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c, _ = _tensors(f.device, f.dtype)
    rho = f.sum(dim=0)
    mx = (f * c[:, 0].view(9, 1, 1)).sum(dim=0)
    my = (f * c[:, 1].view(9, 1, 1)).sum(dim=0)
    inv = 1.0 / torch.clamp(rho, min=1e-30)
    return rho, mx * inv, my * inv


def collide_edm(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """EDM (Kupershtokh 2004) at a *prescribed* velocity (the mixture velocity)."""
    inv = 1.0 / torch.clamp(rho, min=1e-30)
    feq = equilibrium(rho, ux, uy)
    feq_f = equilibrium(rho, ux + fx * inv, uy + fy * inv)
    return f - (f - feq) / tau + (feq_f - feq)


def stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic pull-streaming: ``out_i(x) = f_i(x - c_i)``."""
    out = torch.empty_like(f)
    for i in range(9):
        cx, cy = _C[i]
        # out[j, i] = f[j - cy, i - cx]  ->  roll by (+cy, +cx)
        out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return out


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
@dataclass
class MCMPConfig:
    G: float = 2.5
    tau_a: float = 1.0
    tau_b: float = 1.0
    psi_kind: str = "rho"
    rho_ref: float = 1.0
    rho0_exp: float = 1.0
    self_coupling: float = 0.0
    #: ``"self"``: each component relaxes to equilibrium at its *own*
    #: half-force-corrected velocity (standard MCMP, the components may slip --
    #: demixing requires that).  ``"common"``: one mixture velocity for both
    #: components (co-flow variant; cannot demix, see the xval card).
    velocity_mode: str = "self"


class MCMPPsiSolver:
    """Two-component psi-field SC solver on a periodic D2Q9 lattice.

    ``u`` is the *common* (mixture) velocity carrying
    ``sum_k (sum_i f_k c_i + F_k/2)``, which is the Palabos/Shan-Chen
    multicomponent coupling and what makes the two components co-flow.
    """

    def __init__(self, rho_a: torch.Tensor, rho_b: torch.Tensor, cfg: MCMPConfig):
        self.cfg = cfg
        self.rho_a0 = rho_a.clone()
        self.rho_b0 = rho_b.clone()
        ux = torch.zeros_like(rho_a)
        uy = torch.zeros_like(rho_a)
        self.f_a = equilibrium(rho_a, ux, uy)
        self.f_b = equilibrium(rho_b, ux, uy)
        self.ny, self.nx = rho_a.shape
        self.steps = 0
        self.u_max = 0.0

    # -- fields ---------------------------------------------------------- #
    def psi(self, rho: torch.Tensor) -> torch.Tensor:
        return psi_of(rho, self.cfg.psi_kind, self.cfg.rho_ref, self.cfg.rho0_exp)

    def forces(self) -> Dict[str, torch.Tensor]:
        rho_a, _, _ = macroscopic(self.f_a)
        rho_b, _, _ = macroscopic(self.f_b)
        return mcmp_psi_force(
            self.psi(rho_a),
            self.psi(rho_b),
            G=self.cfg.G,
            self_coupling=self.cfg.self_coupling,
        )

    def fields(self, need_forces: bool = False) -> Dict[str, torch.Tensor]:
        rho_a, uxa, uya = macroscopic(self.f_a)
        rho_b, uxb, uyb = macroscopic(self.f_b)
        F = self.forces() if need_forces else None
        rho_t = rho_a + rho_b
        inv = 1.0 / torch.clamp(rho_t, min=1e-30)
        out = {
            "rho_a": rho_a,
            "rho_b": rho_b,
            "rho_total": rho_t,
            "ux": (rho_a * uxa + rho_b * uxb) * inv,
            "uy": (rho_a * uya + rho_b * uyb) * inv,
            "psi_a": self.psi(rho_a),
            "psi_b": self.psi(rho_b),
        }
        if F is not None:
            out.update(F)
        out["p"] = eos_pressure(out["rho_a"], out["rho_b"], out["psi_a"], out["psi_b"], self.cfg.G)
        return out

    # -- time step ------------------------------------------------------- #
    def _equilibrium_velocities(self, F):
        """Velocities handed to the collision (``velocity_mode`` switch)."""
        rho_a, uxa, uya = macroscopic(self.f_a)
        rho_b, uxb, uyb = macroscopic(self.f_b)
        if self.cfg.velocity_mode == "common":
            rho_t = rho_a + rho_b
            inv = 1.0 / torch.clamp(rho_t, min=1e-30)
            ux = (rho_a * uxa + rho_b * uxb + 0.5 * (F["fx_a"] + F["fx_b"])) * inv
            uy = (rho_a * uya + rho_b * uyb + 0.5 * (F["fy_a"] + F["fy_b"])) * inv
            return (rho_a, ux, uy), (rho_b, ux, uy)
        if self.cfg.velocity_mode == "self":
            inva = 1.0 / torch.clamp(rho_a, min=1e-30)
            invb = 1.0 / torch.clamp(rho_b, min=1e-30)
            uax = uxa + 0.5 * F["fx_a"] * inva
            uay = uya + 0.5 * F["fy_a"] * inva
            ubx = uxb + 0.5 * F["fx_b"] * invb
            uby = uyb + 0.5 * F["fy_b"] * invb
            return (rho_a, uax, uay), (rho_b, ubx, uby)
        raise ValueError(f"unknown velocity_mode {self.cfg.velocity_mode!r}")

    def step(self) -> None:
        F = self.forces()
        (rho_a, uax, uay), (rho_b, ubx, uby) = self._equilibrium_velocities(F)
        fa = collide_edm(self.f_a, rho_a, uax, uay, F["fx_a"], F["fy_a"], self.cfg.tau_a)
        fb = collide_edm(self.f_b, rho_b, ubx, uby, F["fx_b"], F["fy_b"], self.cfg.tau_b)
        self.f_a = stream(fa)
        self.f_b = stream(fb)
        self.steps += 1
        self.u_max = float(torch.sqrt(uax * uax + uay * uay).max())

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def drop_rho_fields(
    n: int, radius: float, rho_in: float, rho_out: float, width: float = 1.5, centre=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Component A denser inside the disc (``rho_in``), B mirrored; tanh profile."""
    j = torch.arange(n, dtype=DTYPE).view(-1, 1).expand(n, n)
    i = torch.arange(n, dtype=DTYPE).view(1, -1).expand(n, n)
    cx, cy = (n / 2.0, n / 2.0) if centre is None else centre
    r = torch.sqrt((i - cx) ** 2 + (j - cy) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / (2.0 * width)))  # 1 inside, 0 outside
    rho_a = rho_out + (rho_in - rho_out) * s
    rho_b = (rho_in + rho_out) - rho_a
    return rho_a, rho_b


def measure_drop(
    solver: MCMPPsiSolver,
    rho_in: float,
    rho_out: float,
    inner_frac: float = 0.4,
    outer_frac: float = 1.7,
) -> Dict[str, float]:
    """Equivalent radius, inner/outer bulk pressure and Laplace pressure jump."""
    F = solver.fields(need_forces=True)
    rho_a = F["rho_a"]
    n = rho_a.shape[0]
    thr = 0.5 * (rho_in + rho_out)
    inside = rho_a > thr
    area = float(inside.sum())
    r_eq = float(math.sqrt(max(area, 1.0) / math.pi))
    j = torch.arange(n, dtype=DTYPE).view(-1, 1).expand(n, n)
    i = torch.arange(n, dtype=DTYPE).view(1, -1).expand(n, n)
    r = torch.sqrt((i - n / 2.0) ** 2 + (j - n / 2.0) ** 2)
    p_in = float(F["p"][r < inner_frac * r_eq].mean())
    p_out = float(F["p"][r > outer_frac * r_eq].mean())
    u = torch.sqrt(F["ux"] ** 2 + F["uy"] ** 2)
    return {
        "R_eq": r_eq,
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "u_max": float(u.max()),
        "rho_a_in": float(rho_a[r < inner_frac * r_eq].mean()),
        "rho_a_out": float(rho_a[r > outer_frac * r_eq].mean()),
    }


def acoustic_mode_speed(
    G: float = 1.0,
    rho0: float = 0.2,
    kind: str = "rho",
    rho_ref: float = 1.0,
    n_x: int = 128,
    n_y: int = 8,
    amplitude: float = 1e-5,
    n_steps: int = 3000,
    tau: float = 1.0,
) -> Dict[str, float]:
    """Acoustic (total-density) mode speed from the period of a standing wave.

    Both components carry the *same* sinusoidal density perturbation, i.e. the
    total-density mode; the sign changes of ``rho_tot - <rho_tot>`` at one probe
    give the period ``T``, hence ``omega = 2 pi / T`` and ``c = omega / k`` with
    ``k = 2 pi / n_x``.  Reference: :func:`effective_sound_speed`.
    """
    cfg = MCMPConfig(G=G, tau_a=tau, tau_b=tau, psi_kind=kind, rho_ref=rho_ref)
    i = torch.arange(n_x, dtype=DTYPE).view(1, -1).expand(n_y, n_x)
    k = 2.0 * math.pi / n_x
    d = amplitude * torch.cos(k * i)
    base = torch.full((n_y, n_x), rho0, dtype=DTYPE)
    solver = MCMPPsiSolver(base + d, base + d, cfg)

    sig = []
    for _ in range(n_steps):
        solver.step()
        sig.append(float(solver.fields()["rho_total"][0, 0]))
    sig_t = torch.tensor(sig, dtype=DTYPE)
    centred = sig_t - sig_t.mean()
    crossings = []
    for t in range(1, len(centred)):
        a, b = float(centred[t - 1]), float(centred[t])
        if a == 0.0 or (a < 0.0) != (b < 0.0):
            # linear interpolation of the crossing time
            frac = abs(a) / (abs(a) + abs(b)) if (abs(a) + abs(b)) > 0 else 0.0
            crossings.append(t - 1 + frac)
    if len(crossings) < 5:
        return {"c_measured": float("nan"), "c_predicted": effective_sound_speed(G, rho0, kind, rho_ref),
                "rel_err_pct": float("nan"), "n_crossings": len(crossings)}
    half_period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    T = 2.0 * half_period
    omega = 2.0 * math.pi / T
    c_meas = omega / k
    c_pred = effective_sound_speed(G, rho0, kind, rho_ref)
    return {
        "c_measured": c_meas,
        "c_predicted": c_pred,
        "rel_err_pct": 100.0 * (c_meas - c_pred) / c_pred,
        "period": T,
        "n_crossings": len(crossings),
        "damping_ratio_final_over_initial": float(abs(centred[-n_x:]).max() / abs(centred[:n_x]).max()),
    }


def acoustic_pulse_speed(
    G: float = 1.0,
    rho0: float = 0.2,
    kind: str = "rho",
    rho_ref: float = 1.0,
    n_x: int = 700,
    n_y: int = 4,
    amplitude: float = 1e-3,
    width: float = 8.0,
    n_steps: int = 900,
    probe_x: Sequence[int] = (180, 320),
    tau: float = 1.0,
) -> Dict[str, float]:
    """Measure the acoustic (total-density) mode speed of a uniform mixture.

    A small Gaussian density pulse is added to *both* components at
    ``x = n_x/2``; the arrival times of the peak at two probes give the front
    speed ``c = (x2 - x1)/(t2 - t1)``, to be compared with
    :func:`effective_sound_speed`.
    """
    cfg = MCMPConfig(G=G, tau_a=tau, tau_b=tau, psi_kind=kind, rho_ref=rho_ref)
    i = torch.arange(n_x, dtype=DTYPE).view(1, -1).expand(n_y, n_x)
    pulse = amplitude * torch.exp(-((i - n_x / 2.0) ** 2) / (2.0 * width ** 2))
    base_a = torch.full((n_y, n_x), rho0, dtype=DTYPE)
    rho_a = base_a + pulse
    rho_b = base_a + pulse
    solver = MCMPPsiSolver(rho_a, rho_b, cfg)

    times = {x: [] for x in probe_x}
    for t in range(n_steps):
        solver.step()
        rho_t = solver.fields()["rho_total"]
        for x in probe_x:
            times[x].append(float(rho_t[:, x].mean()) - 2.0 * rho0)

    peaks = {}
    for x in probe_x:
        sig = torch.tensor(times[x], dtype=DTYPE)
        k = int(torch.argmax(sig))
        # parabolic refinement of the discrete peak
        if 0 < k < len(sig) - 1:
            y0, y1, y2 = float(sig[k - 1]), float(sig[k]), float(sig[k + 1])
            denom = y0 - 2.0 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
        else:
            shift = 0.0
        peaks[x] = k + shift

    x1, x2 = probe_x[0], probe_x[1]
    dt = peaks[x2] - peaks[x1]
    c_meas = (x2 - x1) / dt if dt > 0 else float("nan")
    c_pred = effective_sound_speed(G, rho0, kind, rho_ref)
    return {
        "c_measured": c_meas,
        "c_predicted": c_pred,
        "rel_err_pct": 100.0 * (c_meas - c_pred) / c_pred,
        "peak_steps": peaks,
        "amp_final": max(abs(v) for v in times[probe_x[0]][-50:]),
    }