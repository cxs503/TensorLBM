"""Moving-wall half-way bounce-back (D2Q9 BGK) — cross-validation core.

Anchor: ``XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK``.

Every case here uses only the in-repo primitives
``tensorlbm.d2q9.{equilibrium,macroscopic}`` and
``tensorlbm.solver.{collide_bgk,stream}``; the only thing this module adds is
the wall closure, and it offers *several* closures so that the accuracy of the
reference one can be falsified against the others (ablation, see ``VARIANTS``).

Wall closure (reference, ``ladd_pre_local``)
-------------------------------------------
Wall rows ``y = 0`` (bottom) and ``y = ny-1`` (top) are solid.  Half-way
bounce-back reflects the *pre-collision* populations, so the effective no-slip
surface sits at ``y = 0.5`` and ``y = ny-1.5``.  A wall moving with velocity
``u_w`` (x component only here) adds the Ladd momentum term

    f_new[q](x_w) = f_pre[opp(q)](x_w) + 2 w_q rho_w (c_q . u_w) / cs^2 ,
    rho_w = sum_q f_pre[q](x_w),  cs^2 = 1/3

applied inside the wall cells after collision and before streaming (Ladd 1994
JFM 271:285; Zou & He 1997 Phys. Fluids 9:1591; Krueger et al. 2017 *The
Lattice Boltzmann Method*, §9.3).  Equivalently pylbm ``LinearBounceBack``
with ``alpha = 2``, ``delta = 1/2`` collapses to ``f_opp = f_hat_q - 2 e^-_q``
with ``e^-_q = (f^eq_q - f^eq_opp)/2`` evaluated with the wall velocity — the
wall velocity enters through the *antisymmetric* equilibrium only, which is why
no mass is created at the wall (derivation and source pointers in
``OSS_DIFF.md``).

NB the term must be added to the *reflected* (post-collision) populations: the
wall cell's collided populations are discarded by the bounce-back, so a term
added before collision is annihilated (measured: variant
``ladd_precollision`` reproduces ``static_bb`` to machine precision, i.e. the
wall does not move at all).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorlbm.d2q9 import C, OPPOSITE, _w_for, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS2 = 1.0 / 3.0
DTYPE = torch.float64

#: Ablation variants of the moving-wall closure.
VARIANTS = (
    "ladd_pre_local",     # reference: pre-streaming BB + Ladd term with local rho
    "ladd_pre_unitrho",   # same, but rho_w := 1
    "ladd_precollision",  # Ladd term added to f before collision (ordering error)
    "static_bb",          # no momentum term at all (wall treated as at rest)
    "ladd_sign_flip",     # momentum term with the opposite sign
    "eq_wall_node",       # wall cell reset to equilibrium(1, u_w) each step
)


@dataclass
class Wall:
    """Wall-velocity field for a channel whose solid rows are y=0 and y=ny-1."""

    u_wall: torch.Tensor  # (ny, nx) float64, x-component only

    @classmethod
    def uniform(cls, ny: int, nx: int, u_bottom: float, u_top: float) -> "Wall":
        u = torch.zeros((ny, nx), dtype=DTYPE)
        u[0, :] = u_bottom
        u[-1, :] = u_top
        return cls(u_wall=u)


def _inject(u_wall: torch.Tensor, rho_w: torch.Tensor) -> torch.Tensor:
    """Ladd moving-wall momentum term, shape (9, ny, nx)."""
    c = C.to(u_wall.device).to(DTYPE)
    w = _w_for(DTYPE, u_wall.device).view(9, 1, 1)
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)  # c_q . (u_w, 0)
    return 2.0 * w * rho_w.unsqueeze(0) * cu / CS2


def make_step(ny: int, nx: int, tau: float, variant: str = "ladd_pre_local"):
    """Return ``step(f, u_wall)`` for one LBM time step.

    ``u_wall`` is a (ny, nx) float64 field of the wall x-velocity at the current
    step; only the two wall rows are read.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")
    device = torch.device("cpu")
    opp = OPPOSITE.to(device)
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    ones = torch.ones((ny, nx), dtype=DTYPE, device=device)

    def step(f: torch.Tensor, u_wall: torch.Tensor) -> torch.Tensor:
        f_pre = f
        if variant == "eq_wall_node":
            f = collide_bgk(f, tau)
            feq_w = equilibrium(ones, u_wall, torch.zeros_like(u_wall))
            f = torch.where(wall_mask.unsqueeze(0), feq_w, f)
            return stream(f)
        if variant == "ladd_precollision":
            inj = _inject(u_wall, f_pre.sum(dim=0))
            f = collide_bgk(f + inj * wall_mask.unsqueeze(0), tau)
            return stream(torch.where(wall_mask.unsqueeze(0), f_pre[opp], f))
        # ---- pre-streaming half-way bounce-back at the wall rows ----
        f = collide_bgk(f, tau)
        f_new = torch.where(wall_mask.unsqueeze(0), f_pre[opp], f)
        if variant != "static_bb":
            rho_w = ones if variant == "ladd_pre_unitrho" else f_pre.sum(dim=0)
            inj = _inject(u_wall, rho_w)
            f_new = f_new + (-inj if variant == "ladd_sign_flip" else inj) * wall_mask.unsqueeze(0)
        return stream(f_new)

    return step


def init_field(ny: int, nx: int, ux0: np.ndarray | None = None) -> torch.Tensor:
    """Equilibrium initial state; ``ux0`` is a per-row (ny,) fluid velocity."""
    if ux0 is None:
        ux0 = np.zeros(ny)
    ux = torch.tensor(np.asarray(ux0, dtype=float), dtype=DTYPE).view(ny, 1).repeat(1, nx)
    rho = torch.ones((ny, nx), dtype=DTYPE)
    return equilibrium(rho, ux, torch.zeros((ny, nx), dtype=DTYPE))


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------
def couette_analytic(ny: int, u0: float) -> tuple[np.ndarray, np.ndarray]:
    """Steady plane Couette: walls at y=0.5 and y=ny-1.5, fluid rows y=1..ny-2."""
    h_eff = ny - 2.0
    y = np.arange(ny, dtype=float)
    u_ana = u0 * np.clip((y - 0.5) / h_eff, 0.0, 1.0)
    return y, u_ana


def stokes2_analytic(y: np.ndarray, t: float, u0: float, omega: float, nu: float) -> np.ndarray:
    """Stokes' second problem over a plate at y=0.5 oscillating as U0 cos(omega t).

    u(y,t) = U0 exp(-k (y-y0)) cos(omega t - k (y-y0)),  k = sqrt(omega/(2 nu))
    (Stokes 1851; Schlichting & Gersten, *Boundary-Layer Theory*, ch. 5).
    """
    k = math.sqrt(omega / (2.0 * nu))
    eta = y - 0.5
    return u0 * np.exp(-k * eta) * np.cos(omega * t - k * eta)


def run_couette(ny: int, nx: int, tau: float, u0: float, variant: str = "ladd_pre_local",
                n_steps: int = 4000, tol: float = 1e-9, avg_steps: int = 200) -> dict:
    """Steady Couette (top wall u=U0, bottom at rest) with half-way bounce-back."""
    nu = (tau - 0.5) / 3.0
    h_eff = ny - 2.0
    u_wall = Wall.uniform(ny, nx, 0.0, u0).u_wall
    step = make_step(ny, nx, tau, variant)

    y = np.arange(ny, dtype=float)
    f = init_field(ny, nx, u0 * np.clip((y - 0.5) / h_eff, 0.0, 1.0))
    m0 = float(f.sum())
    hist: list[float] = []
    steady = False
    n_done = 0
    for k in range(1, n_steps + 1):
        f = step(f, u_wall)
        n_done = k
        if k % 200 == 0:
            _, ux, _ = macroscopic(f)
            hist.append(float(ux[ny - 2, nx // 2]))
            if len(hist) >= 10:
                rec = hist[-10:]
                if (max(rec) - min(rec)) / max(abs(sum(rec) / len(rec)), 1e-30) < tol:
                    steady = True
                    break
    acc = np.zeros(ny)
    for _ in range(avg_steps):
        f = step(f, u_wall)
        _, ux, _ = macroscopic(f)
        acc += ux[:, nx // 2].detach().numpy()
    u_num = acc / avg_steps
    _, u_ana = couette_analytic(ny, u0)
    fluid = slice(1, ny - 1)
    un, ua = u_num[fluid], u_ana[fluid]
    umax = max(np.max(np.abs(ua)), 1e-30)
    a, b = np.polyfit(np.arange(1, ny - 1) - 0.5, un, 1)
    return {
        "case": "couette",
        "variant": variant,
        "ny": ny, "nx": nx, "tau": tau, "nu": nu, "H_eff": h_eff,
        "U0": u0, "Ma": u0 / math.sqrt(CS2), "Re": u0 * h_eff / nu,
        "n_steps": n_done, "steady": steady,
        "l2_rel": float(np.linalg.norm(un - ua) / np.linalg.norm(ua)),
        "max_rel_pct_cf_umax": float(np.max(np.abs(un - ua)) / umax * 100.0),
        "slope_num": float(a), "slope_ana": u0 / h_eff,
        "slope_err_pct": float((a - u0 / h_eff) / (u0 / h_eff) * 100.0),
        "y_wall_fit": float(-b / a) if abs(a) > 0 else float("nan"),
        "mass_drift_pct": (float(f.sum()) - m0) / m0 * 100.0,
        "u_num": [float(v) for v in u_num],
        "u_ana": [float(v) for v in u_ana],
    }


def run_stokes2(ny: int, nx: int, tau: float, u0: float, delta_bl: float,
                variant: str = "ladd_pre_local", n_periods: float = 6.0,
                sample_stride: int = 4, nu: float | None = None) -> dict:
    """Stokes' second problem: plate at y=0.5 oscillating as U0 cos(omega t).

    ``delta_bl = sqrt(2 nu / omega)`` is the viscous penetration depth in lattice
    units; the channel depth is set by the caller (``ny = gap + 2``).
    """
    if nu is None:
        nu = (tau - 0.5) / 3.0
    omega = 2.0 * nu / delta_bl ** 2
    k_wave = math.sqrt(omega / (2.0 * nu))  # = 1/delta_bl
    period = 2.0 * math.pi / omega
    n_steps = int(round(n_periods * period))
    step = make_step(ny, nx, tau, variant)
    f = init_field(ny, nx)
    m0 = float(f.sum())
    y = np.arange(ny, dtype=float)

    samples: list[tuple[float, np.ndarray]] = []
    last_period_start = n_steps - period
    for k in range(1, n_steps + 1):
        t = float(k)
        u_wall = torch.zeros((ny, nx), dtype=DTYPE)
        u_wall[0, :] = u0 * math.cos(omega * t)  # oscillating bottom plate
        f = step(f, u_wall)
        if t >= last_period_start and (k % sample_stride == 0):
            _, ux, _ = macroscopic(f)
            samples.append((t, ux[:, nx // 2].detach().numpy().copy()))

    ts = np.array([s[0] for s in samples], dtype=float)
    U = np.array([s[1] for s in samples])          # (nt, ny)
    ana = np.array([stokes2_analytic(y, t, u0, omega, nu) for t in ts])
    fluid = slice(1, ny - 1)
    dU = U[:, fluid] - ana[:, fluid]
    err_row_max_pct = np.max(np.abs(dU), axis=0) / u0 * 100.0   # per fluid row
    rows = np.arange(1, ny - 1)
    eta = rows - 0.5
    resolved = eta <= 3.0 * delta_bl
    return {
        "case": "stokes2",
        "variant": variant,
        "ny": ny, "nx": nx, "tau": tau, "nu": nu,
        "U0": u0, "Ma": u0 / math.sqrt(CS2),
        "omega": omega, "period_steps": period, "delta_bl": delta_bl,
        "gap_over_delta": (ny - 2.0) / delta_bl,
        "n_steps": n_steps, "sample_stride": sample_stride,
        "max_err_pct_U0_full": float(np.max(err_row_max_pct)),
        "max_err_pct_U0_resolved": float(np.max(err_row_max_pct[resolved])),
        "argmax_row_resolved": int(rows[resolved][int(np.argmax(err_row_max_pct[resolved]))]),
        "l2_rel_pct": float(np.linalg.norm(dU) / np.linalg.norm(ana[:, fluid]) * 100.0),
        "far_field_floor_pct": float(math.exp(-k_wave * (ny - 2.0)) * 100.0),
        "mass_drift_pct": (float(f.sum()) - m0) / m0 * 100.0,
        "err_row_max_pct": [float(v) for v in err_row_max_pct],
        "row_index": [int(r) for r in rows],
        "t_samples": [float(v) for v in ts],
        "u_profiles": U[:, fluid].tolist(),
        "u_analytic_profiles": ana[:, fluid].tolist(),
    }


def harmonic_fit(res: dict, eta_over_delta: float) -> dict | None:
    """Fundamental-harmonic amplitude/phase of one row from a stokes2 result."""
    delta = res["delta_bl"]
    nu = res["nu"]
    omega = res["omega"]
    rows = res["row_index"]
    target = 0.5 + eta_over_delta * delta
    row = min(rows, key=lambda r: abs(r - target))
    j = rows.index(row)
    U = np.array(res["u_profiles"])[:, j]
    ts = np.array(res["t_samples"])
    a = 2.0 / len(ts) * np.sum(U * np.cos(omega * ts))
    b = 2.0 / len(ts) * np.sum(U * np.sin(omega * ts))
    amp = math.hypot(a, b)
    phase = math.degrees(math.atan2(b, a)) % 360.0
    k_wave = 1.0 / delta
    eta = row - 0.5
    amp_ana = res["U0"] * math.exp(-k_wave * eta)
    phase_ana = math.degrees(k_wave * eta) % 360.0
    return {
        "row": int(row),
        "eta_over_delta": float(eta / delta),
        "amp_num_over_U0": amp / res["U0"],
        "amp_ana_over_U0": amp_ana / res["U0"],
        "amp_err_pct": (amp - amp_ana) / amp_ana * 100.0,
        "phase_num_deg": phase,
        "phase_ana_deg": phase_ana,
        "phase_err_deg": ((phase - phase_ana + 180.0) % 360.0) - 180.0,
    }