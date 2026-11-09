"""Case studies for the mesh-orientation / Galilean invariance certificate.

Three physical problems, all with an exact or reference answer:

``poiseuille``
    Plane channel flow at inclination ``theta = 0/15/30/45°`` to the lattice.
    The developed solution ``u(y') = G y'(H-y')/(2 nu)`` is imposed at the
    inlet/outlet faces, so the interior error is dominated by the wall
    treatment of the two *inclined* walls.  Four wall resolutions give the
    observed order of accuracy (criterion C1).
``cylinder``
    Flow past a circular cylinder at ``Re = 100 / 200`` in a doubly periodic
    box driven by a uniform body force oriented at ``theta`` to the lattice.
    Reports the drag coefficient (from the global momentum balance) and the
    Strouhal number (wake probe).  Criteria C2/C3/C6.
``taylor_green``
    Decaying Taylor-Green vortex, optionally superposed with a uniform
    translation ``U`` (Galilean boost).  Reports the decay-rate deviation and
    the superposition residual (criteria C4/C5).

Every function returns a plain ``dict`` so that the driver can serialise it
without any hidden state; no randomness is used anywhere (criterion C7).
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import channel_boundary_cells, circle, poiseuille_profile, tilted_channel
from .solver import D2Q9Solver, VelocityDirichlet

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)


# --------------------------------------------------------------------- case 1
def poiseuille_case(
    theta_deg: float,
    width: int,
    *,
    u_mean: float = 0.01,
    nu: float = 0.1,
    bounce_back: str = "interpolated",
    collision: str = "bgk",
    steps_factor: float = 30.0,
    trim: float = 0.25,
    dtype=None,
) -> dict:
    """One resolution of the inclined-channel Poiseuille verification."""
    if dtype is None:
        dtype = np.float64
    th = math.radians(theta_deg)
    nx = int(round(6 * width))
    ny = int(round(nx * math.tan(th) + width / math.cos(th) + 2 * width)) + 2
    geom = tilted_channel(nx, ny, theta_deg, width)
    tau = 0.5 + 3.0 * nu
    solver = D2Q9Solver(geom, tau=tau, dtype=dtype, bounce_back=bounce_back,
                        collision=collision)
    ux_ex, uy_ex = poiseuille_profile(geom, u_mean)
    solver.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    solver.set_velocity_dirichlet([
        VelocityDirichlet(inlet, ux_ex, uy_ex, (0, 1)),
        VelocityDirichlet(outlet, ux_ex, uy_ex, (0, -1)),
    ])
    steps = int(steps_factor * width * width)
    for _ in range(steps):
        solver.step()
    ux, uy = solver.velocity()
    mask = geom.fluid.copy()
    k = max(1, int(trim * nx))
    mask[:, :k] = False
    mask[:, -k:] = False
    ex = (ux - ux_ex)[mask]
    ey = (uy - uy_ex)[mask]
    l2 = float(np.sqrt(np.mean(ex * ex + ey * ey)))
    linf = float(np.max(np.hypot(ex, ey)))
    # hydraulically relevant error: flux through the inlet face
    flux_ex = float((ux_ex * geom.axis[0] + uy_ex * geom.axis[1])[inlet].sum())
    flux_num = float((ux * geom.axis[0] + uy * geom.axis[1])[inlet].sum())
    return {
        "theta_deg": theta_deg,
        "width": width,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "u_mean": u_mean,
        "steps": steps,
        "bounce_back": bounce_back,
        "collision": collision,
        "l2_error": l2,
        "linf_error": linf,
        "rel_l2": l2 / (1.5 * u_mean),
        "flux_error_rel": abs(flux_num - flux_ex) / abs(flux_ex),
        "trim_frac": trim,
    }


def observed_order(errors: list[float], widths: list[float]) -> list[float]:
    """Observed convergence orders ``p = log(e_i/e_{i+1}) / log(h_i/h_{i+1})``."""
    orders = []
    for i in range(len(errors) - 1):
        a, b = errors[i], errors[i + 1]
        ha, hb = widths[i], widths[i + 1]
        if a <= 0 or b <= 0:
            orders.append(float("nan"))
        else:
            orders.append(float(math.log(a / b) / math.log(hb / ha)))
    return orders


# --------------------------------------------------------------------- case 2
def cylinder_case(
    theta_deg: float,
    re: float,
    *,
    diameter: float = 20.0,
    domain_per_diameter: float = 10.0,
    u_target: float = 0.05,
    collision: str = "trt",
    bounce_back: str = "interpolated",
    transient_d: float = 40.0,
    sample_d: float = 25.0,
    sample_every: int = 5,
    gain_rel: float = 0.05,
    log_prefix: str = "",
) -> dict:
    """Flow past a circular cylinder in a periodic box, driven at angle theta."""
    th = math.radians(theta_deg)
    axis = np.array([math.cos(th), math.sin(th)])
    nrm = np.array([-math.sin(th), math.cos(th)])
    nx = ny = int(round(domain_per_diameter * diameter))
    radius = 0.5 * diameter
    geom = circle(nx, ny, radius, theta_deg=theta_deg)
    nu = u_target * diameter / re
    tau = 0.5 + 3.0 * nu
    solver = D2Q9Solver(geom, tau=tau, collision=collision,
                        bounce_back=bounce_back, dtype=np.float32)
    zeros = np.zeros((ny, nx))
    solver.initialise(zeros, zeros)

    fluid = geom.fluid
    n_fluid = float(fluid.sum())
    # Constant-bulk-velocity drive (dead-beat momentum controller).
    #
    # In a doubly periodic box the drag on the cylinder can only be measured
    # through the momentum budget:  f * N_fluid = F_drag + dP/dt.  Using a
    # constant body force would require spinning the whole box up on the
    # momentum-inertia time scale  tau = M / (dF_drag/du) ~ 1e4 steps.  Instead
    # the body-force density is updated every step as
    #     f_n = rho * (u_target - u_mean,n-1)
    # which holds the bulk momentum fixed to machine precision; the running
    # mean of f_n * N_fluid is then the drag force (verified by the momentum
    # budget, see docs/INVARIANCE_CERTIFICATION.md).
    fx_field = np.zeros((ny, nx))
    fy_field = np.zeros((ny, nx))
    fmag = 0.0

    cx, cy = geom.meta["center"]
    probe = (int(round(cy + 2.0 * diameter * axis[1] + 0.5 * diameter * nrm[1])),
             int(round(cx + 2.0 * diameter * axis[0] + 0.5 * diameter * nrm[0])))

    n_trans = int(transient_d * diameter / u_target)
    n_sample = int(sample_d * diameter / u_target)
    signal_t: list[float] = []
    signal_u: list[float] = []
    u_hist: list[float] = []
    fmag_hist: list[float] = []
    cd_hist: list[float] = []
    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    for n in range(n_trans + n_sample):
        umean_prev = float((ux * axis[0] + uy * axis[1])[fluid].mean())
        fmag = (u_target - umean_prev)
        fx_field[fluid] = fmag * axis[0]
        fy_field[fluid] = fmag * axis[1]
        solver.step(fx_field, fy_field)
        if n % 5 == 0 or n > n_trans:
            rho, ux, uy = solver.macroscopic()
            umean = float((ux * axis[0] + uy * axis[1])[fluid].mean())
            if n > n_trans and n % sample_every == 0:
                signal_t.append(float(n))
                signal_u.append(float(ux[probe] * nrm[0] + uy[probe] * nrm[1]))
                u_hist.append(umean)
                fmag_hist.append(fmag)
                cd_hist.append(fmag * n_fluid / (0.5 * diameter * umean ** 2 + 1e-30))
    u_mean = float(np.mean(u_hist))
    f_mean = float(np.mean(fmag_hist))
    cd = f_mean * n_fluid / (0.5 * diameter * u_mean ** 2)
    # independent check of the momentum budget: mean dP/dt must be ~ 0
    momentum_drift = float((u_hist[-1] - u_hist[0]) / max(len(u_hist), 1)) * n_fluid
    cd_std = float(np.std(cd_hist))
    st, f_peak, sig = _strouhal(signal_t, signal_u, diameter, u_mean)
    return {
        "case": "cylinder",
        "theta_deg": theta_deg,
        "re": re,
        "diameter": diameter,
        "domain": [nx, ny],
        "domain_per_diameter": domain_per_diameter,
        "u_mean": u_mean,
        "u_target": u_target,
        "nu": nu,
        "tau": tau,
        "fmag": fmag,
        "fmag_mean": f_mean,
        "cd": cd,
        "cd_std": cd_std,
        "cd_rel_fluct": cd_std / abs(cd) if cd else float("nan"),
        "momentum_drift_per_step": momentum_drift,
        "st": st,
        "st_peak_freq": f_peak,
        "probe": list(probe),
        "collision": collision,
        "bounce_back": bounce_back,
        "steps_transient": n_trans,
        "steps_sample": n_sample,
        "probe_signal": [round(v, 10) for v in signal_u],
        "log_prefix": log_prefix,
    }


def _strouhal(t: list[float], sig: list[float], diameter: float, u_mean: float):
    """Dominant frequency of the wake signal -> Strouhal number."""
    if len(sig) < 32:
        return float("nan"), float("nan"), sig
    dt = float(np.mean(np.diff(t)))
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:  # parabolic peak refinement
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        delta = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    f_peak = float(freqs[k] + delta * df)
    st = f_peak * diameter / u_mean
    return st, f_peak, sig


# --------------------------------------------------------------------- case 3
def tg_exact(n: int, u0: float, nu: float, time: float, boost: float = 0.0,
             boost_axis: tuple[float, float] = (1.0, 0.0)):
    """Analytic boosted Taylor-Green field on an ``n x n`` periodic box."""
    k = 2.0 * math.pi / n
    x = (np.arange(n) + 0.5)[None, :]
    y = (np.arange(n) + 0.5)[:, None]
    amp = u0 * math.exp(-2.0 * nu * k * k * time)
    ax, ay = boost_axis
    ux = amp * np.sin(k * x) * np.cos(k * y) + boost * ax
    uy = -amp * np.cos(k * x) * np.sin(k * y) + boost * ay
    return np.broadcast_to(ux, (n, n)).copy(), np.broadcast_to(uy, (n, n)).copy()


class _PeriodicBox:
    """Trivial geometry wrapper: a fully periodic box with no solid cells."""

    def __init__(self, n: int) -> None:
        self.nx = self.ny = n
        self.solid = np.zeros((n, n), dtype=bool)
        self.sdf = np.full((n, n), 1e6)
        self.gx = np.zeros((n, n))
        self.gy = np.zeros((n, n))
        self.theta_rad = 0.0
        self.axis = (1.0, 0.0)
        self.normal = (0.0, 1.0)
        self.meta = {"kind": "periodic_box", "n": n}

    @property
    def fluid(self):
        return ~self.solid


def taylor_green_case(
    ma: float,
    *,
    n: int = 64,
    u0: float = 0.02,
    nu: float = 0.01,
    steps: int = 6000,
    periods: int = 1,
    collision: str = "bgk",
    fit_start: int = 500,
    sample_every: int = 10,
    dtype=np.float64,
) -> dict:
    """Boosted Taylor-Green decay; returns decay rate + superposition residual.

    The boost magnitude is chosen as ``U = periods * n / steps`` so that the
    vortex is translated by an exact integer number of cells in the periodic
    box at the final time; the analytic and the discrete un-boosted fields then
    coincide in space and the superposition residual can be evaluated without
    any interpolation.
    """
    boost = periods * n / steps if ma > 0 else 0.0
    if ma > 0:
        ma_actual = boost / CS
    else:
        ma_actual = 0.0
    box = _PeriodicBox(n)
    solver = D2Q9Solver(box, tau=0.5 + 3.0 * nu, collision=collision, dtype=dtype)
    ux0, uy0 = tg_exact(n, u0, nu, 0.0, boost)
    solver.initialise(ux0, uy0)
    ts, energies = [], []
    for step in range(1, steps + 1):
        solver.step()
        if step % sample_every == 0 and step >= fit_start:
            rho, ux, uy = solver.macroscopic()
            mx = float((rho * ux).mean())
            my = float((rho * uy).mean())
            e = 0.5 * float(np.mean((ux - mx) ** 2 + (uy - my) ** 2))
            ts.append(float(step))
            energies.append(max(e, 1e-30))
    slope, intercept = np.polyfit(ts, np.log(energies), 1)
    decay = -float(slope)                     # d ln(E)/dt = -4 nu k^2 (1 + ...)
    ux_m, uy_m = solver.velocity()
    ux_e, uy_e = tg_exact(n, u0, nu, float(steps), boost)
    res_ax = float(np.max(np.abs(ux_m - ux_e)))
    res_ay = float(np.max(np.abs(uy_m - uy_e)))
    out = {
        "case": "taylor_green",
        "ma": ma_actual,
        "boost": boost,
        "periods": periods,
        "n": n,
        "u0": u0,
        "nu": nu,
        "steps": steps,
        "decay_rate": decay,
        "theory_decay": 4.0 * nu * (2.0 * math.pi / n) ** 2,
        "residual_abs": max(res_ax, res_ay),
        "final_energy": float(energies[-1]),
        "collision": collision,
        "fit_start": fit_start,
        "times": ts[:: max(1, len(ts) // 40)],
        "energies": energies[:: max(1, len(energies) // 40)],
    }
    if boost > 0:
        out["residual_relU"] = out["residual_abs"] / boost
    else:
        out["residual_relU"] = float("nan")
    return out


def galilean_superposition_residual(ma: float, **kwargs) -> dict:
    """Residual of ``u(U) - u(0) - U`` (scheme-internal Galilean breaking)."""
    boosted = taylor_green_case(ma, **kwargs)
    base_kwargs = dict(kwargs)
    base = taylor_green_case(0.0, **base_kwargs)
    n = boosted["n"]
    steps = boosted["steps"]
    boost = boosted["boost"]
    # re-run both to compare fields at the same time (cheap, deterministic)
    box = _PeriodicBox(n)
    nu = boosted["nu"]
    u0 = boosted["u0"]
    fields = []
    for b in (boost, 0.0):
        solver = D2Q9Solver(box, tau=0.5 + 3.0 * nu, collision=kwargs.get("collision", "bgk"),
                            dtype=kwargs.get("dtype", np.float64))
        ux0, uy0 = tg_exact(n, u0, nu, 0.0, b)
        solver.initialise(ux0, uy0)
        for _ in range(steps):
            solver.step()
        fields.append(solver.velocity())
    (ux_b, uy_b), (ux_0, uy_0) = fields
    d = np.hypot(ux_b - ux_0 - boost, uy_b - uy_0)
    return {
        "ma": boosted["ma"],
        "boost": boost,
        "superposition_residual_relU": float(np.max(d)) / boost if boost > 0 else float("nan"),
        "decay_rate": boosted["decay_rate"],
        "decay_rate_base": base["decay_rate"],
        "residual_relU": boosted["residual_relU"],
    }