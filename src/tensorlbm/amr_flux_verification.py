"""P40-N02 — AMR flux-conservation verification surface for TensorLBM.

Three verification cases on top of :mod:`tensorlbm.amr_composite2d`:

1. **Reflux conservation** (:func:`reflux_conservation_case`): composite mass /
   momentum drift of a periodic 2-D field with a two-level patch, with the
   reflux correction on and off (the off-run is the negative control that shows
   the correction is actually doing work).
2. **Taylor-Green vortex** (:func:`taylor_green_case`): kinetic-energy decay of
   the composite grid vs a uniform grid at the *fine* resolution (the AMR must
   not add dissipation at the coarse-fine interface).
3. **Lid-driven cavity Re = 1000** (:func:`cavity_case`): centre-line velocity
   profiles of an AMR run (coarse interior + refined boundary layers) vs a
   uniform run at the fine resolution, plus the Ghia et al. (1982) reference.
4. **Cylinder at Re = 100** (:func:`cylinder_case`): drag coefficient and
   Strouhal number of an AMR run (refined box around the cylinder) vs a uniform
   run at the fine resolution.

Every case returns a plain ``dict`` (JSON-serialisable) so that the numbers in
the report can be injected programmatically (no hand-copied numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .amr_composite2d import (
    CompositeAMR2D,
    PatchSpec,
    collide,
    equilibrium,
    macroscopic,
    solid_force,
    stream_periodic,
    stream_walls,
)

# ---------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982) J. Comput. Phys. 48:387, Table I/II, Re = 1000
# (u along the vertical centre line x = 0.5 and v along y = 0.5)
# ---------------------------------------------------------------------------
GHIA_U = [
    (1.0000, 1.00000), (0.9766, 0.65928), (0.9688, 0.57492), (0.9609, 0.51117),
    (0.9531, 0.46604), (0.8516, 0.33304), (0.7344, 0.18719), (0.6172, 0.05702),
    (0.5000, -0.06080), (0.4531, -0.10648), (0.2813, -0.27805),
    (0.1719, -0.38289), (0.1016, -0.29730), (0.0703, -0.22220),
    (0.0625, -0.20196), (0.0547, -0.18109), (0.0000, 0.00000),
]
GHIA_V = [
    (1.0000, 0.00000), (0.9688, -0.21388), (0.9609, -0.27669),
    (0.9531, -0.33727), (0.9453, -0.39188), (0.9063, -0.51550),
    (0.8594, -0.42665), (0.8047, -0.31966), (0.5000, 0.02526),
    (0.2344, 0.32235), (0.2266, 0.33075), (0.1563, 0.37095),
    (0.0938, 0.32627), (0.0781, 0.30353), (0.0703, 0.29012),
    (0.0625, 0.27485), (0.0000, 0.00000),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def taylor_green_initial(n: int, u0: float) -> np.ndarray:
    """Analytic 2-D Taylor-Green velocity field on an ``n x n`` periodic grid."""
    k = 2.0 * math.pi / n
    y, x = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    ux = -u0 * np.cos(k * x) * np.sin(k * y)
    uy = u0 * np.sin(k * x) * np.cos(k * y)
    return equilibrium(np.ones_like(x), ux, uy)


def kinetic_energy(f: np.ndarray) -> float:
    _, ux, uy = macroscopic(f)
    return 0.5 * float(np.mean(ux * ux + uy * uy))


def _l2_rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(math.sqrt(float(np.mean((a - b) ** 2))) / max(abs(b).max(), 1e-300))


def _max_rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# ---------------------------------------------------------------------------
# case 1: reflux conservation (criteria (a) and (e))
# ---------------------------------------------------------------------------


def reflux_conservation_case(
    n: int = 32,
    re: float = 200.0,
    u0: float = 0.02,
    steps: int = 200,
    patch=(8, 8, 16, 16),
    reflux_spread: str = "global",
) -> dict:
    """Composite mass/momentum drift with reflux on/off (periodic field)."""
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    f0 = taylor_green_initial(n, u0)
    j0, i0, nj, ni = patch

    def run(reflux: bool) -> dict:
        amr = CompositeAMR2D(f0, tau, [PatchSpec(j0=j0, i0=i0, nj=nj, ni=ni)],
                             periodic=True, reflux=reflux,
                             reflux_momentum=False,
                             reflux_spread=reflux_spread, diag_budget=True)
        m0 = amr.composite_mass()
        p0 = amr.composite_momentum()
        mass = [m0]
        flux_res = []
        corr_frac = []
        gross_sum = 0.0
        for _ in range(steps):
            d = amr.step()
            mass.append(amr.composite_mass())
            # interface budget: the mass crossing the interface in and out must
            # balance -- the net interface flux, normalised by the gross flux
            f_in = d["flux_in_total"]
            f_out = d["flux_out_total"]
            gross = abs(f_in) + abs(f_out) + 1e-300
            gross_sum += gross
            flux_res.append(abs(d["mass_change"]) / gross)
            corr_frac.append(abs(d["patch0_reflux_mass_coarse"]) / gross)
        m1 = amr.composite_mass()
        p1 = amr.composite_momentum()
        return {
            "mass_drift_abs": abs(m1 - m0),
            "mass_drift_rel": abs(m1 - m0) / abs(m0),
            "momentum_drift_abs": float(np.linalg.norm(p1 - p0)),
            "mass_history": mass,
            "interface_net_flux_residual_max": float(max(flux_res)),
            "interface_net_flux_residual_total": float(
                abs(m1 - m0) / max(gross_sum, 1e-300)),
            "reflux_correction_fraction_max": float(max(corr_frac)),
            "energy_initial": amr.composite_energy(),
        }

    on = run(True)
    off = run(False)
    return {
        "case": "reflux_conservation",
        "config": {"n": n, "re": re, "u0": u0, "steps": steps,
                   "patch": [j0, i0, nj, ni], "tau": tau, "nu": nu,
                   "reflux_spread": reflux_spread,
                   "reflux_momentum": False},
        "reflux_on": on,
        "reflux_off": off,
        "criterion_mass_drift_rel": 1e-10,
        "criterion_interface_flux_residual": 1e-8,
        "pass_mass": bool(on["mass_drift_rel"] <= 1e-10),
        "pass_interface_flux": bool(
            on["interface_net_flux_residual_total"] <= 1e-8),
        "control_needs_reflux": bool(off["mass_drift_rel"] > 1e-8),
    }


# ---------------------------------------------------------------------------
# case 2: Taylor-Green vortex decay (criterion (b))
# ---------------------------------------------------------------------------


def taylor_green_case(
    n: int = 64,
    re: float = 100.0,
    u0: float = 0.05,
    steps: int = 600,
    record: int = 10,
    patch=(16, 16, 32, 32),
    reflux: bool = False,
    reflux_spread: str = "interface",
) -> dict:
    """AMR vs uniform (fine resolution) kinetic-energy decay of the TG vortex."""
    ratio = 2
    nu_c = u0 * n / re
    tau_c = 0.5 + 3.0 * nu_c
    tau_r = 2.0 * tau_c - 0.5
    j0, i0, nj, ni = patch

    amr = CompositeAMR2D(taylor_green_initial(n, u0), tau_c,
                         [PatchSpec(j0=j0, i0=i0, nj=nj, ni=ni)], periodic=True,
                         reflux=reflux, reflux_momentum=False,
                         reflux_spread=reflux_spread)
    f_ref = taylor_green_initial(ratio * n, u0)
    m0 = amr.composite_mass()
    e_amr, e_ref, e_theory, times = [], [], [], []
    k_c = 2.0 * math.pi / n
    e0 = 0.25 * u0 * u0

    for step in range(steps + 1):
        if step % record == 0:
            t = float(step * ratio)  # fine time units
            e_amr.append(amr.composite_energy())
            e_ref.append(kinetic_energy(f_ref))
            e_theory.append(e0 * math.exp(-4.0 * nu_c * k_c * k_c * step))
            times.append(t)
        if step == steps:
            break
        amr.step()
        for _ in range(ratio):
            f_ref = stream_periodic(collide(f_ref, tau_r))

    mass_drift = abs(amr.composite_mass() - m0) / abs(m0)
    return {
        "case": "taylor_green",
        "config": {"n_coarse": n, "n_fine": ratio * n, "re": re, "u0": u0,
                   "steps_coarse": steps, "record_every": record,
                   "patch": [j0, i0, nj, ni], "tau_coarse": tau_c,
                   "tau_fine": tau_r, "nu_coarse": nu_c, "nu_fine": 2 * nu_c,
                   "reflux": reflux, "reflux_spread": reflux_spread},
        "times_fine": times,
        "energy_amr": e_amr,
        "energy_uniform_fine": e_ref,
        "energy_theory": e_theory,
        "l2_rel_vs_uniform": _l2_rel(e_amr, e_ref),
        "max_rel_vs_uniform": _max_rel(e_amr, e_ref),
        "max_rel_vs_theory": _max_rel(e_amr, e_theory),
        "decay_rate_amr": float(-math.log(e_amr[-1] / e_amr[0]) / max(steps, 1)),
        "decay_rate_uniform": float(-math.log(e_ref[-1] / e_ref[0]) / max(steps, 1)),
        "decay_rate_theory": 4.0 * nu_c * k_c * k_c,
        "mass_drift_rel": mass_drift,
        "criterion_l2_rel": 0.02,
        "pass": bool(_l2_rel(e_amr, e_ref) <= 0.02),
    }


# ---------------------------------------------------------------------------
# case 3: lid-driven cavity (criterion (c))
# ---------------------------------------------------------------------------


def _cavity_solid(n: int) -> np.ndarray:
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return solid


@dataclass
class CavityConfig:
    fluid: int = 64          # cavity side in coarse lattice units
    re: float = 1000.0
    u_lid: float = 0.1
    strip: int = 16          # thickness (coarse cells) of the refined wall strips
    reflux: bool = False     # see report: reflux biases the long accuracy runs
    max_steps: int = 60000
    tol: float = 5e-6        # convergence: max |du| over a check window / u_lid
    check_every: int = 2000


def _cavity_patches(fluid: int, strip: int) -> list[PatchSpec]:
    n = fluid + 2
    s = strip
    core = fluid - 2 * s
    return [
        PatchSpec(j0=1, i0=1, nj=s, ni=fluid),                  # bottom strip
        PatchSpec(j0=1 + fluid - s, i0=1, nj=s, ni=fluid),      # lid strip
        PatchSpec(j0=1 + s, i0=1, nj=core, ni=s),               # left strip
        PatchSpec(j0=1 + s, i0=1 + fluid - s, nj=core, ni=s),   # right strip
    ]


def _centerline_profiles(field: dict, fluid_fine: int) -> dict:
    """Extract u(x=0.5, y) and v(x, y=0.5) on the fine grid."""
    ux, uy = field["ux"], field["uy"]
    n = fluid_fine
    j = n // 2
    return {
        "y": (np.arange(n) + 0.5) / n,
        "u": ux[..., j].copy(),
        "x": (np.arange(n) + 0.5) / n,
        "v": uy[j, ...].copy(),
    }


def cavity_case(cfg: CavityConfig, kind: str = "amr", record_every: int = 500) -> dict:
    """Run the Re = 1000 cavity on the composite grid (``kind='amr'``) or uniform."""
    fluid = cfg.fluid
    n = fluid + 2
    nu_c = cfg.u_lid * fluid / cfg.re
    tau_c = 0.5 + 3.0 * nu_c
    solid = _cavity_solid(n)
    wall_ux = np.zeros((n, n))
    wall_ux[-1, :] = cfg.u_lid
    wall_uy = np.zeros((n, n))
    f = equilibrium(np.ones((n, n)), np.zeros((n, n)), np.zeros((n, n)))
    fluid_mask = ~solid
    hist = {"step": [], "max_ux": [], "mass": []}

    if kind == "amr":
        amr = CompositeAMR2D(f, tau_c, _cavity_patches(fluid, cfg.strip),
                             solid_coarse=solid, wall_ux_coarse=wall_ux,
                             wall_uy_coarse=wall_uy, periodic=False,
                             reflux=getattr(cfg, "reflux", False))
        m0 = amr.composite_mass()
        prev = None
        steps = 0
        for step in range(cfg.max_steps):
            amr.step()
            steps = step + 1
            if steps % record_every == 0:
                hist["step"].append(steps)
                hist["max_ux"].append(float(np.abs(amr.composite_field()["ux"]).max()))
                hist["mass"].append(amr.composite_mass())
            if steps % cfg.check_every == 0:
                cur = amr.composite_field()["ux"]
                if prev is not None:
                    d = float(np.abs(cur - prev).max()) / cfg.u_lid
                    if d < cfg.tol:
                        break
                prev = cur
        field = amr.composite_field()
        mass_drift = abs(amr.composite_mass() - m0) / abs(m0)
    else:
        tau_f = 2.0 * tau_c - 0.5
        nf = 2 * fluid + 2
        solid_f = _cavity_solid(nf)
        wall_ux_f = np.zeros((nf, nf))
        wall_ux_f[-1, :] = cfg.u_lid
        wall_uy_f = np.zeros((nf, nf))
        f = equilibrium(np.ones((nf, nf)), np.zeros((nf, nf)), np.zeros((nf, nf)))
        fluid_f = ~solid_f
        m0 = float(macroscopic(f)[0][fluid_f].sum())
        prev = None
        steps = 0
        for step in range(cfg.max_steps * 2):
            f, _ = stream_walls(collide(f, tau_f), fluid_f, wall_ux_f, wall_uy_f)
            steps += 1
            if steps % (2 * record_every) == 0:
                rho, ux, uy = macroscopic(f)
                hist["step"].append(steps)
                hist["max_ux"].append(float(np.abs(ux).max()))
                hist["mass"].append(float(rho[fluid_f].sum()))
            if steps % (2 * cfg.check_every) == 0:
                rho, ux, uy = macroscopic(f)
                if prev is not None:
                    d = float(np.abs(ux - prev).max()) / cfg.u_lid
                    if d < cfg.tol:
                        break
                prev = ux.copy()
        rho, ux, uy = macroscopic(f)
        # strip the wall ring so the layout matches the composite fine layout
        field = {"rho": rho, "ux": ux, "uy": uy}
        mass_drift = abs(float(rho[fluid_f].sum()) - m0) / abs(m0)

    if kind == "amr":
        # composite fine index J = 2..2*fluid+1 ↔ uniform fine fluid index J-1
        ux = field["ux"][2:2 + 2 * fluid, 2:2 + 2 * fluid]
        uy = field["uy"][2:2 + 2 * fluid, 2:2 + 2 * fluid]
    else:
        ux = field["ux"][1:1 + 2 * fluid, 1:1 + 2 * fluid]
        uy = field["uy"][1:1 + 2 * fluid, 1:1 + 2 * fluid]
    prof = _centerline_profiles({"ux": ux, "uy": uy}, 2 * fluid)
    prof = {k: np.asarray(v, dtype=float).tolist() for k, v in prof.items()}
    return {
        "kind": kind,
        "steps": steps,
        "tau_coarse": tau_c,
        "nu_coarse": nu_c,
        "mass_drift_rel": mass_drift,
        "history": hist,
        "profiles": prof,
    }


def cavity_compare(amr: dict, uni: dict, cfg: CavityConfig) -> dict:
    """Compare an AMR cavity run with a uniform fine run (+ Ghia)."""
    amr = dict(amr["result"] if "result" in amr else amr)
    uni = dict(uni["result"] if "result" in uni else uni)
    pa = {k: np.asarray(v, dtype=float) for k, v in amr["profiles"].items()}
    pu = {k: np.asarray(v, dtype=float) for k, v in uni["profiles"].items()}
    u_lid = cfg.u_lid
    du = np.abs(pa["u"] - pu["u"])
    dv = np.abs(pa["v"] - pu["v"])
    err_u_lid = float(max(du.max(), dv.max()) / u_lid)
    err_peak = float(max(du.max() / max(np.abs(pu["u"]).max(), 1e-30),
                         dv.max() / max(np.abs(pu["v"]).max(), 1e-30)))

    # Ghia comparison (informational: this is physics, not the AMR criterion)
    def ghia_err(prof_key, ghia):
        vals = []
        for coord, ref in ghia:
            idx = int(round(coord * (len(pa[prof_key]) - 1)))
            vals.append(abs(pa[prof_key][idx] - ref))
        return float(max(vals))

    return {
        "case": "cavity_re1000",
        "config": {"fluid": cfg.fluid, "re": cfg.re, "u_lid": cfg.u_lid,
                   "strip": cfg.strip, "tol": cfg.tol},
        "amr": amr,
        "uniform": uni,
        "error_max_vs_uniform_normalised_by_u_lid": err_u_lid,
        "error_max_vs_uniform_normalised_by_peak": err_peak,
        "ghia_u_max_abs_error": ghia_err("u", GHIA_U),
        "ghia_v_max_abs_error": ghia_err("v", GHIA_V),
        "criterion_error": 0.01,
        "pass": bool(err_u_lid <= 0.01),
    }


# ---------------------------------------------------------------------------
# case 4: cylinder at Re = 100 (criterion (d))
# ---------------------------------------------------------------------------


@dataclass
class CylinderConfig:
    d_coarse: int = 8            # cylinder diameter in coarse lattice units
    lx_over_d: float = 24.0
    ly_over_d: float = 12.0
    re: float = 100.0
    u_in: float = 0.05
    xc_over_d: float = 6.0
    patch_pad: int = 8           # coarse cells of refinement margin around cylinder
    steps: int = 12000           # coarse steps
    transient: int = 6000        # coarse steps discarded before analysis
    reflux: bool = False


def _cylinder_masks(nxc, nyc, d, xc, yc, scale=1.0):
    """Analytic cylinder mask on a grid in coarse cells (``scale`` = refine)."""
    solid = np.zeros((nyc, nxc), dtype=bool)
    yy, xx = np.meshgrid(np.arange(nyc), np.arange(nxc), indexing="ij")
    ypos = (yy + 0.5) / scale
    xpos = (xx + 0.5) / scale
    r2 = (xpos - xc) ** 2 + (ypos - yc) ** 2
    solid |= r2 < (d / 2.0) ** 2
    return solid


def cylinder_case(cfg: CylinderConfig, kind: str = "amr") -> dict:
    """Cylinder at Re = 100: drag coefficient and Strouhal number."""
    d = cfg.d_coarse
    nxc = int(round(cfg.lx_over_d * d))
    nyc = int(round(cfg.ly_over_d * d))
    xc, yc = cfg.xc_over_d * d, nyc / 2.0
    nu_c = cfg.u_in * d / cfg.re
    tau_c = 0.5 + 3.0 * nu_c
    solid = np.zeros((nyc, nxc), dtype=bool)
    yy, xx = np.meshgrid(np.arange(nyc), np.arange(nxc), indexing="ij")
    solid = ((xx + 0.5 - xc) ** 2 + (yy + 0.5 - yc) ** 2) < (d / 2.0) ** 2

    # inflow / outflow / freestream: Zou-He free-stream on the left, equilibrium
    # (zero-gradient) on the right, periodic in y is replaced by a free-slip
    # free-stream top/bottom (simple and adequate for Re = 100 wake studies).
    f = equilibrium(np.ones((nyc, nxc)), cfg.u_in * np.ones((nyc, nxc)),
                    np.zeros((nyc, nxc)))

    pad = cfg.patch_pad
    patches = [PatchSpec(j0=int(yc - d / 2 - pad), i0=int(xc - d / 2 - pad),
                         nj=int(d + 2 * pad), ni=int(d + 2 * pad))]
    j0 = patches[0].j0
    i0 = patches[0].i0
    # the analytic fine geometry is the same physical cylinder at 2x resolution
    fine_solid = lambda y, x: ((x - xc) ** 2 + (y - yc) ** 2) < (d / 2.0) ** 2

    if kind == "amr":
        amr = CompositeAMR2D(f, tau_c, patches, solid_coarse=solid,
                             periodic=False, fine_solid=fine_solid,
                             reflux=getattr(cfg, "reflux", False))
        m0 = amr.composite_mass()
        dt_discard = cfg.transient
        fx, fy, mhist = [], [], []
        steps = 0
        for step in range(cfg.steps):
            amr.step()
            steps = step + 1
            force = amr.patch_force(0)
            if step >= dt_discard:
                fx.append(float(force[0]))
                fy.append(float(force[1]))
            if step % 500 == 0:
                mhist.append(amr.composite_mass())
        mass_drift = abs(amr.composite_mass() - m0) / abs(m0)
    else:
        tau_f = 2.0 * tau_c - 0.5
        nxf, nyf = 2 * nxc, 2 * nyc
        solid_f = _cylinder_masks(nxf, nyf, d, xc, yc, scale=2.0)
        fluid_f = ~solid_f
        f = equilibrium(np.ones((nyf, nxf)), cfg.u_in * np.ones((nyf, nxf)),
                        np.zeros((nyf, nxf)))
        m0 = float(macroscopic(f)[0][fluid_f].sum())
        fx, fy = [], []
        steps = 0
        for step in range(cfg.steps * 2):
            f, _ = stream_walls(collide(f, tau_f), fluid_f)
            steps += 1
            if step >= 2 * cfg.transient:
                force = solid_force(f, fluid_f)
                fx.append(float(force[0]))
                fy.append(float(force[1]))
        mass_drift = abs(float(macroscopic(f)[0][fluid_f].sum()) - m0) / abs(m0)

    fx = np.asarray(fx)
    fy = np.asarray(fy)
    n = len(fx)
    half = n // 2
    f_mean_x = float(fx[half:].mean())
    rho_ref = 1.0
    c_d = 2.0 * f_mean_x / (rho_ref * cfg.u_in ** 2 * d)
    lift = fy[half:] - fy[half:].mean()
    # Strouhal number from the dominant FFT peak of the lift history
    n_fft = len(lift)
    spec = np.abs(np.fft.rfft(lift * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0)  # cycles per fine step
    f_peak = float(freqs[1 + int(np.argmax(spec[1:]))])
    st = f_peak * (d * 2.0) / (cfg.u_in / 2.0) / 2.0  # fine units: D_f=2d, U same
    st = f_peak * (2.0 * d) / cfg.u_in  # per coarse step frequency → fine steps
    return {
        "kind": kind,
        "steps": steps,
        "tau_coarse": tau_c,
        "nu_coarse": nu_c,
        "c_d": c_d,
        "st": st,
        "lift_amplitude": float(lift.std()),
        "drag_series_mean": f_mean_x,
        "mass_drift_rel": mass_drift,
        "samples": n,
    }


def cylinder_compare(amr: dict, uni: dict, cfg: CylinderConfig) -> dict:
    amr = dict(amr["result"] if "result" in amr else amr)
    uni = dict(uni["result"] if "result" in uni else uni)
    e_cd = abs(amr["c_d"] - uni["c_d"]) / abs(uni["c_d"])
    e_st = abs(amr["st"] - uni["st"]) / abs(uni["st"])
    return {
        "case": "cylinder_re100",
        "config": {k: getattr(cfg, k) for k in
                   ("d_coarse", "lx_over_d", "ly_over_d", "re", "u_in",
                    "xc_over_d", "patch_pad", "steps", "transient")},
        "amr": amr,
        "uniform": uni,
        "error_c_d_rel": float(e_cd),
        "error_st_rel": float(e_st),
        "criterion": 0.02,
        "pass": bool(e_cd <= 0.02 and e_st <= 0.02),
    }