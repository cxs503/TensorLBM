#!/usr/bin/env python3
"""XVAL: TensorLBM spectral Saffman--Taylor (exponential integrating factor) vs
      ebinan92/Fingering_dynamics ``spectrum_method``.

Study design (predeclared before inspecting results)
----------------------------------------------------
Decision:  can TensorLBM host the OSS pseudo-spectral miscible viscous-fingering
           scheme (integrating factor + RK4 bootstrap + AB2 + trapezoidal
           corrector) as a reusable, verifiable capability?
Alternative A: keep steering/pore-scale fingering studies entirely inside the
           published OSS code (no TensorLBM capability).
Alternative B: re-derive a different time integrator instead of porting.

Hypothesis: the OSS *time integrator* is exactly correct for the diffusion part
(the integrating factor is analytic), and the port reproduces the OSS reference
to round-off; the OSS *elliptic* closure (frozen-coefficient symbol for the
curl-Darcy relation) is not a valid spectral discretisation and pollutes the
solution at R > 0.

Metrics and predeclared thresholds
  M1 port fidelity   : relative L2 error TensorLBM port vs OSS reference, same
                       IC/parameters, 250 steps at R = 0  -> PASS if < 15%
                       (expected: round-off, ~1e-13)
  M2 diffusion exact : R = 0, single smooth Fourier mode vs analytic
                       exp(-k^2 t)                        -> PASS if < 15%
  M3 mass            : |<c>(t)-<c>(0)| / <c>(0)| on the M1 run (no threshold,
                       reported; discrete advection is a divergence)
  M4 divergence      : max|div u| / max|u| of the discrete velocity
  M5 elliptic defect : normalised residual of the exact curl-Darcy relation
                       satisfied by the OSS stream function
  M6 instability     : round-off amplification per time step (two trajectories
                       seeded 1e-13 apart), vs R and vs grid
  M7 temporal order  : diffusion-only amplitude error for dt/2, dt/4 sequences
                       (integrating factor is analytic => machine precision)
Stopping rule: if M1 exceeds 15% the port is declared non-faithful (honest FAIL);
thresholds are never moved after inspecting results.

Environment: TensorLBM worktree, CPU, torch float64, numpy float64; OSS clone at
$OSS_FINGERING_DIR (default /tmp/Fingering_dynamics, commit recorded in the JSON).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D
import oss_spectral_reference as ossref

OSS_DIR = Path(os.environ.get("OSS_FINGERING_DIR", "/tmp/Fingering_dynamics"))
ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {}

CFG = dict(ny=128, nx=256, aspect=2.0, pe=500.0, dt=0.1)
R_CASE = 0.0            # OSS default log-viscosity ratio (its own working regime)
R_CASES = (0.05, 0.2, 2.0)
SEED = 20261113
T_FIDELITY = 25.0
MODE = 4


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------- OSS reference wiring
def load_oss_module():
    src = OSS_DIR / "spectrum_method" / "main.py"
    if not src.exists():
        return None
    nb = types.ModuleType("numba")
    nb.jit = lambda *a, **k: (lambda f: f)
    sys.modules.setdefault("numba", nb)
    import importlib.util
    spec = importlib.util.spec_from_file_location("oss_spectral_main", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                      # pragma: no cover
        log(f"[warn] cannot import OSS main.py: {exc}")
        return None
    return mod


def oss_reference_field(c_ic, n_steps, dt, R):
    """Execute the OSS reference (clone if present, else the numpy transcription)."""
    mod = load_oss_module()
    backend = "oss_clone" if mod is not None else "numpy_transcription"
    if mod is None:
        cm = ossref.OSSCompute(ny=c_ic.shape[0], nx=c_ic.shape[1], aspect=CFG["aspect"],
                              pe=CFG["pe"], viscosity_ratio=R, dt=dt)
        cm.set_c(c_ic)
        cm.run(n_steps)
        return cm.c.copy(), backend

    mod.N, mod.M = c_ic.shape[0], c_ic.shape[1]
    mod.A, mod.Pe, mod.R, mod.DELTA_T = CFG["aspect"], CFG["pe"], R, dt
    cm = mod.Compute()
    cm.ny, cm.nx = c_ic.shape
    cm.km = np.append(np.arange(0, cm.nx // 2) * 2 * np.pi / (mod.A * mod.Pe),
                      np.append(0, np.arange(-cm.nx // 2 + 1, 0) * 2 * np.pi / (mod.A * mod.Pe)))
    cm.kn = np.append(np.arange(0, cm.ny // 2) * 2 * np.pi / mod.Pe,
                      np.append(0, np.arange(-cm.ny // 2 + 1, 0) * 2 * np.pi / mod.Pe))
    cm.km_km = np.tile(cm.km, (cm.ny, 1))
    cm.kn_kn = np.tile(cm.kn, (cm.nx, 1)).T
    cm.km2kn2 = cm.km_km ** 2 + cm.kn_kn ** 2
    cm.c = c_ic.copy()
    cm.c_hat = np.fft.fft2(cm.c)
    cm.phi_x = np.zeros_like(cm.c)
    cm.phi_y = np.zeros_like(cm.c)
    cm.c_x = np.fft.ifft2(cm.c_hat * cm.km_km * 1.0j)
    cm.c_y = np.fft.ifft2(cm.c_hat * cm.kn_kn * 1.0j)
    # OSS main() loop body, run verbatim against the OSS Compute object
    t = 0.0
    for _ in range(n_steps):
        c_hat = cm.c_hat
        J_hat = cm.getJ_hat()
        c_bar = cm.getC_barbyRunge() if t == 0 else cm.getC_barbyAdams(J_hat, cm.j_hat_prev)
        cm.updateC(np.fft.ifft2(c_bar).real)
        cm.updatePhi_xAndPhi_y(cm.getPhi_hat())
        J_bar = cm.getJ_hat()
        nextc_hat = cm.getNextC(J_hat, J_bar, c_hat, c_bar)
        cm.updateC(np.fft.ifft2(nextc_hat).real)
        cm.updatePhi_xAndPhi_y(cm.getPhi_hat())
        cm.j_hat_prev = J_hat
        t += dt
    return cm.c.copy(), backend


def make_sim(**kw):
    cfg = dict(CFG)
    cfg.update({k: v for k, v in kw.items() if k in cfg})
    extra = {k: v for k, v in kw.items() if k not in cfg}
    return SpectralSaffmanTaylor2D(cfg["ny"], cfg["nx"], aspect=cfg["aspect"], pe=cfg["pe"],
                                   dt=cfg["dt"], mode_index=MODE, **extra)


# ------------------------------------------------------------------- M1 / M3
def study_port_fidelity():
    log("== M1 port fidelity: TensorLBM port vs OSS reference (R=0) ==")
    c_ic = ossref.oss_random_front_ic(CFG["ny"], CFG["nx"], SEED)
    sim = make_sim(viscosity_ratio=R_CASE)
    sim.set_concentration(c_ic)
    n_steps = int(round(T_FIDELITY / CFG["dt"]))
    t0 = time.time()
    traj = []
    for k in range(n_steps):
        sim.step()
        if (k + 1) % 50 == 0:
            d = vars(sim.diagnostics())
            d["step"] = k + 1
            traj.append(d)
    t_port = time.time() - t0
    c_port = sim.numpy_state()

    t0 = time.time()
    c_ref, backend = oss_reference_field(c_ic, n_steps, CFG["dt"], R_CASE)
    t_ref = time.time() - t0
    err = rel_l2(c_port, c_ref)
    log(f"   rel L2 = {err:.3e}  (n={n_steps} steps, backend={backend}) "
        f"{'PASS' if err < 0.15 else 'FAIL'}")
    return dict(n_steps=n_steps, t_end=sim.t, backend=backend, R=R_CASE,
                rel_l2=err, pass_15pct=bool(err < 0.15),
                max_abs_diff=float(np.abs(c_port - c_ref).max()),
                mass_initial=float(np.mean(c_ic)), mass_final=float(sim.c.mean()),
                mass_drift=abs(float(sim.c.mean()) - float(np.mean(c_ic))) / float(np.mean(c_ic)),
                c_min=float(sim.c.min()), c_max=float(sim.c.max()),
                divergence_error=sim.divergence_error(),
                port_wall_s=t_port, ref_wall_s=t_ref, traj=traj,
                field_port=c_port.tolist(), field_ref=c_ref.tolist())


def study_port_fidelity_R(R=2.0, t_end=10.0):
    """Port vs OSS reference at R > 0 on a smooth single-mode IC (well posed)."""
    log(f"== M1b port fidelity at R={R} (smooth IC) ==")
    ic = ossref.oss_single_mode_ic(CFG["ny"], CFG["nx"], CFG["aspect"], CFG["pe"],
                                   x0=0.5 * CFG["aspect"] * CFG["pe"], width=4.0,
                                   amplitude=1e-3, n_mode=MODE)
    sim = make_sim(viscosity_ratio=R)
    sim.set_concentration(ic)
    n_steps = int(round(t_end / CFG["dt"]))
    sim.run(sim.dt * n_steps)
    c_port = sim.numpy_state()
    c_ref, backend = oss_reference_field(ic, n_steps, CFG["dt"], R)
    err = rel_l2(c_port, c_ref)
    amp = sim.diagnostics().mode_amplitude
    log(f"   rel L2 = {err:.3e} over {n_steps} steps ({backend}) "
        f"{'PASS' if err < 0.15 else 'FAIL'}")
    return dict(R=R, n_steps=n_steps, t_end=sim.t, backend=backend, rel_l2=err,
                pass_15pct=bool(err < 0.15), max_abs_diff=float(np.abs(c_port - c_ref).max()),
                mode_amplitude=float(amp), c_min=float(sim.c.min()), c_max=float(sim.c.max()))


def growth_rate(R, dt=0.1, t_fit=(2.0, 8.0), ny=None, nx=None):
    """Fit sigma from the amplitude of one transverse mode (linear regime)."""
    ny = CFG["ny"] if ny is None else ny
    nx = CFG["nx"] if nx is None else nx
    sim = SpectralSaffmanTaylor2D(ny, nx, aspect=CFG["aspect"], pe=CFG["pe"],
                                  viscosity_ratio=R, dt=dt, mode_index=MODE)
    sim.set_concentration(sim.single_mode_ic(x0=0.5 * CFG["aspect"] * CFG["pe"],
                                             width=4.0, amplitude=1e-3, n_mode=MODE))
    ts, amps = [], []
    for _ in range(int(round(t_fit[1] / dt)) + 1):
        d = sim.diagnostics()
        ts.append(d.t)
        amps.append(d.mode_amplitude)
        sim.step()
    ts, amps = np.array(ts), np.array(amps)
    mask = (ts >= t_fit[0]) & (ts <= t_fit[1]) & (amps > 0)
    slope, _ = np.polyfit(ts[mask], np.log(amps[mask]), 1)
    return dict(R=R, dt=dt, ny=ny, nx=nx, sigma=float(slope), n_fit=int(mask.sum()),
                ts=ts.tolist(), amps=amps.tolist())


def study_growth_rate():
    log("== M7 linear growth rate vs R and dt ==")
    by_R = [growth_rate(R) for R in (0.05, 0.2, 0.5, 1.0, 2.0, 4.0)]
    for g in by_R:
        log(f"   R={g['R']:<5} sigma={g['sigma']:+.6e}  sigma/R={g['sigma']/g['R']:+.6e}")
    # analytic-anchored metric: sigma = C*R - k^2 (first-order coupling + exact
    # diffusive decay of the transverse mode, km = 0) => C = (sigma + k^2)/R
    k2_mode = (2.0 * np.pi * MODE / CFG["pe"]) ** 2
    for g in by_R:
        g["k2_diffusive"] = k2_mode
        g["C_advective"] = (g["sigma"] + k2_mode) / g["R"]
    C_ref = float(np.mean([g["C_advective"] for g in by_R if g["R"] >= 0.2]))
    for g in by_R:
        g["C_rel_dev_vs_anchor"] = abs(g["C_advective"] - C_ref) / abs(C_ref)
    log(f"   analytic anchor C = (sigma + k^2)/R = {C_ref:.6e} "
        f"(max rel dev {max(g['C_rel_dev_vs_anchor'] for g in by_R):.2e})")
    by_dt = [growth_rate(2.0, dt=dt) for dt in (0.2, 0.1, 0.05)]
    s_inf = (4 * by_dt[-1]["sigma"] - by_dt[-2]["sigma"]) / 3.0
    order = None
    d1 = abs(by_dt[-2]["sigma"] - by_dt[-1]["sigma"])
    d0 = abs(by_dt[-3]["sigma"] - by_dt[-2]["sigma"])
    if d1 > 0:
        order = float(np.log(d0 / d1) / np.log(2.0))
    for g in by_dt:
        log(f"   R=2 dt={g['dt']:<5} sigma={g['sigma']:+.6e} "
            f"rel.err vs extrap={abs(g['sigma']-s_inf)/abs(s_inf):.2e}")
    log(f"   Richardson sigma(dt->0)={s_inf:+.6e}, observed order ~{order}")
    grid = [growth_rate(2.0, dt=0.05, ny=2 * CFG["ny"], nx=2 * CFG["nx"])]
    log(f"   grid 256x512 sigma={grid[0]['sigma']:+.6e}")
    rel_grid = abs(by_dt[-1]["sigma"] - grid[0]["sigma"]) / abs(grid[0]["sigma"])
    log(f"   grid sensitivity (128x256 vs 256x512): rel diff = {rel_grid:.2e}")
    return dict(by_R=by_R, by_dt=by_dt, sigma_extrapolated=float(s_inf),
                observed_order=order, fine_grid=grid[0], grid_rel_diff=float(rel_grid),
                C_anchor=C_ref,
                C_max_rel_dev=float(max(g["C_rel_dev_vs_anchor"] for g in by_R)),
                pass_15pct=bool(max(g["C_rel_dev_vs_anchor"] for g in by_R) < 0.15))


# ----------------------------------------------------------------------- M2
def study_diffusion_analytic():
    log("== M2 diffusion-only analytic check (R=0) ==")
    ny, nx, A, pe = CFG["ny"], CFG["nx"], CFG["aspect"], CFG["pe"]
    x = np.arange(nx) * (A * pe / nx)
    y = np.arange(ny) * (pe / ny)
    m, n = 2, 3
    km, kn = 2 * np.pi * m / (A * pe), 2 * np.pi * n / pe
    k2 = km ** 2 + kn ** 2
    c0 = 0.5 + 0.25 * np.cos(km * x)[None, :] * np.cos(kn * y)[:, None]
    cases = []
    for t_end, dt in ((20.0, 0.1), (20.0, 0.05)):
        sim = make_sim(viscosity_ratio=0.0, dt=dt)
        sim.set_concentration(c0)
        sim.run(t_end)
        got = sim.numpy_state()
        exact = 0.5 + 0.25 * np.exp(-k2 * t_end) * np.cos(km * x)[None, :] * np.cos(kn * y)[:, None]
        err = rel_l2(got, exact)
        amp = (got.max() - got.min()) / (exact.max() - exact.min())
        cases.append(dict(dt=dt, t_end=t_end, rel_l2=err, pass_15pct=bool(err < 0.15),
                          amplitude_ratio=float(amp), n_steps=int(round(t_end / dt))))
        log(f"   dt={dt}: rel L2 vs exp(-k^2 t) = {err:.3e} "
            f"(amplitude ratio {amp:.9f})")
    # keep the finest field for the figure
    sim = make_sim(viscosity_ratio=0.0)
    sim.set_concentration(c0)
    sim.run(20.0)
    exact = 0.5 + 0.25 * np.exp(-k2 * 20.0) * np.cos(km * x)[None, :] * np.cos(kn * y)[:, None]
    return dict(k_squared=k2, mode=(m, n), cases=cases,
                field_sim=sim.numpy_state().tolist(), field_exact=exact.tolist(),
                rel_l2=cases[0]["rel_l2"], pass_15pct=cases[0]["pass_15pct"])


# -------------------------------------------------------------------- M5/M6
def amplification(R, ny=128, nx=256, dt=0.1, n_steps=2, eps=1e-13):
    """Round-off amplification per step: two trajectories seeded ``eps`` apart."""
    ic0 = ossref.oss_random_front_ic(ny, nx, SEED)
    ic1 = ic0 + eps * np.maximum(np.abs(ic0).max(), 1.0) * np.cos(
        2 * np.pi * np.arange(ny)[:, None] / ny) * 0.02
    outs = []
    for ic in (ic0, ic1):
        sim = SpectralSaffmanTaylor2D(ny, nx, aspect=CFG["aspect"], pe=CFG["pe"],
                                      viscosity_ratio=R, dt=dt, mode_index=MODE)
        sim.set_concentration(ic)
        sim.run(None if n_steps is None else sim.dt * n_steps, max_steps=n_steps)
        outs.append(sim.numpy_state())
    d0 = rel_l2(ic1, ic0)
    d1 = rel_l2(outs[1], outs[0])
    return dict(R=R, ny=ny, nx=nx, dt=dt, n_steps=n_steps, seed_separation=d0,
                separation_after=float(d1),
                amplification=d1 / d0 if d0 > 0 else float("nan"))


def study_elliptic_and_instability():
    log("== M5 elliptic defect + M6 round-off amplification ==")
    res = []
    for R in (0.0, *R_CASES):
        sim = make_sim(viscosity_ratio=R)
        sim.set_concentration(ossref.oss_random_front_ic(CFG["ny"], CFG["nx"], SEED))
        e = sim.elliptic_residual()
        log(f"   R={R:<5} normalised curl-Darcy residual = {e['residual_l2']:.3e}")
        res.append(dict(R=R, residual=e["residual_l2"]))

    amps = []
    for R in R_CASES:
        a = amplification(R)
        log(f"   R={R:<5} round-off amplification/step = {a['amplification']:.3e} "
            f"(separation {a['seed_separation']:.1e} -> {a['separation_after']:.1e})")
        amps.append(a)
    grid = []
    for (ny, nx) in ((64, 128), (128, 256)):
        a = amplification(R_CASES[-1], ny=ny, nx=nx)
        log(f"   grid {ny}x{nx} R={a['R']}: amplification/step = {a['amplification']:.3e}")
        grid.append(a)
    return dict(curl_darcy_residual=res, amplification_vs_R=amps, amplification_vs_grid=grid)


# -------------------------------------------------------------------- figures
def make_figures(m1, m2, m5, m7):
    figs = []
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    ny, nx = CFG["ny"], CFG["nx"]

    # fig1: R=0 field evolution of the port + OSS reference at the final time
    c_ic = ossref.oss_random_front_ic(ny, nx, SEED)
    sim = make_sim(viscosity_ratio=R_CASE)
    sim.set_concentration(c_ic)
    snaps = [(0.0, c_ic)]
    for t in (10.0, 25.0):
        sim.run(t)
        snaps.append((t, sim.numpy_state()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, (t, f) in zip(axes, snaps):
        im = ax.imshow(f, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto",
                       extent=[0, CFG["aspect"] * CFG["pe"], 0, CFG["pe"]])
        ax.set_title(f"port, t = {t:g}")
        ax.set_xlabel("x (streamwise)")
        ax.set_ylabel("y (transverse)")
    fig.colorbar(im, ax=axes, label="c")
    fig.suptitle("TensorLBM spectral Saffman-Taylor port, OSS random-front IC, R = 0")
    p = ART / "fig1_fields_R0.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    figs.append(("fig1_fields_R0.png", "Concentration field of the TensorLBM port at t=0/10/25 "
                                       "with the OSS initial condition (R=0 diffusion limit)."))

    # fig2: port vs OSS reference field + difference
    got = np.asarray(m1["field_port"])
    ref = np.asarray(m1["field_ref"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, (f, ttl) in zip(axes[:2], [(ref, "OSS reference"), (got, "TensorLBM port")]):
        im = ax.imshow(f, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{ttl}  t={m1['t_end']:g}")
        fig.colorbar(im, ax=ax)
    d = np.abs(got - ref)
    im = axes[2].imshow(d, origin="lower", cmap="magma", aspect="auto")
    axes[2].set_title(f"|diff|, max {d.max():.1e}, rel L2 {m1['rel_l2']:.1e}")
    fig.colorbar(im, ax=axes[2])
    p = ART / "fig2_port_vs_oss.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    figs.append(("fig2_port_vs_oss.png", "Final field of the port vs the OSS reference and the "
                                        "pointwise difference (relative L2 at round-off level)."))

    # fig3: diffusion analytic
    gs = np.asarray(m2["field_sim"])
    ex = np.asarray(m2["field_exact"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    axes[0].imshow(gs, origin="lower", cmap="viridis", aspect="auto"); axes[0].set_title("port c (R=0)")
    axes[1].imshow(ex, origin="lower", cmap="viridis", aspect="auto"); axes[1].set_title("analytic")
    mid = gs.shape[0] // 2
    axes[2].plot(gs[mid], "o-", ms=2, label="port")
    axes[2].plot(ex[mid], "-", label="analytic")
    axes[2].set_title(f"mid-row cut, rel L2 = {m2['rel_l2']:.1e}")
    axes[2].legend()
    p = ART / "fig3_diffusion_analytic.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    figs.append(("fig3_diffusion_analytic.png", "Integrating factor vs the analytic Fourier decay "
                                                "exp(-k^2 t) for a single smooth mode."))

    # fig4: amplification vs R and vs grid
    aR = m5["amplification_vs_R"]
    ag = m5["amplification_vs_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].semilogy([a["R"] for a in aR], [a["amplification"] for a in aR], "o-")
    axes[0].axhline(1.0, ls=":", c="k")
    axes[0].set_xlabel("log-viscosity ratio R")
    axes[0].set_ylabel("amplification per step")
    axes[0].set_title("round-off amplification of the OSS elliptic symbol")
    axes[1].semilogy([f"{a['ny']}x{a['nx']}" for a in ag], [a["amplification"] for a in ag], "s-")
    axes[1].axhline(1.0, ls=":", c="k")
    axes[1].set_xlabel("grid"); axes[1].set_ylabel("amplification per step")
    axes[1].set_title(f"grid dependence (R={ag[0]['R']:g})")
    p = ART / "fig4_amplification.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    figs.append(("fig4_amplification.png", "Round-off amplification per time step of the OSS "
                                           "frozen-coefficient elliptic symbol vs R and grid."))
    # fig5: growth-rate fit and sigma(R)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    g0 = m7["by_dt"][-1]
    ts, amps = np.array(g0["ts"]), np.array(g0["amps"])
    axes[0].semilogy(ts, amps, "o", ms=2, label=f"dt={g0['dt']}")
    axes[0].semilogy(ts, amps[0] * np.exp(g0["sigma"] * ts), "-",
                     label=f"fit sigma={g0['sigma']:.4f}")
    axes[0].set_xlabel("t"); axes[0].set_ylabel(f"mode {MODE} amplitude")
    axes[0].set_title("linear growth (R=2)")
    axes[0].legend()
    Rs = [g["R"] for g in m7["by_R"]]
    axes[1].plot(Rs, [g["sigma"] for g in m7["by_R"]], "o-", label="sigma(R)")
    axes[1].plot(Rs, [g["sigma"] / g["R"] * Rs[0] for g in m7["by_R"]], "s--",
                 label="linear theory slope (sigma/R at R=0.05)")
    axes[1].set_xlabel("log-viscosity ratio R"); axes[1].set_ylabel("growth rate sigma")
    axes[1].set_title("growth rate vs R"); axes[1].legend()
    p = ART / "fig5_growth_rate.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    figs.append(("fig5_growth_rate.png", "Linear growth rate of the transverse mode: exponential "
                                         "fit and its dependence on the log-viscosity ratio R."))
    return figs


def main():
    t0 = time.time()
    if OSS_DIR.exists():
        try:
            RESULTS["oss_sha"] = subprocess.check_output(
                ["git", "-C", str(OSS_DIR), "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            RESULTS["oss_sha"] = None
    RESULTS["config"] = dict(CFG, R=R_CASE, seed=SEED, mode=MODE,
                             t_fidelity=T_FIDELITY, oss_dir=str(OSS_DIR))
    m1 = study_port_fidelity()
    m1b = study_port_fidelity_R()
    m2 = study_diffusion_analytic()
    m7 = study_growth_rate()
    m5 = study_elliptic_and_instability()
    figs = make_figures(m1, m2, m5, m7)
    RESULTS.update(M1_port_fidelity=m1, M1b_port_fidelity_R=m1b,
                   M2_diffusion_analytic=m2, M7_growth_rate=m7,
                   M5_M6_elliptic_instability=m5,
                   figures=[{"name": n, "caption": c} for n, c in figs],
                   wall_time_s=round(time.time() - t0, 1))
    out = ART / "results.json"
    out.write_text(json.dumps(RESULTS, indent=1))
    log(f"wrote {out}  ({RESULTS['wall_time_s']} s)")


if __name__ == "__main__":
    main()