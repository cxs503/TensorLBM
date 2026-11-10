"""Node p234w2-n05 -- inlet turbulence synthesis: spectral/moment calibration run.

Produces every number that goes into the report and the acceptance JSON:

1. digital-filter (DFM/Turin) branch against the von Kármán target, with the
   seven machine gates of the node (spectrum bands, Reynolds stresses, integral
   scale, autocorrelation fit, divergence, sha256 reproducibility, negative
   control);
2. the same branch against the Kaimal target (second target family);
3. the DSRFG branch (auxiliary generator) cross-checked against the DFM branch;
4. a ``dt`` convergence sweep showing that the residual error against the
   *continuous* von Kármán target is a time-resolution effect (unresolved tail),
   not a generator defect;
5. the application layer: mutual arbitration of the two imposition routes
   (momentum sponge vs non-equilibrium extrapolation) in a D2Q9 receiver, plus a
   column-resolved diagnostic of what differs between them;
6. downstream impact in the receiver: turbulent vs laminar inlet.

Run from the repository root::

    python experiments/p234w2_n05_inlet_synthesis/run_inlet_synthesis.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.inlet_downstream import (  # noqa: E402
    ChannelConfig,
    D2Q9Channel,
    cross_check_routes,
    downstream_impact,
    inlet_signal_2d,
)
from tensorlbm.turbulence.inlet_spectral_synthesis import (  # noqa: E402
    DSRFGInlet,
    SpectralInletSynthesis,
    SynthesisConfig,
    evaluate_gates,
    inlet_statistics,
)

OUT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "p234w2_n05_inlet_synthesis"
FIG = ART / "figures"
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RESULTS: dict = {
    "node": "p234w2-n05-inlet-turbulence-synthesis",
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Digital-filter branch, von Kármán target, full acceptance
# ---------------------------------------------------------------------------
log("1/6 DFM von Kármán branch")
CFG = SynthesisConfig(ny=64, nz=64, n_time=2048, dt=0.15, u_rms=0.1, L_ux=2.0, U=1.0, seed=20261031)
field_dfm = SpectralInletSynthesis(CFG).synthesize()
stats_dfm = inlet_statistics(field_dfm, CFG)
sha_a = hashlib.sha256(np.ascontiguousarray(field_dfm, dtype=np.float64).tobytes()).hexdigest()
field_again = SpectralInletSynthesis(CFG).synthesize()
sha_b = hashlib.sha256(np.ascontiguousarray(field_again, dtype=np.float64).tobytes()).hexdigest()
gates_dfm = evaluate_gates(stats_dfm, sha_a=sha_a, sha_b=sha_b)

# negative control: zero integral scale -> white noise in time
NEG_CFG = SynthesisConfig(
    ny=64, nz=64, n_time=2048, dt=0.15, u_rms=0.1, L_ux=0.0, U=1.0, seed=20261031
)
field_neg = SpectralInletSynthesis(NEG_CFG).synthesize()
stats_neg = inlet_statistics(field_neg, CFG)
gates_neg = {"max_band_error": stats_neg.max_band_error, "pass": stats_neg.max_band_error > 0.50}

RESULTS["dfm_von_karman"] = {
    "config": {k: v for k, v in CFG.__dict__.items() if k != "stress_target"},
    "statistics": stats_dfm.as_dict(),
    "gates": gates_dfm,
    "negative_control": gates_neg,
}
log(
    f"    DFM gates: all_pass={gates_dfm['all_pass']} band={gates_dfm['spectrum_band_error']:.4f} "
    f"L_ux={stats_dfm.L_ux_sample:.3f}/{stats_dfm.L_ux_target} div={stats_dfm.divergence_max:.4f} "
    f"neg={stats_neg.max_band_error:.3f}"
)

# ---------------------------------------------------------------------------
# 2. Kaimal target family
# ---------------------------------------------------------------------------
log("2/6 DFM Kaimal branch")
CFG_K = SynthesisConfig(
    ny=64, nz=64, n_time=2048, dt=0.15, u_rms=0.1, L_ux=2.0, U=1.0, seed=20261031, spectrum="kaimal"
)
field_k = SpectralInletSynthesis(CFG_K).synthesize()
stats_k = inlet_statistics(field_k, CFG_K)
# The Kaimal autocorrelation has a fatter tail than the von Kármán one, so the
# integral-scale estimate depends on the pre-registered integration window; all
# three windows are reported (they are not tuned, they are disclosed).
kaimal_windows = {}
for factor in (2.0, 4.0, 8.0):
    n_int = int(round(factor * (CFG_K.L_ux / CFG_K.U) / CFG_K.dt))
    n_int = min(n_int, stats_k.acf.size - 1)
    l_ux = CFG_K.U * float(np.trapezoid(stats_k.acf[: n_int + 1], stats_k.acf_tau[: n_int + 1]))
    kaimal_windows[f"{factor:.0f}T"] = {
        "L_ux": l_ux,
        "rel_error": abs(l_ux - stats_k.L_ux_target) / stats_k.L_ux_target,
    }
RESULTS["dfm_kaimal"] = {
    "statistics": stats_k.as_dict(),
    "gates": evaluate_gates(stats_k),
    "integral_scale_windows": kaimal_windows,
}
log(
    f"    Kaimal gates: all_pass={RESULTS['dfm_kaimal']['gates']['all_pass']} "
    f"band={stats_k.max_band_error:.4f}"
)

# ---------------------------------------------------------------------------
# 3. DSRFG auxiliary branch + cross-method consistency
# ---------------------------------------------------------------------------
log("3/6 DSRFG auxiliary branch")
CFG_DS = SynthesisConfig(ny=64, nz=64, n_time=512, dt=0.5, u_rms=0.1, L_ux=2.0, U=1.0, seed=7)
field_ds = DSRFGInlet(config=CFG_DS, n_modes=96).synthesize()
stats_ds = inlet_statistics(field_ds, CFG_DS)
# Reference DFM run on the *same* reduced grid/time base for an apples-to-apples
# comparison (the DFT branches must share dt to be comparable).
CFG_DS_REF = SynthesisConfig(ny=64, nz=64, n_time=512, dt=0.5, u_rms=0.1, L_ux=2.0, U=1.0, seed=7)
stats_ds_ref = inlet_statistics(SpectralInletSynthesis(CFG_DS_REF).synthesize(), CFG_DS_REF)
consistency = {
    "band_error_ds": stats_ds.max_band_error,
    "band_error_dfm": stats_ds_ref.max_band_error,
    "L_ux_ds": stats_ds.L_ux_sample,
    "L_ux_dfm": stats_ds_ref.L_ux_sample,
    "L_ux_rel_diff": abs(stats_ds.L_ux_sample - stats_ds_ref.L_ux_sample)
    / stats_ds_ref.L_ux_sample,
    "stress_rel_diff": float(
        np.max(np.abs(np.diag(stats_ds.stress_sample) - np.diag(stats_ds_ref.stress_sample)))
        / np.max(np.diag(stats_ds_ref.stress_sample))
    ),
}
RESULTS["dsrfg"] = {
    "statistics": stats_ds.as_dict(),
    "gates": evaluate_gates(stats_ds),
    "cross_method": consistency,
}
log(
    f"    DSRFG gates: all_pass={RESULTS['dsrfg']['gates']['all_pass']} "
    f"L_ux rel diff vs DFM={consistency['L_ux_rel_diff']:.4f}"
)

# ---------------------------------------------------------------------------
# 4. dt convergence: is the residual vs the *continuous* target a resolution effect?
# ---------------------------------------------------------------------------
log("4/6 dt convergence sweep")
sweep = []
for dt, n_time in ((0.5, 640), (0.25, 1280), (0.15, 2048), (0.1, 3072)):
    cfg = SynthesisConfig(ny=64, nz=64, n_time=n_time, dt=dt, u_rms=0.1, L_ux=2.0, U=1.0, seed=3)
    st = inlet_statistics(SpectralInletSynthesis(cfg).synthesize(), cfg)
    sweep.append(
        {
            "dt": dt,
            "n_time": n_time,
            "f_nyquist_over_f_peak": (1.0 / (2.0 * dt)) / 0.07278722939074646,
            "tail_fraction": st.tail_fraction,
            "max_band_error_resolved": st.max_band_error,
            "max_band_error_continuous": st.max_band_error_continuous,
            "L_ux_rel_error": abs(st.L_ux_sample - st.L_ux_target) / st.L_ux_target,
            "acf_exp_r2": st.acf_exp_r2,
            "divergence_max": st.divergence_max,
        }
    )
    log(
        f"    dt={dt}: tail={st.tail_fraction:.3f} band(cont)={st.max_band_error_continuous:.3f} "
        f"L_ux_err={abs(st.L_ux_sample - st.L_ux_target) / st.L_ux_target:.3f}"
    )
RESULTS["dt_sweep"] = sweep

# ---------------------------------------------------------------------------
# 5. Application layer: route arbitration + column diagnostics
# ---------------------------------------------------------------------------
log("5/6 application layer")
# Channel-scale inflow: integral scale 20 lattice steps, 1 % turbulence intensity
CH_CFG = SynthesisConfig(
    ny=48, nz=64, n_time=3000, dt=1.0, u_rms=0.01, L_ux=20.0, U=1.0, L_spatial=8.0, seed=11
)
field_ch = SpectralInletSynthesis(CH_CFG).synthesize()
stats_ch = inlet_statistics(field_ch, CH_CFG)
signal = inlet_signal_2d(field_ch)
RESULTS["channel_inlet_field"] = {
    "config": {k: v for k, v in CH_CFG.__dict__.items() if k != "stress_target"},
    "statistics": stats_ch.as_dict(),
    "gates": evaluate_gates(stats_ch),
    "signal_shape": list(signal.shape),
}

xcheck = cross_check_routes(signal, n_steps=1500, n_relax=1, near_field_station=6)
RESULTS["route_arbitration"] = xcheck
log(
    f"    route arbitration: worst_rel={xcheck['differences']['worst_rel']:.4f} "
    f"pass={xcheck['differences']['pass']}"
)

# Column-resolved diagnostic (why the two routes differ near the field).
diag = {}
for route in ("momentum", "nee"):
    cfg = ChannelConfig(
        nx=64,
        ny=48,
        nu=0.05,
        u_bulk=0.06,
        route=route,
        n_relax=1,
        n_steps=1500,
        monitor_stations=(0,),
        body_force=0.0,
    )
    sim = D2Q9Channel(cfg, signal=signal)
    us, rhos = [], []
    for it in range(1500):
        sim.step(it)
        if it >= 500:
            rho, u, _ = sim._macroscopic()
            us.append(u[:, :5].copy())
            rhos.append(rho[:, :5].copy())
    U, R = np.array(us), np.array(rhos)
    diag[route] = {
        "u_rms_per_column": U[:, 1:-1, :].std(axis=0).mean(axis=0).tolist(),
        "u_mean_per_column": U[:, 1:-1, :].mean(axis=0).mean(axis=0).tolist(),
        "rho_rms_per_column": R[:, 1:-1, :].std(axis=0).mean(axis=0).tolist(),
    }
RESULTS["route_column_diagnostic"] = diag

# Poiseuille validation of the receiver itself
pois = {}
for route in ("nee", "momentum"):
    cfg = ChannelConfig(
        nx=32,
        ny=48,
        nu=0.05,
        u_bulk=0.06,
        periodic_x=True,
        route=route,
        turbulence_amplitude=0.0,
        monitor_stations=(16,),
    )
    pois[route] = D2Q9Channel(cfg).poiseuille_error(4000)
RESULTS["receiver_validation"] = {
    "poiseuille_rel_l2": pois,
    "note": "periodic-x driven channel: the discrete steady state IS the analytic parabola",
}
log(f"    Poiseuille rel L2: {pois}")

# ---------------------------------------------------------------------------
# 6. Downstream impact
# ---------------------------------------------------------------------------
log("6/6 downstream impact")
down = downstream_impact(signal, n_steps=2400, nx=160, ny=48, stations=(20, 80, 140))
RESULTS["downstream_impact"] = down
log(
    f"    turbulent rms: {[round(down['turbulent'][f'x{x}']['rms'], 6) for x in (20, 80, 140)]} "
    f"decay={down['log_decay_rate_per_cell']:.5f}/cell"
)

RESULTS["elapsed_s"] = round(time.time() - T0, 1)
(ART / "inlet_synthesis_metrics.json").write_text(json.dumps(RESULTS, indent=2))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

figs: list[dict] = []

# F1: spectrum + 1/3-octave band errors
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
m = stats_dfm.f
ax[0].loglog(m[1:], stats_dfm.psd[1:], lw=1.0, label="sampled E(f) (DFM)")
ax[0].loglog(m[1:], stats_dfm.target_psd[1:], "--", lw=1.2, label="von Kármán target")
ax[0].axvline(0.07278722939074646, color="k", ls=":", lw=0.8)
ax[0].text(0.078, ax[0].get_ylim()[1] * 0.3, "f_peak", fontsize=8)
ax[0].set_xlabel("f [lattice$^{-1}$]")
ax[0].set_ylabel("E(f) [u$^2$]")
ax[0].set_title("inlet spectrum vs target (u_rms=0.1, L_ux=2)")
ax[0].legend(fontsize=8)
ax[1].semilogx(
    stats_dfm.band_centres, 100 * stats_dfm.band_error, "o-", label="vs band-limited target"
)
ax[1].semilogx(
    stats_dfm.band_centres,
    100 * stats_dfm.band_error_continuous,
    "s--",
    label="vs continuous target",
)
ax[1].axhline(15.0, color="r", ls=":", label="15 % gate")
ax[1].set_xlabel("1/3-octave band centre [lattice$^{-1}$]")
ax[1].set_ylabel("band energy error [%]")
ax[1].set_title("1/3-octave band match, f in [0.2, 5] f_peak")
ax[1].legend(fontsize=8)
fig.tight_layout()
f1 = FIG / "fig1_spectrum_bands.png"
fig.savefig(f1, dpi=150)
plt.close(fig)
figs.append({"name": "fig1_spectrum_bands.png", "path": f1})

# F2: autocorrelation + exponential fit
fig, ax = plt.subplots(figsize=(6.4, 4.2))
n_fit = max(2, int(round(1.5 * (2.0 / 1.0) / CFG.dt)))
tau = stats_dfm.acf_tau[: n_fit + 1]
ax.plot(
    stats_dfm.acf_tau[: 8 * n_fit],
    stats_dfm.acf[: 8 * n_fit],
    ".",
    ms=3,
    label="sampled $R_{uu}(\\tau)$",
)
ax.plot(
    tau,
    np.exp(-tau / stats_dfm.acf_exp_time),
    "-",
    label=f"exp fit $T$={stats_dfm.acf_exp_time:.2f}, $R^2$={stats_dfm.acf_exp_r2:.4f}",
)
ax.axvspan(0, tau[-1], color="0.9", zorder=0)
ax.set_xlabel("$\\tau$ [lattice steps]")
ax.set_ylabel("$R_{uu}(\\tau)/u_{rms}^2$")
ax.set_title("temporal correlation and exponential model")
ax.legend(fontsize=8)
fig.tight_layout()
f2 = FIG / "fig2_autocorrelation.png"
fig.savefig(f2, dpi=150)
plt.close(fig)
figs.append({"name": "fig2_autocorrelation.png", "path": f2})

# F3: dt convergence
fig, ax = plt.subplots(figsize=(6.4, 4.2))
dts = [s["dt"] for s in sweep]
ax.plot(
    dts,
    [100 * s["max_band_error_continuous"] for s in sweep],
    "o-",
    label="band error vs continuous vK",
)
ax.plot(
    dts,
    [100 * s["max_band_error_resolved"] for s in sweep],
    "s-",
    label="band error vs band-limited vK",
)
ax.plot(dts, [100 * s["tail_fraction"] for s in sweep], "^--", label="unresolved spectral tail")
ax.plot(dts, [100 * s["L_ux_rel_error"] for s in sweep], "d--", label="$L_{ux}$ error")
ax.axhline(15.0, color="r", ls=":", lw=0.8)
ax.set_xlabel("sampling step dt [lattice steps]")
ax.set_ylabel("error [%]")
ax.set_title("residual mismatch is a time-resolution effect")
ax.invert_xaxis()
ax.legend(fontsize=8)
fig.tight_layout()
f3 = FIG / "fig3_dt_convergence.png"
fig.savefig(f3, dpi=150)
plt.close(fig)
figs.append({"name": "fig3_dt_convergence.png", "path": f3})

# F4: route arbitration diagnostics
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
cols = np.arange(5)
for route, style in (("momentum", "o-"), ("nee", "s--")):
    ax[0].plot(cols, diag[route]["u_rms_per_column"], style, label=f"{route}: u' rms")
    ax[1].plot(cols, diag[route]["rho_rms_per_column"], style, label=f"{route}: rho' rms")
ax[0].set_xlabel("column x [cells]")
ax[0].set_ylabel("$u'_{rms}$ [lattice]")
ax[0].set_title("delivered fluctuation vs distance from inlet")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("column x [cells]")
ax[1].set_ylabel("$\\rho'_{rms}$ (acoustic)")
ax[1].set_title("spurious compressibility of each route")
ax[1].legend(fontsize=8)
fig.tight_layout()
f4 = FIG / "fig4_route_diagnostic.png"
fig.savefig(f4, dpi=150)
plt.close(fig)
figs.append({"name": "fig4_route_diagnostic.png", "path": f4})

# F5: downstream development
fig, ax = plt.subplots(figsize=(6.4, 4.2))
st_x = [20, 80, 140]
ax.semilogy(st_x, [down["turbulent"][f"x{x}"]["rms"] for x in st_x], "o-", label="turbulent inlet")
ax.semilogy(
    st_x, [max(down["laminar"][f"x{x}"]["rms"], 1e-8) for x in st_x], "s--", label="laminar inlet"
)
ax.set_xlabel("streamwise station x [cells]")
ax.set_ylabel("$u'_{rms}$ at station [lattice]")
ax.set_title("downstream decay of the synthesized turbulence (D2Q9 receiver)")
ax.legend(fontsize=8)
fig.tight_layout()
f5 = FIG / "fig5_downstream.png"
fig.savefig(f5, dpi=150)
plt.close(fig)
figs.append({"name": "fig5_downstream.png", "path": f5})

RESULTS["figures"] = [{"name": f["name"], "path": str(f["path"].relative_to(ROOT))} for f in figs]
(ART / "inlet_synthesis_metrics.json").write_text(json.dumps(RESULTS, indent=2))
log("done; artifacts at " + str(ART))
print(
    json.dumps(
        {k: RESULTS[k]["gates"]["all_pass"] for k in ("dfm_von_karman", "dfm_kaimal", "dsrfg")},
        indent=2,
    )
)
