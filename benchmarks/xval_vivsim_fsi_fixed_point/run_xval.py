"""XVAL: TensorLBM partitioned FSI coupling  vs  haimingz/vivsim
``pure-function-fsi-fixed-point`` (OSS tag, inventory ``VIVSIM_INVENTORY.md``).

The OSS practice under test
---------------------------
``examples/2d/vortex_induced_vibration.py`` (commit ``fb84b12``) couples a
spring-mounted cylinder to the IB-LBM via a **fixed-point sub-iteration on the
structural state** with the fluid field frozen inside the time step::

    a_old, v_old, d_old = a, v, d                      # :119
    for _ in range(FSI_ITER):                          # :114   FSI_ITER = 1
        marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)
        stencil_weights, stencil_indices = ib.get_ib_stencil(...)
        marker_v = jnp.repeat(v[None, :], N_MARKER, axis=0)
        ib_g, marker_h = ib.multi_direct_forcing(...)   # :124-133 frozen ib_u
        h = dyn.get_force_to_obj(marker_h)              # :135
        h += a * CYL_AREA                               # :136  added-mass line
        a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, M, K, C)   # :137
    ...
    K = (2 * pi * FN) ** 2 * M * (1 + 1 / MR)           # :58   FN = *wet* frequency

This benchmark executes the OSS structure module **as shipped** (``dyn.py`` is
vendored byte-identical, MIT, sha256 in ``vendor/PROVENANCE.json``) through a
NumPy shim, and drives it with the coupling loop above; the LBM/MDF block is
replaced by one documented linear interface operator so that both arms see the
identical fluid:

    F_fluid(a^k) = -m_a * a^k ,      m_a = C_M * rho * A ,  C_M = 1

i.e. the accelerative (added-mass) part of the frozen-field IBM reaction --
exactly the interface operator of :mod:`tensorlbm.fsi_partitioned`
(``f_fluid = f_ext - m_added * a_k``), so the comparison is diagonal in the
fluid.  Velocity-proportional (drag) IBM forces are out of scope here.

Benchmark: harmonically forced damped oscillator with added mass,
``(m_s + m_a) x'' + c x' + k x = F_0 sin(w_f t)``, at ``w_f = w_n`` (wet
natural frequency, the value the OSS calibrates ``K`` to).  The closed-form
transient+steady solution is the reference, so every error below is measured
against an exact number, not against another solver.

Arms
----
``oss_*``  OSS loop verbatim (``dyn.newmark_2dof`` executed from ``vendor/``),
           ``compensate`` toggles line 136, ``fsi_iter`` = ``FSI_ITER``.
``tl_*``   :func:`tensorlbm.fsi_partitioned.solve_partitioned_1d` with the
           matching variant/accelerator; ``tl_dry`` sets ``m_added = 0``.

Cross-implementation identities (machine-checkable, see ``stage operator``):
Q1 ``oss_plain_1``      == ``tl_exp_1``      (TL reproduces the OSS loop)
Q2 ``oss_plain_*``(inf) == ``tl_aitken``/``tl_iqn``  (OSS fixed point == strong coupling)
Q3 ``oss_compen_*``     == ``tl_dry``        (line 136 cancels the added mass)

Usage::

    python benchmarks/xval_vivsim_fsi_fixed_point/run_xval.py            # all stages
    python benchmarks/xval_vivsim_fsi_fixed_point/run_xval.py --stage operator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)
ROOT = HERE.parents[1]

SHIM = ROOT / "benchmarks" / "xval_ib3d_diffuse_interface_mdf" / "jaxshim"


def _add_paths() -> None:
    sys.path.insert(0, str(SHIM))
    sys.path.insert(0, str(HERE / "vendor"))
    sys.path.insert(0, str(ROOT / "src"))


_add_paths()

from vivsim import dyn  # noqa: E402  vendored OSS module (MIT)

from tensorlbm.fsi_partitioned import (  # noqa: E402
    AddedMassModel1D,
    make_accelerator,
    solve_partitioned_1d,
)

OSS_COMMIT = "fb84b12f02e18f3b9dfdf738d6a1a318d81539f9"
OSS_SRC = "examples/2d/vortex_induced_vibration.py"
OSS_DYN_SHA256 = "34e6202420af8378b9645f27ead7aaf6a9ef1976a249188d0c03f1f6f5d054eb"

# ---------------------------------------------------------------------------
# Physical / numerical parameters -- taken from the OSS example itself
# ---------------------------------------------------------------------------
D = 64.0                       # OSS :31   cylinder diameter (lattice units)
U0 = 0.05                      # OSS :50   inflow velocity
UR = 5.0                       # OSS :52   reduced velocity U0/(FN*D)
RE = 150.0                     # OSS :51   Reynolds number (not exercised here)
DR = 0.0                       # OSS :54   structural damping ratio (0 in OSS)
FSI_ITER = 1                   # OSS :70
C_M = 1.0                      # potential-flow added mass of a circular section
AREA = math.pi * (D / 2.0) ** 2  # OSS :38 CYL_AREA, rho_f = 1
FN = U0 / (UR * D)             # OSS :56   *wet* natural frequency
TN = 1.0 / FN                  # natural period (lattice steps)
MASS_RATIOS = (0.5, 1.0, 2.0, 10.0)   # OSS MR = m_s / m_a
ZETA = 0.02                    # total damping ratio (OSS DR = 0; see report)
X_TARGET = 0.05 * D            # steady-state amplitude at resonance (5 % of D)
W_F_OVER_WN = 1.0              # drive at the wet natural frequency
N_PERIODS = 20.0               # record length in natural periods
DT_LEVELS = (16.0, 64.0)       # time step in lattice units (OSS native dt = 1)
DT_NATIVE_SUBSET = ("oss_compen_1", "oss_plain_1", "tl_aitken", "tl_wet_1",
                    "tl_theta1")

TOL_PCT = 15.0                 # owner precision standard (<15 %)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------
def build_model(mass_ratio: float) -> dict:
    """Structural/fluid parameters for one mass ratio (OSS conventions)."""
    m_a = C_M * AREA                       # rho_f = 1
    m_s = mass_ratio * AREA                # OSS :57  M = CYL_AREA * MR
    wn = 2.0 * math.pi * FN
    m_tot = m_s + m_a                      # OSS :58 calibrates K on the wet mass
    k = wn * wn * m_tot
    c = 2.0 * ZETA * wn * m_tot
    x_ss = X_TARGET
    f0 = 2.0 * ZETA * k * x_ss             # resonance amplification 1/(2 zeta)
    return dict(
        mass_ratio=mass_ratio,
        m_s=m_s,
        m_a=m_a,
        k=k,
        c=c,
        fn=FN,
        wn=wn,
        zeta=ZETA,
        w_f=W_F_OVER_WN * wn,
        f0=f0,
        x_ss_target=x_ss,
        beta=m_a / m_s,
    )


def analytic(t, model) -> np.ndarray:
    """Closed-form response of the wet oscillator to ``F0 sin(w_f t)``, at rest."""
    t = np.asarray(t, dtype=float)
    wn, z, wf = model["wn"], model["zeta"], model["w_f"]
    m = model["m_s"] + model["m_a"]
    f0, k = model["f0"], model["k"]
    wd = wn * math.sqrt(1.0 - z * z)
    x_ss = f0 / (m * math.sqrt((wn * wn - wf * wf) ** 2 + (2.0 * z * wn * wf) ** 2))
    phi = math.atan2(2.0 * z * wn * wf, wn * wn - wf * wf)
    a_coef = x_ss * math.sin(phi)
    b_coef = (z * wn * a_coef - x_ss * wf * math.cos(phi)) / wd
    return np.exp(-z * wn * t) * (
        a_coef * np.cos(wd * t) + b_coef * np.sin(wd * t)
    ) + x_ss * np.sin(wf * t - phi)


def analytic_steady(model) -> tuple[float, float]:
    """Steady-state amplitude ``X`` and phase lag ``phi`` of the wet system."""
    wn, z, wf = model["wn"], model["zeta"], model["w_f"]
    m = model["m_s"] + model["m_a"]
    x_ss = model["f0"] / (
        m * math.sqrt((wn * wn - wf * wf) ** 2 + (2.0 * z * wn * wf) ** 2)
    )
    phi = math.atan2(2.0 * z * wn * wf, wn * wn - wf * wf)
    return x_ss, phi


# ---------------------------------------------------------------------------
# Arm 1: OSS coupling loop, verbatim structure, vendored `dyn` executed
# ---------------------------------------------------------------------------
def oss_run(model, *, dt: float, n_steps: int, fsi_iter: int, compensate: bool,
            n_markers: int = 256) -> dict:
    """OSS ``update_step`` FSI block with the documented linear fluid operator.

    Every call into ``dyn`` is the upstream call (``get_markers_coords_2dof``,
    ``get_force_to_obj``, ``newmark_2dof``); the MDF block is replaced by the
    frozen-field linear reaction ``F = -m_a a`` spread over the markers, and
    the harmonic external load is added to the force resultant (the OSS example
    has no external load; this is the only added line, marked EXT below).
    """
    x_markers = 0.5 * D * np.cos(np.linspace(0.0, 2.0 * math.pi, n_markers, endpoint=False))
    y_markers = 0.5 * D * np.sin(np.linspace(0.0, 2.0 * math.pi, n_markers, endpoint=False))

    m_s, m_a, k, c = model["m_s"], model["m_a"], model["k"], model["c"]
    wf, f0 = model["w_f"], model["f0"]

    a = np.zeros(2)
    v = np.zeros(2)
    d = np.zeros(2)
    xs = np.empty(n_steps)
    unstable = False
    wall0 = time.perf_counter()
    for n in range(n_steps):
        t_next = (n + 1) * dt
        h_ext = np.array([0.0, f0 * math.sin(wf * t_next)])        # EXT
        a_old, v_old, d_old = a, v, d
        for _ in range(fsi_iter):
            marker_x, marker_y = dyn.get_markers_coords_2dof(x_markers, y_markers, d)
            # frozen-field linear reaction, spread over the markers (Model A)
            marker_h = np.repeat((-m_a * a)[None, :] / n_markers, n_markers, axis=0)
            h = np.asarray(dyn.get_force_to_obj(marker_h)) + h_ext
            if compensate:
                h = h + a * AREA                                   # OSS :136
            a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, m_s, k, c, dt)
            a, v, d = np.asarray(a), np.asarray(v), np.asarray(d)
        xs[n] = d[1]
        if not np.all(np.isfinite(d)) or abs(d[1]) > 1e3 * X_TARGET:
            unstable = True
            xs[n:] = np.nan if not np.isfinite(d[1]) else xs[n]
            break
    wall = time.perf_counter() - wall0
    return dict(x=xs, t=(np.arange(n_steps) + 1.0) * dt, unstable=unstable,
                iters=np.full(n_steps, fsi_iter, dtype=float), wall=wall)


# ---------------------------------------------------------------------------
# Arm 2: TensorLBM partitioned solver
# ---------------------------------------------------------------------------
def tl_run(model, *, dt: float, n_steps: int, variant: str = "explicit",
           accel: str = "none", max_iter: int = 1, m_added: float | None = None,
           theta: float = 0.0, tol: float = 1e-12) -> dict:
    m_a = model["m_a"] if m_added is None else float(m_added)
    am = AddedMassModel1D(m_struct=model["m_s"], m_added=m_a,
                          stiffness=model["k"], damping=model["c"])
    a_scale = model["f0"] / (model["m_s"] + model["m_a"])
    res = solve_partitioned_1d(
        am,
        n_steps=n_steps,
        dt=dt,
        variant=variant,
        forcing=lambda t: model["f0"] * math.sin(model["w_f"] * t),
        accelerator=make_accelerator(accel) if accel != "none" else None,
        tol=tol,
        max_iter=max_iter,
        a_init=0.0,
        theta=theta,
        ref_scale=a_scale,
    )
    wall = None
    x = np.asarray(res.displacement, dtype=float)
    unstable = (not np.all(np.isfinite(x))) or np.nanmax(np.abs(x)) > 1e3 * X_TARGET
    return dict(x=x, t=np.asarray(res.t), unstable=unstable,
                iters=np.asarray(res.iterations, dtype=float),
                converged=bool(res.converged), wall=wall)


# ---------------------------------------------------------------------------
# Metrics against the analytic reference
# ---------------------------------------------------------------------------
def metrics(out: dict, model) -> dict:
    x = out["x"]
    t = out["t"]
    n = min(x.size, t.size)
    x, t = x[:n], t[:n]
    if not np.all(np.isfinite(x)) or np.nanmax(np.abs(x)) > 1e3 * X_TARGET:
        return dict(unstable=True, blew_up=True, l2_rel_pct=None,
                    amp_rel_pct=None, phase_err_deg=None,
                    iters_mean=float(np.mean(out["iters"])) if n else None,
                    iters_max=float(np.max(out["iters"])) if n else None,
                    wall_s=out.get("wall"))
    ref = analytic(t, model)
    scale = np.sqrt(np.mean(ref ** 2))
    l2 = 100.0 * np.sqrt(np.mean((x - ref) ** 2)) / scale
    # steady-state amplitude / phase from the last 5 periods.  The *same*
    # least-squares fit is applied to the analytic record of the same window,
    # so the residual transient (still ~O(10 %) at 20 periods, zeta = 0.02) is
    # not counted as model error.
    n_win = int(min(n, max(8, round(5.0 * TN / (out["t"][1] - out["t"][0])))))
    tw = t[-n_win:]
    A = np.column_stack([np.sin(model["w_f"] * tw), np.cos(model["w_f"] * tw)])
    coef_n, *_ = np.linalg.lstsq(A, x[-n_win:], rcond=None)
    coef_r, *_ = np.linalg.lstsq(A, analytic(tw, model), rcond=None)
    amp = float(math.hypot(*coef_n))
    amp_ref = float(math.hypot(*coef_r))
    phase = float(math.atan2(-coef_n[1], coef_n[0]))
    phase_ref = float(math.atan2(-coef_r[1], coef_r[0]))
    dphase = (phase - phase_ref + math.pi) % (2.0 * math.pi) - math.pi
    x_ss, _phi = analytic_steady(model)
    return dict(
        unstable=bool(out["unstable"]),
        l2_rel_pct=float(l2),
        amp_rel_pct=float(100.0 * (amp - amp_ref) / max(amp_ref, 1e-30)),
        phase_err_deg=float(math.degrees(dphase)),
        amp=amp,
        amp_ref=amp_ref,
        x_max=float(np.max(np.abs(x))),
        iters_mean=float(np.mean(out["iters"])) if n else None,
        iters_max=float(np.max(out["iters"])) if n else None,
        wall_s=out.get("wall"),
    )


LEGS = {
    "oss_compen_1": dict(kind="oss", fsi_iter=1, compensate=True),
    "oss_compen_10": dict(kind="oss", fsi_iter=10, compensate=True),
    "oss_plain_1": dict(kind="oss", fsi_iter=1, compensate=False),
    "oss_plain_10": dict(kind="oss", fsi_iter=10, compensate=False),
    "oss_plain_40": dict(kind="oss", fsi_iter=40, compensate=False),
    "tl_exp_1": dict(kind="tl", variant="explicit", accel="none", max_iter=1),
    "tl_exp_10": dict(kind="tl", variant="explicit", accel="none", max_iter=10),
    "tl_wet_1": dict(kind="tl", variant="wet_mass", accel="none", max_iter=1),
    "tl_theta1": dict(kind="tl", variant="theta_split", theta=1.0, accel="none",
                      max_iter=1),
    "tl_aitken": dict(kind="tl", variant="explicit", accel="aitken", max_iter=200),
    "tl_iqn": dict(kind="tl", variant="explicit", accel="iqn_ils", max_iter=200),
    "tl_dry": dict(kind="tl", variant="explicit", accel="none", max_iter=1,
                   m_added=0.0),
}


def run_leg(name: str, model, dt: float, n_steps: int) -> dict:
    spec = dict(LEGS[name])
    kind = spec.pop("kind")
    if kind == "oss":
        return oss_run(model, dt=dt, n_steps=n_steps, **spec)
    return tl_run(model, dt=dt, n_steps=n_steps, **spec)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_operator() -> dict:
    """Q1/Q2/Q3: the two implementations agree where they must."""
    out = {}
    dt, n_per = 16.0, 6.0
    n_steps = int(round(n_per * TN / dt))
    pairs = [("Q1", "oss_plain_1", "tl_exp_1"),
             ("Q2", "oss_plain_40", "tl_aitken"),
             ("Q2b", "oss_plain_40", "tl_iqn"),
             ("Q3", "oss_compen_1", "tl_dry"),
             ("Q3b", "oss_compen_10", "oss_compen_1")]
    for mr in MASS_RATIOS:
        model = build_model(mr)
        got = {}
        for name in {p for pair in pairs for p in pair[1:]}:
            got[name] = run_leg(name, model, dt, n_steps)["x"]
        for tag, lhs, rhs in pairs:
            a, b = got[lhs], got[rhs]
            m = min(a.size, b.size)
            fa = np.all(np.isfinite(a[:m]))
            fb = np.all(np.isfinite(b[:m]))
            if not (fa and fb):
                # both arms must diverge together for the identity to hold
                out[f"m{mr:g}_{tag}_{lhs}_vs_{rhs}"] = (
                    0.0 if fa == fb else float("inf"))
                out[f"m{mr:g}_{tag}_{lhs}_vs_{rhs}_diverged"] = True
                continue
            scale = max(np.max(np.abs(b[:m])), 1e-30)
            out[f"m{mr:g}_{tag}_{lhs}_vs_{rhs}"] = float(
                np.max(np.abs(a[:m] - b[:m])) / scale
            )
    # the OSS structure module itself: Newmark kinematics vs TensorLBM's
    a0 = np.array([1.0, -0.5])
    v0 = np.array([0.3, -0.2])
    d0 = np.array([0.1, 0.05])
    h = np.array([2.0, -1.0])
    m_s, k, c = 10.0, 3.0, 0.4
    a1, v1, d1 = dyn.newmark_2dof(a0, v0, d0, h, m_s, k, c, dt)
    c1, c2 = 0.5 * dt, 0.25 * dt * dt
    v_p = v0 + dt * (1.0 - 0.5) * a0
    d_p = d0 + dt * v0 + dt * dt * (0.5 - 0.25) * a0
    a_tl = (h - c * v_p - k * d_p) / (m_s + c1 * c + c2 * k)
    v_tl = v_p + c1 * a_tl
    d_tl = d_p + c2 * a_tl
    out["newmark_max_abs_diff"] = float(
        max(np.max(np.abs(a1 - a_tl)), np.max(np.abs(v1 - v_tl)),
            np.max(np.abs(d1 - d_tl)))
    )
    out["oss_dyn_sha256"] = hashlib.sha256(
        (HERE / "vendor" / "vivsim" / "dyn.py").read_bytes()
    ).hexdigest()
    return out


def stage_sweep() -> dict:
    rows = []
    for mr in MASS_RATIOS:
        model = build_model(mr)
        levels = list(DT_LEVELS) + [1.0]
        for dt in levels:
            n_steps = int(round(N_PERIODS * TN / dt))
            for name in LEGS:
                if dt == 1.0 and name not in DT_NATIVE_SUBSET:
                    continue
                try:
                    out = run_leg(name, model, dt, n_steps)
                    met = metrics(out, model)
                except Exception as exc:  # honest FAIL, keep the row
                    met = dict(unstable=True, error=f"{type(exc).__name__}: {exc}")
                row = dict(mass_ratio=mr, dt=dt, steps=n_steps, leg=name,
                           spec={k: v for k, v in LEGS[name].items()}, **met)
                row["within_15pct"] = (
                    None if met.get("l2_rel_pct") is None
                    else bool(met["l2_rel_pct"] < TOL_PCT)
                )
                rows.append(row)
                l2 = met.get("l2_rel_pct")
                amp = met.get("amp_rel_pct")
                print(f"  m*={mr:<5g} dt={dt:<5g} {name:<14} "
                      f"l2={'-' if l2 is None else f'{l2:.3f}'!s:>10} "
                      f"amp%={'-' if amp is None else f'{amp:+.2f}'!s:>9} "
                      f"dtl={'BLOWUP' if met.get('blew_up') else 'ok'}", flush=True)
    return dict(
        tolerance_pct=TOL_PCT,
        params=dict(D=D, U0=U0, UR=UR, RE=RE, FN=FN, TN=TN, C_M=C_M, AREA=AREA,
                    ZETA=ZETA, X_TARGET=X_TARGET, N_PERIODS=N_PERIODS,
                    W_F_OVER_WN=W_F_OVER_WN, OSS_FSI_ITER=FSI_ITER,
                    mass_ratios=MASS_RATIOS, dt_levels=list(DT_LEVELS) + [1.0]),
        models={str(mr): build_model(mr) for mr in MASS_RATIOS},
        rows=rows,
    )


def stage_sensitivity() -> dict:
    """Second parameter axis: drive frequency detuning (w_f/wn = 0.8)."""
    global W_F_OVER_WN
    keep = W_F_OVER_WN
    rows = []
    for ratio in (0.8, 1.2):
        W_F_OVER_WN = ratio
        for mr in (2.0, 10.0):
            model = build_model(mr)
            dt = 16.0
            n_steps = int(round(N_PERIODS * TN / dt))
            for name in ("oss_compen_1", "oss_plain_1", "tl_aitken"):
                out = run_leg(name, model, dt, n_steps)
                met = metrics(out, model)
                rows.append(dict(wf_over_wn=ratio, mass_ratio=mr, dt=dt,
                                 leg=name, **met))
    W_F_OVER_WN = keep
    return dict(rows=rows)


def stage_summarize(sweep: dict, ops: dict, sens: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt_fig = 16.0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    model = build_model(10.0)
    n_steps = int(round(6.0 * TN / dt_fig))
    t = (np.arange(n_steps) + 1.0) * dt_fig
    axes[0].plot(t / TN, analytic(t, model), "k-", lw=1.6, label="analytic (wet)")
    for name, style in (("oss_plain_1", "r--"), ("oss_compen_1", "b-."),
                        ("tl_aitken", "g:"), ("tl_theta1", "k:"),
                        ("tl_wet_1", "m--")):
        x = run_leg(name, model, dt_fig, n_steps)["x"]
        if np.all(np.isfinite(x)):
            axes[0].plot(t / TN, x, style, lw=1.3, label=name)
    axes[0].set_xlabel("t / T_n")
    axes[0].set_ylabel("x / D")
    axes[0].set_title("m* = 10, dt = 16: response")
    axes[0].legend(fontsize=7)

    legs = ("oss_plain_1", "oss_compen_1", "tl_exp_1", "tl_wet_1", "tl_aitken",
            "tl_theta1")
    for name, mark in zip(legs, ("o", "s", "^", "v", "d", "P")):
        xs, ys = [], []
        for row in sweep["rows"]:
            if row["leg"] == name and row["dt"] == dt_fig and row.get("l2_rel_pct") is not None:
                xs.append(row["mass_ratio"])
                ys.append(row["l2_rel_pct"])
        axes[1].semilogy(xs, ys, mark + "-", label=name)
    axes[1].axhline(TOL_PCT, color="k", ls=":", lw=1.0, label="15 % limit")
    axes[1].set_xlabel("mass ratio m* = m_s / m_a")
    axes[1].set_ylabel("relative L2 error vs analytic [%]")
    axes[1].set_title("dt = 16")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig_path = ART / "xval_fsi_fixed_point.png"
    fig.savefig(fig_path, dpi=140)

    summary = dict(
        anchor="XVAL-LBM-PURE-FUNCTION-FSI-FIXED-POINT",
        oss=dict(repo="haimingz/vivsim", commit=OSS_COMMIT, source=OSS_SRC,
                 tag="pure-function-fsi-fixed-point",
                 vendored_dyn_sha256=ops["oss_dyn_sha256"]),
        operator_identities=ops,
        figure=str(fig_path.relative_to(ROOT)),
        sweep=sweep,
        sensitivity=sens,
    )
    (ART / "xval_fsi_fixed_point.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {fig_path} and artifacts/xval_fsi_fixed_point.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "operator", "sweep", "sensitivity"))
    args = ap.parse_args()
    ops = stage_operator()
    print("operator identities:")
    for k, v in ops.items():
        print(f"  {k}: {v:.3e}" if isinstance(v, float) else f"  {k}: {v}")
    if args.stage in ("all", "sweep"):
        print("\nsweep:")
        sweep = stage_sweep()
        sens = stage_sensitivity()
        stage_summarize(sweep, ops, sens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())