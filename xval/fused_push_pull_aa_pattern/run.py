#!/usr/bin/env python3
"""XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN -- cross-validation bench.

Question
--------
Upstream ``lbm-principles-practice/code`` chapter 13 (``chapter13/gpu/LBM.cu``,
MIT) fuses streaming and collision into one kernel: pull the eight shifted
neighbours of the post-collision field, compute moments, relax, write the
post-collision populations of the *other* 8-direction buffer, rest direction in
its own array -- no intermediate post-stream field.  Does this repository
reproduce that step, and does the reproduction pass the owner's 15 % gate
against the closed-form 2-D Taylor-Green decay that upstream itself grades
against (``chapter13/gpu/LBM.cu:231-262``)?

Arms
----
``oss_numpy``    line-by-line NumPy transliteration of the upstream kernel
                 (``oss_port.py``); the reference for every other arm.
``torch_fused``  :class:`tensorlbm.fused_push_pull.FusedPushPullStepper`,
                 upstream storage (f0 + f1/f2 ping-pong).
``torch_single`` same stepper with a single 8-direction buffer (extension).
``torch_ab``     repository general path: ``solver.stream`` + ``collide_bgk``
                 on one ``(9, ny, nx)`` field (textbook AB pattern).
Controls
--------
``nu_bias``      kernel relaxes to nu*1.25 but is graded against nu
                 (does the 15 % analytic gate bite at all?).
``mirror_x``     pull index with cx negated, i.e. West/East swapped -- the
                 classic direction-table porting bug.
``flip_all``     pull index with (cx,cy) negated -- a *global* sign flip, which
                 happens to be a symmetry of the Taylor-Green state; kept in the
                 report as a documented blind spot of this xval case.

Run:  python run.py --out result.json          # full (~2 min, background it)
      python run.py --quick --out result.json  # CI-sized
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import oss_port  # noqa: E402
from oss_port import OSSFusedStepper, taylor_green_eval  # noqa: E402

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.fused_push_pull import (  # noqa: E402
    FusedPushPullStepper,
    equilibrium_upstream,
    tau_from_nu,
    tg_l2_errors,
)
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

GATE = 0.15          # owner accuracy standard: > 15 % is unacceptable
TARGET = 0.03        # the stricter "<3 %?" read-out required by the report spec


# --------------------------------------------------------------------------- config
def base_cfg(scale: int = 2, *, nu: float = 1.0 / 6.0, u_max: float = 0.02,
             nsteps: int | None = None, nsave: int | None = None) -> dict:
    """Upstream default is ``LBM.h`` scale=2 -> NX=64, u_max=0.04/scale, NSTEPS=200*scale^2."""
    nx = 32 * scale
    return {
        "nx": nx,
        "ny": nx,
        "nu": nu,
        "nu_ref": nu,
        "tau": tau_from_nu(nu),
        "u_max": u_max,
        "rho0": 1.0,
        "nsteps": int(nsteps if nsteps is not None else 200 * scale * scale),
        "nsave": int(nsave if nsave is not None else 50 * scale * scale),
    }


def nondim(cfg: dict) -> dict:
    nx, nu, u_max = cfg["nx"], cfg["nu"], cfg["u_max"]
    k = 2.0 * math.pi / nx
    k2 = 2.0 * k * k
    td = 1.0 / (nu * k2)
    return {
        "NX": nx, "NY": cfg["ny"],
        "nu_lu": nu, "tau": cfg["tau"], "omega": 1.0 / cfg["tau"],
        "u_max_lu": u_max, "Ma": u_max / math.sqrt(1.0 / 3.0),
        "Re_L": u_max * nx / (2.0 * math.pi) / nu,
        "k_lu": k, "k_h": k * 1.0,
        "t_decay_lu": td, "T_total_lu": cfg["nsteps"],
        "decay_at_T": math.exp(-cfg["nsteps"] / td),
        "c_s_lu": math.sqrt(1.0 / 3.0),
        "lattice": "D2Q9, BGK, periodic, single phase, no forcing",
    }


# ----------------------------------------------------------------------------- arms
def run_oss(cfg: dict, *, nsteps: int | None = None, sample_every_step: bool = False) -> dict:
    if nsteps is not None:
        cfg = {**cfg, "nsteps": nsteps}
    s = OSSFusedStepper(cfg["nx"], cfg["ny"], cfg["nu"], cfg["u_max"], cfg["rho0"])
    samples = [_sample(s.time, s.rho, s.ux, s.uy, cfg)]
    t0 = time.perf_counter()
    for n in range(cfg["nsteps"]):
        s.step(save=True)
        if sample_every_step or (n + 1) % cfg["nsave"] == 0:
            samples.append(_sample(s.time, s.rho, s.ux, s.uy, cfg))
    wall = time.perf_counter() - t0
    return {
        "arm": "oss_numpy", "wall_s": wall, "samples": samples,
        "mass_final": float(s.fields().sum()), "fields": s.fields(),
        "mac": {"rho": s.rho.copy(), "ux": s.ux.copy(), "uy": s.uy.copy()}, "state": s,
    }


def run_torch(cfg: dict, *, single_buffer: bool = False) -> dict:
    st = FusedPushPullStepper(_eq_fields(cfg), cfg["tau"], single_buffer=single_buffer)
    samples = [_sample(0, *st.macroscopic(), cfg)]
    t0 = time.perf_counter()
    for n in range(cfg["nsteps"]):
        st.step()
        if (n + 1) % cfg["nsave"] == 0:
            samples.append(_sample(st.time, *st.macroscopic(), cfg))
    wall = time.perf_counter() - t0
    rho, ux, uy = st.macroscopic()
    return {
        "arm": "torch_single" if single_buffer else "torch_fused",
        "wall_s": wall, "samples": samples,
        "mass_final": float(st.fields().sum()), "fields": st.fields().numpy(),
        "mac": {"rho": rho.numpy(), "ux": ux.numpy(), "uy": uy.numpy()},
    }


def run_ab(cfg: dict) -> dict:
    f = _eq_fields(cfg)
    samples = [_sample(0, *macroscopic(f), cfg)]
    t0 = time.perf_counter()
    for n in range(cfg["nsteps"]):
        f = collide_bgk(stream(f), cfg["tau"])
        if (n + 1) % cfg["nsave"] == 0:
            samples.append(_sample(n + 1, *macroscopic(f), cfg))
    wall = time.perf_counter() - t0
    rho, ux, uy = macroscopic(f)
    return {"arm": "torch_ab", "wall_s": wall, "samples": samples,
            "mass_final": float(f.sum()), "fields": f.numpy(),
            "mac": {"rho": rho.numpy(), "ux": ux.numpy(), "uy": uy.numpy()}}


def run_variant(cfg: dict, mode: str, *, nu_kernel: float | None = None) -> dict:
    """Control arm: upstream kernel with one deliberate defect (``mode``)."""
    nu_k = cfg["nu"] if nu_kernel is None else nu_kernel
    s = OSSFusedStepper(cfg["nx"], cfg["ny"], nu_k, cfg["u_max"], cfg["rho0"])
    samples = [_sample(0, s.rho, s.ux, s.uy, cfg)]
    for n in range(cfg["nsteps"]):
        _variant_step(s, nu_k, mode)
        s.f1, s.f2 = s.f2, s.f1
        s.time += 1
        if (n + 1) % cfg["nsave"] == 0:
            samples.append(_sample(s.time, s.rho, s.ux, s.uy, cfg))
    return {"arm": mode, "wall_s": float("nan"), "samples": samples,
            "mass_final": float(s.fields().sum()), "fields": s.fields(),
            "mac": {"rho": s.rho.copy(), "ux": s.ux.copy(), "uy": s.uy.copy()}}


def _variant_step(s: OSSFusedStepper, nu: float, mode: str) -> None:
    """Upstream kernel with a selectable convention/physics defect."""
    f0, f1, f2 = s.f0, s.f1, s.f2
    tauinv = 2.0 / (6.0 * nu + 1.0)
    om = 1.0 - tauinv

    def sh(a, cx, cy):
        if mode == "flip_all":
            return np.roll(a, shift=(-cy, -cx), axis=(-2, -1))
        if mode == "mirror_x":
            return np.roll(a, shift=(cy, -cx), axis=(-2, -1))
        raise ValueError(f"unknown variant mode {mode!r}")

    ft = [sh(f1[0], 1, 0), sh(f1[1], 0, 1), sh(f1[2], -1, 0), sh(f1[3], 0, -1),
          sh(f1[4], 1, 1), sh(f1[5], -1, 1), sh(f1[6], -1, -1), sh(f1[7], 1, -1)]
    r = f0 + ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7]
    ri = 1.0 / r
    ux = ri * (ft[0] + ft[4] + ft[7] - (ft[2] + ft[5] + ft[6]))
    uy = ri * (ft[1] + ft[4] + ft[5] - (ft[3] + ft[6] + ft[7]))
    s.rho, s.ux, s.uy = r, ux, uy
    omusq = 1.0 - 1.5 * (ux * ux + uy * uy)
    tux, tuy = 3.0 * ux, 3.0 * uy
    f0[:] = om * f0 + tauinv * oss_port.W0 * r * omusq
    cd = (tux, tuy, -tux, -tuy, tux + tuy, tuy - tux, -(tux + tuy), tux - tuy)
    for i, cdi in enumerate(cd):
        w = oss_port.WD if i >= 4 else oss_port.WS
        f2[i] = om * ft[i] + tauinv * w * r * (omusq + cdi * (1.0 + 0.5 * cdi))


# --------------------------------------------------------------------------- helpers
def _eq_fields(cfg: dict) -> torch.Tensor:
    rho, ux, uy = taylor_green_eval(0, cfg["nx"], cfg["ny"], cfg["nu"], cfg["u_max"], cfg["rho0"])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    return equilibrium_upstream(t(rho), t(ux), t(uy))


def _sample(t: int, rho, ux, uy, cfg: dict) -> dict:
    ten = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    err = tg_l2_errors(
        ten(rho), ten(ux), ten(uy),
        nx=cfg["nx"], ny=cfg["ny"], u_max=cfg["u_max"],
        nu=cfg.get("nu_ref", cfg["nu"]), t=t, rho0=cfg["rho0"],
    )
    return {"t": int(t), **{k: float(v) for k, v in err.items()}}


def fit_nu(samples: list[dict], cfg: dict) -> dict:
    """Invert nu from the kinetic-energy decay: E(t) ~ E0 exp(-2 nu k^2 t)."""
    t = np.array([s["t"] for s in samples], dtype=float)[1:]
    e = np.array([s["energy"] for s in samples], dtype=float)[1:]
    keep = e > 0
    t, e = t[keep], e[keep]
    if len(t) < 2:
        return {"nu_fit": float("nan"), "rel_err": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(t, np.log(e), 1)
    k = 2.0 * math.pi / cfg["nx"]
    nu_fit = -slope / (2.0 * 2.0 * k * k)
    resid = np.log(e) - (slope * t + intercept)
    ss_res = float((resid**2).sum())
    ss_tot = float(((np.log(e) - np.log(e).mean()) ** 2).sum())
    return {"nu_fit": float(nu_fit),
            "rel_err": float(abs(nu_fit - cfg["nu"]) / cfg["nu"]),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n_points": int(len(t))}


def compare(a: dict, b: dict) -> dict:
    fa, fb = np.asarray(a["fields"]), np.asarray(b["fields"])
    if fa.shape != fb.shape:
        return {"shape_mismatch": True, "rel_l2": float("nan")}
    d = np.abs(fa - fb)
    scale = float(np.abs(fa).max())
    return {"max_abs": float(d.max()),
            "rel_to_max": float(d.max() / scale) if scale else float("nan"),
            "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(fa)) if np.linalg.norm(fa) else float("nan"),
            "bitwise_equal": bool(np.array_equal(fa, fb))}


def compare_mac(a: dict, b: dict) -> dict:
    """Macroscopic cross-arm comparison (populations can mask a velocity error)."""
    am, bm = a.get("mac"), b.get("mac")
    if am is None or bm is None:
        return {"available": False}
    out = {}
    for name in ("rho", "ux", "uy"):
        x, y = np.asarray(am[name]), np.asarray(bm[name])
        den = float(np.linalg.norm(x))
        out[name + "_rel_l2"] = float(np.linalg.norm(x - y) / den) if den else float("nan")
        out[name + "_max_abs"] = float(np.abs(x - y).max())
    return out


def gate_of(value: float) -> str:
    return "PASS" if value < GATE else "FAIL"


def rho_mode_analysis(cfg: dict, nsteps: int = 400) -> dict:
    """Where does the density diagnostic live: wavevector + envelope period."""
    s = OSSFusedStepper(cfg["nx"], cfg["ny"], cfg["nu"], cfg["u_max"], cfg["rho0"])
    series: list[float] = []
    spec_top: list = []
    for _ in range(nsteps):
        s.step()
        ra, _, _ = taylor_green_eval(s.time, cfg["nx"], cfg["ny"], cfg["nu"], cfg["u_max"])
        d = s.rho - ra
        series.append(float(np.sqrt((d**2).sum() / ((ra - cfg["rho0"]) ** 2).sum())))
        if s.time % 100 == 0:
            F = np.abs(np.fft.fft2(d))
            flat = np.argsort(F.ravel())[::-1][:4]
            spec_top.append({"t": s.time,
                             "modes": [[int(i), int(j), float(F[i, j])]
                                       for i, j in (np.unravel_index(f, F.shape) for f in flat)]})
    y = np.array(series) - float(np.mean(series))
    F = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), d=1.0)
    idx = int(np.argmax(F[1:]) + 1)
    period = float(1.0 / fr[idx])
    k = 2.0 * math.pi / cfg["nx"]
    cs = math.sqrt(1.0 / 3.0)
    # dominant spatial mode is |k| = 2k (one component), envelope alternates at T/2
    t_ac = 2.0 * math.pi / (cs * 2.0 * k)
    return {
        "nsteps": nsteps,
        "series": [{"t": i + 1, "rho_lu2": v} for i, v in enumerate(series)],
        "spectra": spec_top,
        "envelope_period_measured": period,
        "acoustic_period_2k": t_ac,
        "predicted_envelope_period": 0.5 * t_ac,
        "period_rel_err": abs(period - 0.5 * t_ac) / (0.5 * t_ac),
        "rho_err_min": float(np.min(series)),
        "rho_err_max": float(np.max(series)),
    }


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "result.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    result: dict = {
        "anchor": "XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN",
        "gate": GATE, "target": TARGET,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "machine": platform.machine(), "dtype": "float64"},
        "oss_ref": json.loads((HERE / "ref" / "PROVENANCE.json").read_text()),
    }

    # --- 1. base case (= upstream default config): all arms ----------------
    cfg = base_cfg(2)
    if args.quick:
        cfg = base_cfg(2, nsteps=200, nsave=100)
    result["base_cfg"] = cfg
    result["nondim"] = nondim(cfg)

    runs = {
        "oss_numpy": run_oss(cfg),
        "torch_fused": run_torch(cfg),
        "torch_single": run_torch(cfg, single_buffer=True),
        "torch_ab": run_ab(cfg),
    }
    result["base"] = {k: {kk: vv for kk, vv in v.items()
                          if kk not in ("fields", "state", "mac")}
                      for k, v in runs.items()}
    for v in result["base"].values():
        v["nu_inversion"] = fit_nu(v["samples"], cfg)

    result["cross_arm"] = {
        "oss_vs_torch_fused": compare(runs["oss_numpy"], runs["torch_fused"]),
        "oss_vs_torch_single": compare(runs["oss_numpy"], runs["torch_single"]),
        "oss_vs_torch_ab": compare(runs["oss_numpy"], runs["torch_ab"]),
    }
    result["cross_arm_mac"] = {
        "oss_vs_torch_fused": compare_mac(runs["oss_numpy"], runs["torch_fused"]),
        "oss_vs_torch_single": compare_mac(runs["oss_numpy"], runs["torch_single"]),
        "oss_vs_torch_ab": compare_mac(runs["oss_numpy"], runs["torch_ab"]),
    }
    result["storage"] = {
        "upstream_doubles_per_node": runs["oss_numpy"]["state"].fields().shape[0] + 8,
        "torch_fused_doubles_per_node": 17,
        "torch_single_doubles_per_node": 9,
        "ab_doubles_per_node": 18,
        "note": "structural (per-node doubles), not a bandwidth measurement",
    }

    # --- 2. controls -------------------------------------------------------
    if not args.quick:
        bias = dict(cfg)
        bias["nu_ref"] = cfg["nu"]
        bias_run = _run_nu_bias(bias)
        result["controls"] = {
            "nu_bias_25pct": {
                "nu_kernel": bias["nu"] * 1.25, "nu_reference": cfg["nu"],
                "final": bias_run["samples"][-1],
                "gate": gate_of(bias_run["samples"][-1]["ux_lu2"]),
            },
            "mirror_x": {},
            "flip_all": {},
        }
        for mode in ("mirror_x", "flip_all"):
            r = run_variant(cfg, mode)
            result["controls"][mode] = {
                "final": r["samples"][-1],
                "max_uy_err": max(s["uy_lu2"] for s in r["samples"]),
                "gate_analytic_ux": gate_of(r["samples"][-1]["ux_lu2"]),
                "gate_analytic_uy": gate_of(r["samples"][-1]["uy_lu2"]),
                "cross_arm_vs_oss": compare(runs["oss_numpy"], r),
                "cross_arm_mac_vs_oss": compare_mac(runs["oss_numpy"], r),
            }

    # --- 3. resolution study ---------------------------------------------
    result["resolution"] = []
    for sc in ([1, 2] if args.quick else [1, 2, 4]):
        c = base_cfg(sc)
        r_oss = run_oss(c)
        entry = {"cfg": c, "nondim": nondim(c), "oss_final": r_oss["samples"][-1],
                 "nu_inversion": fit_nu(r_oss["samples"], c),
                 "oss_samples": r_oss["samples"]}
        if sc <= 2:  # torch arm at every resolution that fits the time budget
            r_t = run_torch(c)
            entry["torch_final"] = r_t["samples"][-1]
            entry["oss_vs_torch"] = compare(r_oss, r_t)
        result["resolution"].append(entry)
    obs = [(r["nondim"]["k_lu"], r["oss_final"]["ux_lu2"]) for r in result["resolution"]]
    result["resolution_order"] = [
        math.log(e1 / e0) / math.log(k1 / k0)
        for (k0, e0), (k1, e1) in zip(obs, obs[1:]) if e0 > 0 and e1 > 0
    ]

    # --- 4. sensitivity: tau and u_max (NX=64, 200 steps) ----------------
    result["sensitivity"] = {"tau": [], "u_max": []}
    if not args.quick:
        for tau in (0.6, 0.8, 1.0, 1.4):
            c = base_cfg(2, nsteps=200, nsave=50, nu=(tau - 0.5) / 3.0)
            r = run_oss(c)
            result["sensitivity"]["tau"].append({
                "tau": tau, "nu_lu": c["nu"], "final": r["samples"][-1],
                "nu_inversion": fit_nu(r["samples"], c)})
        for um in (0.005, 0.01, 0.02, 0.04):
            c = base_cfg(2, nsteps=200, nsave=50, u_max=um)
            r = run_oss(c)
            result["sensitivity"]["u_max"].append({
                "u_max": um, "Ma": um / math.sqrt(1.0 / 3.0),
                "final": r["samples"][-1], "nu_inversion": fit_nu(r["samples"], c)})

        # --- 5. where the density diagnostic lives ------------------------
        result["rho_mode"] = rho_mode_analysis(base_cfg(2), nsteps=400)

    # --- 6. verdict -------------------------------------------------------
    fin = runs["oss_numpy"]["samples"][-1]
    verdict = {
        "ux_err_oss": fin["ux_lu2"],
        "ux_err_torch_fused": runs["torch_fused"]["samples"][-1]["ux_lu2"],
        "ux_err_torch_ab": runs["torch_ab"]["samples"][-1]["ux_lu2"],
        "ux_gate": gate_of(fin["ux_lu2"]),
        "ux_target_3pct": bool(fin["ux_lu2"] < TARGET),
        "uy_err_oss": fin["uy_lu2"],
        "uy_gate": gate_of(fin["uy_lu2"]),
        "rho_err_oss": fin["rho_lu2"],
        "rho_gate": gate_of(fin["rho_lu2"]),
        "nu_inversion_rel_err": result["base"]["oss_numpy"]["nu_inversion"]["rel_err"],
        "nu_gate": gate_of(result["base"]["oss_numpy"]["nu_inversion"]["rel_err"]),
        "cross_arm_rel_l2": result["cross_arm"]["oss_vs_torch_fused"]["rel_l2"],
        "cross_arm_bitwise": result["cross_arm"]["oss_vs_torch_fused"]["bitwise_equal"],
        "single_buffer_bitwise": result["cross_arm"]["oss_vs_torch_single"]["bitwise_equal"],
        "ab_vs_fused_rel_l2": result["cross_arm"]["oss_vs_torch_ab"]["rel_l2"],
        "ab_vs_fused_ux_rel_l2": result["cross_arm_mac"]["oss_vs_torch_ab"]["ux_rel_l2"],
        "oss_vs_fused_ux_rel_l2": result["cross_arm_mac"]["oss_vs_torch_fused"]["ux_rel_l2"],
        "resolution_order": result["resolution_order"],
    }
    if "controls" in result:
        ctl = result["controls"]
        verdict["control_nu_bias_ux"] = ctl["nu_bias_25pct"]["final"]["ux_lu2"]
        verdict["control_nu_bias_gate"] = ctl["nu_bias_25pct"]["gate"]
        verdict["control_mirror_x_uy"] = ctl["mirror_x"]["final"]["uy_lu2"]
        verdict["control_mirror_x_gate"] = ctl["mirror_x"]["gate_analytic_uy"]
        verdict["control_mirror_x_cross_arm_rel_l2"] = ctl["mirror_x"]["cross_arm_vs_oss"]["rel_l2"]
        verdict["control_mirror_x_ux_rel_l2"] = ctl["mirror_x"]["cross_arm_mac_vs_oss"]["ux_rel_l2"]
        verdict["control_flip_all_ux"] = ctl["flip_all"]["final"]["ux_lu2"]
        verdict["control_flip_all_cross_arm_rel_l2"] = ctl["flip_all"]["cross_arm_vs_oss"]["rel_l2"]
        verdict["control_flip_all_ux_rel_l2"] = ctl["flip_all"]["cross_arm_mac_vs_oss"]["ux_rel_l2"]
        verdict["gate_can_fail"] = bool(
            ctl["nu_bias_25pct"]["gate"] == "FAIL" and ctl["mirror_x"]["gate_analytic_uy"] == "FAIL")
    if "rho_mode" in result:
        verdict["rho_envelope_period"] = result["rho_mode"]["envelope_period_measured"]
        verdict["rho_period_rel_err"] = result["rho_mode"]["period_rel_err"]

    result["verdict"] = verdict
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(verdict, indent=1))
    print("wrote", args.out)
    return 0


def _run_nu_bias(cfg: dict) -> dict:
    """Kernel relaxes to nu*1.25, graded against the true nu (analytic gate)."""
    c = dict(cfg)
    c["nu_ref"] = cfg["nu"]
    s = OSSFusedStepper(c["nx"], c["ny"], cfg["nu"] * 1.25, c["u_max"], c["rho0"])
    samples = [_sample(0, s.rho, s.ux, s.uy, c)]
    for n in range(c["nsteps"]):
        s.step(save=True)
        if (n + 1) % c["nsave"] == 0:
            samples.append(_sample(s.time, s.rho, s.ux, s.uy, c))
    return {"samples": samples}


if __name__ == "__main__":
    raise SystemExit(main())