"""Front-estimator calibration for the finite-density-ratio RT arm (CASE-RT-INSTABILITY-ATWOOD).

Background
----------
``docs/CASE-RT-TWOFLUID-PRECOND.md`` (mission m21884) ended with an honest FAIL and a
three-item checklist that any front estimator must pass **before** it may be used in a
gate: (i) seed readback error < 5 %, (ii) null-dynamics (``g = 0``) drift < 0.1 u_s,
(iii) hydrostatic common-mode rejection.  The estimator then in place (``column_height``,
a clamped conserved mass moment) failed (i) with a readback of ``-0.0021 lam`` against a
planned ``0.02 lam``.

This script runs the checklist on **real relaxed SCMP interfaces** (not on an idealised
``tanh``), for the legacy estimator and for the gradient-centroid estimator added in
``src/tensorlbm/rayleigh_taylor_twofluid.py`` (``gradient_front_state``), and then asks the
only question that matters for the anchor: given the calibrated instrument, is the spike
terminal velocity of this model above its own noise floor?

Checks
------
``K1``  seed readback: displace the *relaxed* flat profile by a known ``a0`` (single cosine
        mode) and read the amplitude back; sweep ``a0 / lam`` over 0.005 ... 0.2.
``K2``  null dynamics: seeded mode, gravity **off**, measured drift of the amplitude and of
        the spike/bubble tips -> estimator noise floor (must be < 0.1 u_s).
``K3``  hydrostatic common mode: multiply the relaxed profile by a linear-in-y factor of the
        measured magnitude ``g H / c_s^2`` (and 3x that); the *amplitude* must survive while
        the common mode absorbs the offset.
``K4``  coexistence convergence: Atwood number from the same coupling on two box/step
        settings -> the reference ``u_s(A)`` sensitivity to that spread.
``K5``  physics attempt on the calibrated instrument: staged start-up (precondition + gravity
        ramp), 64 x 6 lam box, tau = 1.0, spike terminal velocity vs the two-fluid Layzer value
        at the *measured* A, plus the signal-to-noise ratio against K2's floor.

Everything is written programmatically to
``artifacts_estimator/calibration.json`` (zero hand-transcription) together with a markdown
table ``artifacts_estimator/calibration.md``.

Usage::

    python3 probe_estimator_calibration.py                 # all checks (~5 min, 6 threads)
    python3 probe_estimator_calibration.py --only K1,K3    # subset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "6")))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
import sys

for _p in (str(REPO / "src"), str(REPO)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from tensorlbm.rayleigh_taylor_twofluid import (  # noqa: E402
    SCMPRTConfig,
    _perturb_density,
    atwood_from_ratio,
    column_front_state,
    gradient_front_state,
    layzer_velocity,
    precondition_scmp_rt,
    run_scmp_rt,
    scmp_coexistence,
)

OUT = HERE / "artifacts_estimator"
SC_G = 2.8                  # coupling of the case calibration (A = 0.505 at ny = 64, 3000 steps)
TAU = 1.0                   # lowest spurious-current floor of the tau scan (m21884 P2)
LAM = 64
LAYERS = 4.0
RELAX_STEPS = 600
G = 2.0e-4
A0_OVER_LAM = 0.02


def _cfg(layers: float = LAYERS, **kw) -> SCMPRTConfig:
    base = dict(lam=LAM, layers=layers, gravity=G, tau_relax=TAU,
                a0_over_lam=A0_OVER_LAM, steps=200, sc_g=SC_G, sample_every=10)
    base.update(kw)
    return SCMPRTConfig(**base)


def _relaxed_profile(cfg: SCMPRTConfig) -> tuple[torch.Tensor, dict]:
    """Flat relaxed density field (gravity off) and its preconditioning info."""
    f, info = precondition_scmp_rt(cfg, rho_high=1.2078184525247444,
                                   rho_low=0.2545369048466741,
                                   relax_steps=RELAX_STEPS, sample_every=100)
    return f.sum(dim=0), info


def _read_both(rho: torch.Tensor, cfg: SCMPRTConfig, rho_high: float, rho_low: float,
               ref_col: float | None, ref_grad: float | None) -> dict:
    col = column_front_state(rho, cfg, rho_high=rho_high, rho_low=rho_low,
                             reference_row=ref_col)
    grad = gradient_front_state(rho, cfg, rho_high=rho_high, rho_low=rho_low,
                                reference_row=ref_grad)
    return {
        "column_amp_lam": col["amplitude_lam"],
        "column_spike_q10_lam": col["spike_q10_lam"],
        "column_common_lam": col["common_mode_lam"],
        "gradient_amp_lam": grad["amplitude_lam"],
        "gradient_spike_q10_lam": grad["spike_q10_lam"],
        "gradient_common_lam": grad["common_mode_lam"],
        "gradient_lock_spread_rows": grad["grad_lock_spread_rows"],
        "gradient_peak_ratio": grad["grad_peak_ratio"],
    }


def check_k1_k3(cfg: SCMPRTConfig, rho_relaxed: torch.Tensor,
                rho_high: float, rho_low: float) -> dict:
    """Seed readback (K1) and hydrostatic common-mode rejection (K3)."""
    # reference levels: the flat relaxed interface, per estimator
    ref_col = float(
        (cfg.ny - 1) - ((rho_relaxed[1:-1, :] - rho_low) / (rho_high - rho_low))
        .clamp(0.0, 1.0).sum(dim=0).mean()
    )
    ref_grad = float(
        gradient_front_state(rho_relaxed, cfg, rho_high=rho_high, rho_low=rho_low)["mean_interface_lam"]
        * cfg.lam + cfg.y_interface
    )
    zeros = _read_both(rho_relaxed, cfg, rho_high, rho_low, ref_col, ref_grad)

    k1 = {"zero_state": zeros, "sweep": []}
    for a_lam in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        probe_cfg = _cfg(a0_over_lam=a_lam)
        rho_seeded = _perturb_density(rho_relaxed, probe_cfg, rho_high=rho_high,
                                      rho_low=rho_low)
        row = _read_both(rho_seeded, cfg, rho_high, rho_low, ref_col, ref_grad)
        row["a0_over_lam"] = a_lam
        for est in ("column", "gradient"):
            row[f"{est}_err_pct"] = 100.0 * (row[f"{est}_amp_lam"] / a_lam - 1.0)
        k1["sweep"].append(row)

    k3 = []
    # hydrostatic-like compression: rho -> rho (1 + eps y/H), eps = g H / c_s^2 * n
    eps0 = G * cfg.ny / (1.0 / 3.0)
    for mult in (1.0, 3.0, -1.0):
        eps = mult * eps0
        y = torch.arange(cfg.ny, dtype=rho_relaxed.dtype).view(cfg.ny, 1)
        rho_c = _perturb_density(rho_relaxed, cfg, rho_high=rho_high, rho_low=rho_low)
        rho_c = rho_c * (1.0 + eps * y / cfg.ny)
        rho_c[0, :] = 0.0
        rho_c[-1, :] = 0.0
        row = _read_both(rho_c, cfg, rho_high, rho_low, ref_col, ref_grad)
        row.update({"eps": eps, "eps_over_cs2": eps, "mult": mult})
        for est in ("column", "gradient"):
            row[f"{est}_err_pct"] = 100.0 * (row[f"{est}_amp_lam"] / A0_OVER_LAM - 1.0)
        k3.append(row)

    # wall depletion layer: the artefact that made the phi = 0 crossing invent a second
    # crossing (m21884 L3); the gradient estimator must be blind to it
    rho_w = _perturb_density(rho_relaxed, cfg, rho_high=rho_high, rho_low=rho_low)
    rho_w[1:5, :] = 0.05
    wall = _read_both(rho_w, cfg, rho_high, rho_low, ref_col, ref_grad)
    # interface geometry of the *relaxed* model interface: the sharp-interface
    # reference (Layzer-Goncharov) requires k delta << 1
    prof = rho_relaxed.mean(dim=1)
    # the profile is not monotone (near-wall wetting bumps), so the transition is
    # located at the strongest gradient and its 10 %/90 % levels are taken against
    # the local bulks found by walking outwards from that point
    dprof = torch.zeros_like(prof)
    dprof[1:-1] = 0.5 * (prof[2:] - prof[:-2])
    inner = dprof.abs().clone()
    inner[:10] = 0.0
    inner[-10:] = 0.0           # wall layers have their own (steeper) density drop
    r_star = int(torch.argmax(inner))
    bulk_high = float(prof[r_star + 25]) if r_star + 25 < cfg.ny - 1 else float(prof.max())
    bulk_low = float(prof[r_star - 25]) if r_star > 26 else float(prof.min())
    delta = bulk_high - bulk_low
    lo_level = bulk_low + 0.1 * delta
    hi_level = bulk_low + 0.9 * delta
    lo_row = r_star
    while lo_row > 1 and float(prof[lo_row]) > lo_level:
        lo_row -= 1
    hi_row = r_star
    while hi_row < cfg.ny - 2 and float(prof[hi_row]) < hi_level:
        hi_row += 1
    width_rows = hi_row - lo_row
    k_delta = 2.0 * math.pi / LAM * width_rows
    geometry = {
        "transition_row": r_star,
        "bulk_low_measured": bulk_low,
        "bulk_high_measured": bulk_high,
        "bulk_low_target": float(rho_low),
        "bulk_high_target": float(rho_high),
        "ten_ninety_rows": width_rows,
        "ten_ninety_lam": width_rows / LAM,
        "k_delta": k_delta,
        "sharp_interface_ok": bool(k_delta < 0.2),
        "profile_every_8_rows": [round(float(prof[i]), 4) for i in range(0, cfg.ny, 8)],
    }
    return {"relax_geometry": geometry,
            "K1": k1, "K3": {"compression": k3, "wall_layer": wall,
                             "eps0": eps0, "reference_col_row": ref_col,
                             "reference_grad_row": ref_grad}}


def check_k2(cfg: SCMPRTConfig, rho_high: float, rho_low: float,
             u_ref: float) -> dict:
    """Null dynamics: staged start-up, seeded mode, gravity off."""
    null_cfg = _cfg(layers=cfg.layers, steps=600)
    res = run_scmp_rt(null_cfg, rho_high=rho_high, rho_low=rho_low,
                      precondition_steps=RELAX_STEPS, ramp_steps=0,
                      gravity_scale=0.0, ma_guard="bulk", front_estimator="gradient")
    hist = res["history"]
    amps = [abs(r["amplitude_lam"]) for r in hist if r.get("amplitude_lam") is not None]
    tips = [max(abs(r["spike_q10_lam"]), abs(r["bubble_q90_lam"])) for r in hist]
    seed = hist[0] if hist else {}
    return {
        "steps": res["steps_run"],
        "stop_reason": res["stop_reason"],
        "planned_a0_over_lam": null_cfg.a0_over_lam,
        "readback_a0_over_lam": seed.get("amplitude_lam"),
        "readback_err_pct": (100.0 * (seed["amplitude_lam"] / null_cfg.a0_over_lam - 1.0)
                             if seed.get("amplitude_lam") is not None else float("nan")),
        "amp_drift_lam": (max(amps) - min(amps)) if amps else float("nan"),
        "amp_max_abs_lam": max(amps) if amps else float("nan"),
        "front_drift_lam": max(tips) if tips else float("nan"),
        "u_ref_noise": u_ref,
        "noise_over_u_ref": (max(tips) / u_ref if u_ref else float("nan")),
        "ma_peak": res["ma_peak"],
        "u_spurious": res["ma_peak"] * math.sqrt(1.0 / 3.0),
    }


def check_k4() -> dict:
    """Coexistence (Atwood) convergence on two box/step settings."""
    runs = []
    for ny, steps in ((64, 3000), (64, 6000), (48, 3000), (96, 3000)):
        t0 = time.time()
        co = scmp_coexistence(SC_G, steps=steps, ny=ny, tau=TAU)
        d = co.as_dict()
        d.update({"wall_time_s": time.time() - t0})
        runs.append(d)
    at = [r["atwood"] for r in runs]
    return {
        "runs": runs,
        "atwood_spread": (max(at) - min(at)) if at else float("nan"),
        "atwood_min": min(at) if at else float("nan"),
        "atwood_max": max(at) if at else float("nan"),
        "u_spike_sensitivity_pct_per_dA": 100.0 * (
            layzer_velocity(max(at), G, LAM)["u_spike"]
            / layzer_velocity(min(at), G, LAM)["u_spike"] - 1.0
        ) if at else float("nan"),
    }


def check_k5(cfg: SCMPRTConfig, rho_high: float, rho_low: float) -> dict:
    """Physics attempt on the calibrated instrument."""
    res = run_scmp_rt(cfg, rho_high=rho_high, rho_low=rho_low,
                      precondition_steps=RELAX_STEPS, ramp_steps=400,
                      ma_guard="bulk", front_estimator="gradient",
                      amp_window=(0.03, 0.08))
    hist = res.pop("history")
    keep = ["name", "steps_run", "stop_reason", "atwood_measured", "u_spike_sim",
            "u_bubble_sim", "u_spike_ref", "u_bubble_ref", "asym_sim", "asym_ref",
            "err_spike_pct", "err_bubble_pct", "window_tau_span", "ma_peak",
            "ma_peak_bulk", "ma_peak_iface", "sigma_amp", "sigma_theory_visc",
            "err_sigma_pct", "mass_drift_pct", "terminal_drift_pct", "fit_spike_r2"]
    out = {k: res.get(k) for k in keep}
    out["history_tail"] = [
        {k: r.get(k) for k in ("t", "tau", "spike_q10_lam", "bubble_q90_lam",
                               "amplitude_lam", "common_mode_lam", "ma_bulk", "ma_iface")}
        for r in hist[:: max(1, len(hist) // 25)]
    ]
    return out


def _markdown(payload: dict) -> str:
    k1 = payload["K1"]["sweep"]
    lines = [
        "# Front-estimator calibration (CASE-RT-INSTABILITY-ATWOOD, finite-density-ratio arm)",
        "",
        f"`sc_g = {SC_G}`, `tau = {TAU}`, `{LAM} x {int(LAYERS * LAM)}` box, "
        f"`precondition_steps = {RELAX_STEPS}`, `g = {G}` (lattice).",
        "",
        "## K1 seed readback (relaxed profile displaced by a known single mode)",
        "",
        "| a0/lam | column amp | column err | gradient amp | gradient err | "
        "lock spread [rows] | peak ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in k1:
        lines.append(
            f"| {r['a0_over_lam']:.3f} | {r['column_amp_lam']:+.5f} | {r['column_err_pct']:+.1f}% | "
            f"{r['gradient_amp_lam']:+.5f} | {r['gradient_err_pct']:+.1f}% | "
            f"{r['gradient_lock_spread_rows']} | {r['gradient_peak_ratio']:.3f} |"
        )
    lines += [
        "",
        "## K3 hydrostatic common mode (profile x (1 + eps y/H))",
        "",
        "| eps | column amp | column err | column common | gradient amp | gradient err | "
        "gradient common |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in payload["K3"]["compression"]:
        lines.append(
            f"| {r['eps']:+.3e} | {r['column_amp_lam']:+.5f} | {r['column_err_pct']:+.1f}% | "
            f"{r['column_common_lam']:+.5f} | {r['gradient_amp_lam']:+.5f} | "
            f"{r['gradient_err_pct']:+.1f}% | {r['gradient_common_lam']:+.5f} |"
        )
    geo = payload.get("relax_geometry", {})
    lines += [
        "",
        "## Relaxed-interface geometry (sharp-interface reference validity)",
        "",
        f"10-90 % transition width {geo.get('ten_ninety_rows')} rows = "
        f"{geo.get('ten_ninety_lam')} lam -> k delta = {geo.get('k_delta')}; "
        f"sharp interface (k delta < 0.2) ok: {geo.get('sharp_interface_ok')}. "
        f"Measured bulks rho_low ~ {geo.get('bulk_low_measured')}, rho_high ~ "
        f"{geo.get('bulk_high_measured')} against the coexistence targets "
        f"{geo.get('bulk_low_target')} / {geo.get('bulk_high_target')}.",
        "",
    ]
    w = payload["K3"]["wall_layer"]
    if "K2" not in payload:
        lines += ["", "(K2 not requested)", ""]
    lines += [
        "",
        f"Wall depletion layer (rows 1-4 set to 0.05): column amp "
        f"{w['column_amp_lam']:+.5f} / spike {w['column_spike_q10_lam']:+.5f}; gradient amp "
        f"{w['gradient_amp_lam']:+.5f} / spike {w['gradient_spike_q10_lam']:+.5f}.",
        "",
    ] + ([
        f"## K2 null dynamics (g = 0, {payload['K2']['steps']} steps, stop="
        f"{payload['K2']['stop_reason']})",
        "",
        f"readback a0 {payload['K2']['readback_a0_over_lam']:+.5f} "
        f"({payload['K2']['readback_err_pct']:+.1f}%), amplitude wander "
        f"{payload['K2']['amp_drift_lam']:.4f} lam, front drift "
        f"{payload['K2']['front_drift_lam']:.4f} lam vs u_s ref {payload['K2']['u_ref_noise']:.4f} "
        f"-> noise/u_s = {payload['K2']['noise_over_u_ref']:.2f}.",
        "",
    ] if "K2" in payload else []) + [
        f"## K4 coexistence convergence (sc_g = {SC_G}, tau = {TAU})",
        "",
        "| ny | steps | rho_high | rho_low | ratio | A | wall time [s] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in payload.get("K4", {}).get("runs", []):
        lines.append(f"| {r['ny']} | {r['steps']} | {r['rho_high']:.4f} | {r['rho_low']:.4f} | "
                     f"{r['ratio']:.4f} | {r['atwood']:.4f} | {r['wall_time_s']:.1f} |")
    k4 = payload.get("K4", {})
    lines += [
        "",
        f"A spread {k4.get('atwood_spread', float('nan')):.4f} -> u_s reference changes "
        f"{k4.get('u_spike_sensitivity_pct_per_dA', float('nan')):+.1f}% across the spread.",
        "",
        "## K5 physics attempt",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    for k, v in payload.get("K5", {}).items():
        if k == "history_tail":
            continue
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None)
    ap.add_argument("--steps", type=int, default=4000, help="K5 physics steps")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip().upper() for s in args.only.split(",")} if args.only else \
        {"K1", "K2", "K3", "K4", "K5"}

    t_start = time.time()
    cfg = _cfg()
    rho_high, rho_low = 1.2078184525247444, 0.2545369048466741
    atwood = atwood_from_ratio(rho_high / rho_low)
    u_ref = layzer_velocity(atwood, G, LAM)["u_spike"]
    payload: dict = {
        "sc_g": SC_G, "tau": TAU, "lam": LAM, "layers": LAYERS,
        "relax_steps": RELAX_STEPS, "gravity": G, "a0_over_lam": A0_OVER_LAM,
        "rho_high_calib": rho_high, "rho_low_calib": rho_low,
        "atwood_calib": atwood, "u_spike_ref": u_ref,
    }
    if {"K1", "K2", "K3"} & wanted:
        rho_relaxed, info = _relaxed_profile(cfg)
        payload["relax_info"] = {k: v for k, v in info.items() if k != "relax_trace"}
        payload["relaxed_density"] = {"rho_min": float(rho_relaxed[1:-1].min()),
                                      "rho_max": float(rho_relaxed[1:-1].max())}
        if {"K1", "K3"} & wanted:
            payload.update(check_k1_k3(cfg, rho_relaxed, rho_high, rho_low))
        if "K2" in wanted:
            payload["K2"] = check_k2(cfg, rho_high, rho_low, u_ref)
    if "K4" in wanted:
        payload["K4"] = check_k4()
    if "K5" in wanted:
        payload["K5"] = check_k5(_cfg(layers=6.0, steps=args.steps), rho_high, rho_low)
    payload["wall_time_s"] = time.time() - t_start

    (out_root / "calibration.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_root / "calibration.md").write_text(_markdown(payload))
    print(json.dumps({k: v for k, v in payload.items()
                      if k in ("K2", "K5", "wall_time_s", "relaxed_density")},
                     indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())