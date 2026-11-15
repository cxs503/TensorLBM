#!/usr/bin/env python3
"""Round-3 fragment: the wall-bounded (Poiseuille) half of the predeclared ladder.

Round 2 (`docs/XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL_R2.md` §6/§7) shipped the Taylor-Green half of the
predeclared ladder but left the channel half *withheld*: the Ny in {16,24}, 30000-step cases never ran
(compute job cj2867 was still queued at the mission wall). This fragment closes that debt and, because a
withheld case is expensive to run twice, extends it along two axes that the round-2 report could only
speculate about:

  L1  Ny ladder      ny in {12, 16, 24, 32}, 30000 steps, both transport arms   -> the withheld gate
  L2  spin-up        umax(t) sampled every 500 steps inside the *same* run      -> is 30000 steps steady?
  L3  Mach sweep     umax_target in {.005,.01,.02,.04} at ny=16                 -> attribute the residual

L2 matters because a single-shot check at 3000 steps showed ny=32 still 3.8 % off the analytic peak while
ny=16 was 0.14 % off; without the trajectory one cannot tell a discretisation error from an unfinished
transient, and the predeclared stopping rule ("30000 steps = 15 viscous times") was written for H=14 (ny=16).

Nothing is re-implemented: transport / collision / forcing / bounce come from ``run_xval``. The only new
code is the sampling loop, and its final state is checked bit-wise against ``run_xval.run_poiseuille``
(item ``crosscheck``) so a transcription slip in the loop cannot pass silently.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_xval as R  # noqa: E402

OUT = HERE.parent.parent / "runs" / "xval_conv_as_streaming_r3"

#: predeclared before any ladder number was seen (round-2 wording, unchanged thresholds)
LADDER_NY = [12, 16, 24, 32]
LADDER_STEPS = 30000
LADDER_TAU = 0.8
LADDER_UMAX = 0.02
MA_SWEEP = [0.005, 0.01, 0.02, 0.04]
MA_STEPS = 20000
SAMPLE_EVERY = 500


def run_poiseuille_sampled(
    ny: int, nx: int, tau: float, umax_target: float, steps: int, arm: str, oss: dict,
    sample_every: int = SAMPLE_EVERY,
) -> dict:
    """Identical physics to ``run_xval.run_poiseuille`` + a umax(t) trajectory.

    Verified bit-wise against the reference runner in :func:`crosscheck`.
    """
    dt = torch.float64
    torch.set_num_threads(max(1, 4))
    transport = R.make_transport(arm if arm == "A_native_gather" else "B_conv_fixed_kernel", oss)
    nu = (tau - 0.5) / 3.0
    height = float(ny - 2)
    g = 8.0 * nu * umax_target / height**2
    force = (g, 0.0)
    mask = torch.zeros(ny, nx, dtype=dt)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    opposite = R.OPPOSITE.to(torch.int64)

    f = R.equilibrium(
        torch.ones(ny, nx, dtype=dt), torch.zeros(ny, nx, dtype=dt), torch.zeros(ny, nx, dtype=dt)
    )
    samples: list[dict] = []
    half_profile = None
    wall0 = time.perf_counter()
    with torch.no_grad():
        for step in range(1, steps + 1):
            f_streamed = transport(f)
            f_collided = R.collide_bgk(f_streamed, tau) + R.guo_term(f_streamed, force, tau)
            if arm == "A_native_gather":
                f_new = f_collided.clone()
                f_new[:, mask > 0] = f_streamed[opposite][:, mask > 0]
            else:
                f_new = R.SC.bounce_blend(f_collided, f_streamed, mask, opposite=opposite)
            f = f_new
            if step % sample_every == 0 or step == steps:
                _, ux_s, _ = R.velocity_with_force(f, force)
                prof = ux_s.numpy()[1:-1, nx // 2]
                y_index = np.arange(1, ny - 1)
                y_pos = y_index - 0.5
                exact = g / (2.0 * nu) * y_pos * (height - y_pos)
                samples.append(
                    {
                        "step": step,
                        "umax": float(prof.max()),
                        "peak_rel_err_pct": 100.0 * float(abs(prof.max() - exact.max()) / exact.max()),
                    }
                )
                if step == steps // 2:
                    half_profile = prof.copy()
    wall = time.perf_counter() - wall0

    rho, ux, uy = R.velocity_with_force(f, force)
    prof = ux.numpy()[1:-1, nx // 2]
    y_index = np.arange(1, ny - 1)
    y_pos = y_index - 0.5
    exact = g / (2.0 * nu) * y_pos * (height - y_pos)
    return {
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "height": height,
        "g": g,
        "umax_target": umax_target,
        "umax_analytic": g * height**2 / (8.0 * nu),
        "umax_got": float(prof.max()),
        "re_channel": umax_target * height / nu,
        "ma_channel": float(umax_target / np.sqrt(1.0 / 3.0)),
        "steps": steps,
        "steps_per_viscous_time": steps / (height**2 / nu),
        "profile_l2_pct": 100.0 * float(np.linalg.norm(prof - exact) / np.linalg.norm(exact)),
        "peak_rel_err_pct": 100.0 * float(abs(prof.max() - exact.max()) / exact.max()),
        "half_step_profile_rel_dev_pct": (
            None
            if half_profile is None
            else 100.0 * float(np.linalg.norm(prof - half_profile) / np.linalg.norm(prof))
        ),
        "samples": samples,
        "profile": prof.tolist(),
        "exact": exact.tolist(),
        "finite": bool(torch.isfinite(f).all()),
        "f_sha256": R.f_digest(f),
        "wall_s": round(wall, 2),
    }


def crosscheck(oss: dict) -> dict:
    """The sampling loop must reproduce the reference runner bit-wise (same steps -> same state)."""
    out = {}
    for ny, steps in ((12, 2000), (16, 2000)):
        for arm in ("A_native_gather", "B_conv_fixed_kernel"):
            kw = dict(ny=ny, nx=max(16, ny), tau=LADDER_TAU, umax_target=LADDER_UMAX, steps=steps)
            a = R.run_poiseuille(arm=arm, oss=oss, **kw)
            b = run_poiseuille_sampled(arm=arm, oss=oss, sample_every=1000, **kw)
            out[f"ny{ny}_{arm}"] = {
                "identical_populations": a["f_sha256"] == b["f_sha256"],
                "peak_err_abs_diff": abs(a["peak_rel_err_pct"] - b["peak_rel_err_pct"]),
                "profile_l2_abs_diff": abs(a["profile_l2_pct"] - b["profile_l2_pct"]),
            }
    return out


def _case_path(tag: str) -> Path:
    d = OUT / "cases"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.json"


def run_one(tag: str, kind: str, oss: dict, **kw) -> dict:
    c = run_poiseuille_sampled(oss=oss, **kw)
    c["tag"] = tag
    c["kind"] = kind
    _case_path(tag).write_text(json.dumps(c, indent=1) + "\n")
    print(json.dumps({"tag": tag, "peak_pct": round(c["peak_rel_err_pct"], 5),
                      "l2_pct": round(c["profile_l2_pct"], 5), "wall_s": c["wall_s"]}), flush=True)
    return c


def assemble(oss: dict) -> dict:
    """Collect the per-case files into the fragment consumed by the renderer."""
    cases = sorted((OUT / "cases").glob("*.json"))
    if not cases:
        raise SystemExit("no per-case results to assemble")
    got = [json.loads(p.read_text()) for p in cases]

    ladder = [c for c in got if c["kind"] == "ladder"]
    ladder.sort(key=lambda c: (c["ny"], c["arm"]))
    mach = [c for c in got if c["kind"] == "mach"]
    # the Mach sweep's u_max = 0.02 point *is* the Ny=16 ladder pair -- reuse it, do not re-run it
    have = {(c["umax_target"], c["arm"]) for c in mach}
    for c in ladder:
        if c["ny"] == 16 and (c["umax_target"], c["arm"]) not in have:
            d = dict(c)
            d["kind"] = "mach_reused_from_ladder"
            mach.append(d)
    mach.sort(key=lambda c: (c["umax_target"], c["arm"]))

    xc_path = OUT / "crosscheck.json"
    xcheck = json.loads(xc_path.read_text()) if xc_path.exists() else crosscheck(oss)
    xc_path.write_text(json.dumps(xcheck, indent=1) + "\n")

    pairs = []
    for ny in sorted({c["ny"] for c in ladder}):
        a = next((c for c in ladder if c["ny"] == ny and c["arm"] == "A_native_gather"), None)
        b = next((c for c in ladder if c["ny"] == ny and c["arm"].startswith("B")), None)
        if a is None or b is None:
            continue
        ta = np.array([s["umax"] for s in a["samples"]])
        tb = np.array([s["umax"] for s in b["samples"]])
        pairs.append({
            "ny": ny,
            "identical_populations": a["f_sha256"] == b["f_sha256"],
            "max_traj_abs_diff": float(np.max(np.abs(ta - tb))) if ta.shape == tb.shape else None,
            "peak_rel_err_pct_spread": abs(a["peak_rel_err_pct"] - b["peak_rel_err_pct"]),
        })

    frag = {
        "anchor": R.ANCHOR,
        "fragment": "channel_ladder",
        "round": 3,
        "predeclared": {
            "ladder_ny": sorted({c["ny"] for c in ladder}) or LADDER_NY,
            "ladder_steps": LADDER_STEPS,
            "tau": LADDER_TAU,
            "umax_target": LADDER_UMAX,
            "ma_sweep": sorted({c["umax_target"] for c in mach}) or MA_SWEEP,
            "ma_steps": MA_STEPS,
            "sample_every": SAMPLE_EVERY,
            "thresholds_pct": {"owner_gate": 15.0, "internal_target": 3.0},
        },
        "crosscheck": xcheck,
        "ladder": ladder,
        "ladder_pairs": pairs,
        "mach_sweep": mach,
        "n_cases": len(got),
    }
    (OUT / "channel_ladder_fragment.json").write_text(json.dumps(frag, indent=1) + "\n")
    worst = max((max(c["peak_rel_err_pct"], c["profile_l2_pct"]) for c in ladder), default=float("nan"))
    print(json.dumps({
        "cases": len(got),
        "ladder_cases": len(ladder),
        "mach_cases": len(mach),
        "worst_ladder_kpi_pct": round(worst, 4),
        "all_pairs_identical": all(p["identical_populations"] for p in pairs),
        "crosscheck_ok": all(v["identical_populations"] for v in xcheck.values()),
    }, indent=1))
    return frag


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="run a single case: 'ladder:<ny>:<arm>' or 'mach:<umax>:<arm>'")
    ap.add_argument("--assemble", action="store_true", help="merge per-case files into the fragment")
    ap.add_argument("--crosscheck", action="store_true", help="only the sampling-loop cross-check")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    oss = R.load_oss("D2Q9")

    if args.crosscheck:
        xcheck = crosscheck(oss)
        (OUT / "crosscheck.json").write_text(json.dumps(xcheck, indent=1) + "\n")
        print(json.dumps(xcheck, indent=1))
        return 0

    if args.case:
        kind, a, arm = args.case.split(":")
        if kind == "ladder":
            ny = int(a)
            run_one(f"ladder_ny{ny}_{arm}", "ladder", oss, ny=ny, nx=max(16, ny), tau=LADDER_TAU,
                    umax_target=LADDER_UMAX, steps=LADDER_STEPS, arm=arm)
        elif kind == "mach":
            um = float(a)
            run_one(f"mach_u{um:g}_{arm}", "mach", oss, ny=16, nx=16, tau=LADDER_TAU,
                    umax_target=um, steps=MA_STEPS, arm=arm)
        else:
            raise SystemExit(f"unknown case kind {kind!r}")
        return 0

    if args.assemble:
        assemble(oss)
        return 0

    t0 = time.time()
    xcheck = crosscheck(oss)
    (OUT / "crosscheck.json").write_text(json.dumps(xcheck, indent=1) + "\n")
    print(json.dumps({"crosscheck": xcheck}), flush=True)

    for ny in LADDER_NY:
        for arm in ("A_native_gather", "B_conv_fixed_kernel"):
            run_one(f"ladder_ny{ny}_{arm}", "ladder", oss, ny=ny, nx=max(16, ny), tau=LADDER_TAU,
                    umax_target=LADDER_UMAX, steps=LADDER_STEPS, arm=arm)

    for um in MA_SWEEP:
        for arm in ("A_native_gather", "B_conv_fixed_kernel"):
            run_one(f"mach_u{um:g}_{arm}", "mach", oss, ny=16, nx=16, tau=LADDER_TAU,
                    umax_target=um, steps=MA_STEPS, arm=arm)

    assemble(oss)
    print(json.dumps({"wall_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
