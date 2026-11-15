"""Probe: where does the SCMP RT case's Mach spike come from?

``run_twofluid.py`` (mission m21799) stopped both finite-density-ratio runs at
step 5 with ``ma_peak`` 0.26 / 0.30 against the ``ma_limit`` 0.20 guard, i.e.
*before* the interface moved at all.  The case report blames "parasitic currents
from a tanh profile whose width does not match the model's equilibrium interface
width".  This probe tests that claim by separating the two candidate sources:

P1  profile mismatch -- initialise with the *measured* equilibrium (coexistence)
    profile instead of a hand-picked ``tanh(width=2)``;
P2  force imbalance -- run with ``g = 0`` (no gravity at all) and watch the same
    metric: if it still spikes, gravity and the interface perturbation cannot be
    the cause.

Diagnostics are split per the case report's next-step item 2: ``ma_bulk`` (away
from the interface band) is the physical compressibility check, while
``ma_iface`` (inside the band) is a pseudopotential artefact and is *reported*,
not gated.

Usage::

    python3 probe_spurious.py            # ~1 min on 4 threads
    python3 probe_spurious.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO / "src"), str(REPO)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.rayleigh_taylor_twofluid import (  # noqa: E402
    CS2,
    scmp_coexistence,
    scmp_step,
    velocity_field,
)

SC_G = 2.8
TAU = 1.0
LAM = 64
LAYERS = 4.0
GRAV = 7.5e-5


def equilibrium_profile(sc_g: float = SC_G, tau: float = TAU, ny: int = 96,
                        width: float = 4.0, steps: int = 4000) -> dict:
    """Relax a flat slab with ``g = 0`` and return the measured 1-D profile."""
    dtype = torch.float64
    x = torch.arange(4, dtype=dtype).view(1, 4)
    y = torch.arange(ny, dtype=dtype).view(ny, 1)
    mid = 0.5 * (ny - 1)
    rho = 0.65 + 0.45 * torch.tanh((y - mid) / width).expand(ny, 4).contiguous()
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    mass0 = float(rho.sum())
    for _ in range(steps):
        f = scmp_step(f, tau=tau, sc_g=sc_g, gravity_y=0.0)
    r = f.sum(dim=0).mean(dim=1)
    rho_hi = float(r[ny // 2])
    rho_lo = float(r[3])
    # width: distance over which (rho - rho_lo)/(rho_hi - rho_lo) runs 0.9 -> 0.1
    frac = ((r - rho_lo) / (rho_hi - rho_lo)).clamp(0.0, 1.0)
    n = frac.numel()
    rows90 = (frac >= 0.9).nonzero().flatten()
    rows10 = (frac <= 0.1).nonzero().flatten()
    w = float(rows90.max() - rows10.min()) if rows90.numel() and rows10.numel() else float("nan")
    return {
        "profile": [float(v) for v in r],
        "rho_high": rho_hi,
        "rho_low": rho_lo,
        "ratio": rho_hi / rho_lo,
        "atwood": (rho_hi - rho_lo) / (rho_hi + rho_lo),
        "width_10_90": w,
        "mass_drift_pct": 100.0 * (float(r.sum() * 4) - mass0) / mass0,
    }


def init_rt(rho_high: float, rho_low: float, *, a0_over_lam: float,
            width: float | None, profile: list[float] | None):
    """Build ``(f, rho, y_int)`` for the single-mode RT box (heavy on top)."""
    dtype = torch.float64
    ny, nx = int(LAYERS * LAM), LAM
    y = torch.arange(ny, dtype=dtype).view(ny, 1)
    x = torch.arange(nx, dtype=dtype).view(1, nx)
    y_int = 0.5 * ny + a0_over_lam * LAM * torch.cos(2.0 * math.pi * x / LAM)

    if profile is None:
        mid_rho = 0.5 * (rho_high + rho_low)
        half_amp = 0.5 * (rho_high - rho_low)
        rho = mid_rho + half_amp * torch.tanh((y - y_int) / float(width or 2.0))
    else:
        # Map the measured equilibrium profile onto the *signed distance* to the
        # interface: the slab profile is symmetric about its mid row, so take the
        # upper half as rho(distance), then evaluate it at |y - y_int|.
        prof = torch.tensor(profile, dtype=dtype)
        n = prof.numel()
        half = prof[(n - 1) // 2:]              # mid row -> high-density end
        dist = torch.arange(half.numel(), dtype=dtype)
        d = (y - y_int).abs().expand(ny, nx).contiguous()
        idx = torch.searchsorted(dist, d.contiguous().clamp(0, float(dist[-1])))
        i0 = (idx - 1).clamp(0, half.numel() - 2)
        t = (d - dist[i0]) / (dist[i0 + 1] - dist[i0])
        rho = half[i0] * (1 - t) + half[i0 + 1] * t

    rho = rho.clamp(min=0.0)
    rho[0, :] = 0.0
    rho[-1, :] = 0.0
    z = torch.zeros_like(rho)
    return equilibrium(rho, z, z), rho, y_int


def run_probe(label: str, rho_high: float, rho_low: float, *, a0_over_lam: float,
              width: float | None, profile: list[float] | None, gravity: float,
              steps: int = 60, sample_every: int = 5) -> dict:
    f, rho0, y_int = init_rt(rho_high, rho_low, a0_over_lam=a0_over_lam,
                             width=width, profile=profile)
    ny, nx = rho0.shape
    band = torch.zeros_like(rho0, dtype=torch.bool)
    for i in range(nx):
        c = float(y_int[0, i])
        lo = max(1, int(c) - 6)
        hi = min(ny - 1, int(c) + 7)
        band[lo:hi, i] = True
    mass0 = float(f.sum())
    trace = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f = scmp_step(f, tau=TAU, sc_g=SC_G, gravity_y=-gravity)
        if step % sample_every and step != steps:
            continue
        rho, ux, uy = velocity_field(f, sc_g=SC_G, gravity_y=-gravity)
        speed = torch.sqrt(ux ** 2 + uy ** 2)
        fluid = torch.ones_like(rho, dtype=torch.bool)
        fluid[0, :] = False
        fluid[-1, :] = False
        trace.append({
            "step": step,
            "ma_all": float(speed[fluid].max()) / math.sqrt(CS2),
            "ma_iface": float(speed[band & fluid].max()) / math.sqrt(CS2),
            "ma_bulk": float(speed[fluid & ~band].max()) / math.sqrt(CS2),
            "mass_drift_pct": 100.0 * (float(f.sum()) - mass0) / mass0,
        })
    return {
        "label": label,
        "gravity": gravity,
        "profile_init": profile is not None,
        "init_width": width,
        "steps": steps,
        "wall_time_s": time.time() - t0,
        "ma_peak_iface": max(r["ma_iface"] for r in trace),
        "ma_peak_bulk": max(r["ma_bulk"] for r in trace),
        "ma_final_iface": trace[-1]["ma_iface"],
        "ma_final_bulk": trace[-1]["ma_bulk"],
        "mass_drift_pct": trace[-1]["mass_drift_pct"],
        "trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "artifacts_twofluid" / "probe_spurious.json"))
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    print("[1/3] relaxing a flat slab (g=0) to the model's equilibrium profile ...")
    prof = equilibrium_profile()
    print(f"      rho_high={prof['rho_high']:.4f} rho_low={prof['rho_low']:.4f} "
          f"ratio={prof['ratio']:.3f} A={prof['atwood']:.3f} "
          f"width_10_90={prof['width_10_90']:.2f} cells")

    out = {"equilibrium_profile": {k: v for k, v in prof.items() if k != "profile"},
           "profile": prof["profile"], "sc_g": SC_G, "tau": TAU,
           "lam": LAM, "layers": LAYERS, "cases": []}

    hi, lo = prof["rho_high"], prof["rho_low"]
    combos = [
        ("tanh_w2.0_g_on", dict(width=2.0, profile=None, gravity=GRAV)),
        ("tanh_w2.0_g_off", dict(width=2.0, profile=None, gravity=0.0)),
        ("equil_profile_g_off", dict(width=None, profile=prof["profile"], gravity=0.0)),
        ("equil_profile_g_on", dict(width=None, profile=prof["profile"], gravity=GRAV)),
    ]
    for label, kw in combos:
        print(f"[2/3] {label} ...", flush=True)
        res = run_probe(label, hi, lo, a0_over_lam=0.02, steps=args.steps, **kw)
        out["cases"].append(res)
        print(f"      ma_iface_peak={res['ma_peak_iface']:.4f} "
              f"ma_bulk_peak={res['ma_peak_bulk']:.4f} "
              f"({res['wall_time_s']:.1f}s)")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[3/3] wrote {args.json}")

    print("\nsummary (ma = |u|/cs, gate candidate: bulk):")
    print(f"{'case':24s} {'ma_iface':>9s} {'ma_bulk':>9s}")
    for c in out["cases"]:
        print(f"{c['label']:24s} {c['ma_peak_iface']:9.4f} {c['ma_peak_bulk']:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())