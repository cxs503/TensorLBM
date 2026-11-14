"""V3 verification: moving-wall / non-zero-velocity Dirichlet term of the alpha family.

Anchor: ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``.

What is new relative to ``run_xval.py`` (mission m18218) and ``verify_v2.py``
(mission m20373)
-------------------------------------------------------------------------------
Both previous verifiers exercised **walls at rest only**.  For a wall at rest the
antisymmetric equilibrium term ``-alpha e^-_q(x_w)`` of the pylbm
``LinearBounceBack`` form vanishes identically, so the production operator
(``src/tensorlbm/boundary/interpolated_bounce_back.py``) legitimately omitted it
-- and ``OSS_DIFF.md`` listed the omission as the remaining gap "mandatory for
inclined / moving walls".  This driver closes that gap:

* the operator now accepts ``wall_velocity`` / ``wall_density`` and applies
  ``-alpha e^-_q(x_w)`` with ``e^-_q = 3 w_q rho_w (c_q . u_w)``;
* three benchmarks with exact analytic references exercise it, all at *sub-grid*
  wall distances ``delta_b`` / ``delta_t`` (``H = (N-1) + delta_b + delta_t``):

  ``couette``             sliding lower wall, ``F = 0``, ``u(y) = U_w (1 - y/H)``;
  ``forced``              force-driven plane Poiseuille, walls at rest,
                          ``u(y) = g/(2 nu) y (H-y)``  (benchmark of REPORT_V2);
  ``couette_poiseuille``  both at once, ``u(y) = U_w (1 - y/H) + g/(2 nu) y (H-y)``.

Arms
----
``none``   operator without the Dirichlet term (pre-patch behaviour = control);
``term``   ``u_w = U_w`` on the sliding wall, ``0`` on the fixed wall (Couette);
``half``   forced channel, stored wall velocity ``g/2`` (half-step force);
``lam``    forced channel, stored wall velocity ``lambda^- g`` (form used in
           REPORT_V2.md);
``zero``   forced channel, term present but zero (code-path control);
``couette_poiseuille``  ``u_w = U_w + g/2`` (moving) / ``g/2`` (fixed).

Conventions (identical to ``verify_v2.py`` so numbers are comparable)
--------------------------------------------------------------------
post-collision isotropic forcing ``3 w_q c_qx g`` on fluid cells only, collision
restricted to fluid cells, stored momentum ``m = sum f c``, equilibrium built from
``m/rho``, reported velocity ``m/rho - g/2``.

Diagnostics
-----------
``l2_rel`` / ``max_rel``  against the analytic profile (fluid rows only);
``l2_rel_1000``           same at a fixed earlier horizon, to expose drifts;
``dM_per_step``           total-mass leak of the boundary operator per step
                          (measured over the whole run, relative to initial mass);
``mass_conservation``     an independent single-step probe: mass change caused by
                          one operator application alone (no collision/streaming).
The initial condition is the analytic profile: the steady state is unique and
independent of it, and starting on it removes the slow viscous spin-up
(O(1/n) decay) from the budget.  A wrong closure still departs from the analytic
profile within a few steps and settles on its own biased state -- that is exactly
what the ``none`` arms measure.

Usage
-----
    PYTHONPATH=src python3 verify_v3_moving_wall.py --part couette
    PYTHONPATH=src python3 verify_v3_moving_wall.py --part forced
    PYTHONPATH=src python3 verify_v3_moving_wall.py --part couette_poiseuille
    PYTHONPATH=src python3 verify_v3_moving_wall.py --merge      # tables + figures
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
ANCHOR = "XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME"
DEFAULT_OUT = Path(__file__).resolve().parent / "artifacts_v3"

from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    alpha_bfl,
    alpha_cli,
    interpolated_bounce_back_2d,
)
from tensorlbm.d2q9 import C as C9  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE as OPP9  # noqa: E402
from tensorlbm.d2q9 import W as W9  # noqa: E402

DT = torch.float64
Q = int(C9.shape[0])
CI = C9.to(torch.int64)
CX = CI[:, 0].to(DT).view(Q, 1, 1)
CY = CI[:, 1].to(DT).view(Q, 1, 1)
WI = W9.to(DT).view(Q, 1, 1)

ALPHAS = {"bfl": alpha_bfl, "cli": alpha_cli, "halfway": lambda q: torch.full_like(q, 2.0)}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
class Channel:
    """Two straight walls at sub-grid distances ``delta_b`` / ``delta_t`` outside the rows."""

    def __init__(self, n_fluid: int, delta_b: float, delta_t: float, nx: int = 3, n_wall: int = 2):
        self.n_fluid, self.db, self.dt, self.nx = n_fluid, delta_b, delta_t, nx
        ny = n_fluid + 2 * n_wall
        self.height = (n_fluid - 1) + delta_b + delta_t
        y = torch.arange(ny, dtype=DT) - n_wall + delta_b
        sdf1d = torch.minimum(y, self.height - y)
        fluid1d = torch.zeros(ny, dtype=torch.bool)
        fluid1d[n_wall : n_wall + n_fluid] = True
        self.sdf = sdf1d.view(ny, 1).expand(ny, nx).contiguous()
        self.fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
        self.solid = ~self.fluid
        self.ny, self.nx = ny, nx
        self.y_fluid = y[n_wall : n_wall + n_fluid]
        self.rows = (n_wall, n_wall + n_fluid)

    def wall_velocity(self, u_bottom: float, u_top: float) -> torch.Tensor:
        """Stored-momentum wall velocity ``(ny, nx, 2)`` of the two cut rows."""
        wv = torch.zeros((self.ny, self.nx, 2), dtype=DT)
        wv[self.rows[0], :, 0] = u_bottom
        wv[self.rows[1], :, 0] = u_top
        return wv


# --------------------------------------------------------------------------- #
# LBM primitives (torch, float64)
# --------------------------------------------------------------------------- #
def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    cu = CX * ux + CY * uy
    usq = ux * ux + uy * uy
    return WI * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def moments(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return f.sum(0), (CX * f).sum(0), (CY * f).sum(0)


def collide(f, rho, ux, uy, carrier: str, nu: float, lam_m: float | None):
    feq = equilibrium(rho, ux, uy)
    if carrier == "bgk":
        tau = 0.5 + 3.0 * nu
        return f - (1.0 / tau) * (f - feq)
    opp = OPP9.to(torch.int64)
    lam_p = 3.0 * nu
    lm = 3.0 / 16.0 if lam_m is None else lam_m
    sp, sm = 1.0 / (0.5 + lam_p), 1.0 / (0.5 + lm)
    fp, fm = 0.5 * (f + f[opp]), 0.5 * (f - f[opp])
    ep, em = 0.5 * (feq + feq[opp]), 0.5 * (feq - feq[opp])
    return f - sp * (fp - ep) - sm * (fm - em)


def stream(f: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(f)
    for i in range(Q):
        out[i] = torch.roll(f[i], shifts=(int(CI[i, 1]), int(CI[i, 0])), dims=(0, 1))
    return out


def profile(f: torch.Tensor, g: float, ch: Channel, shift: float = -0.5) -> torch.Tensor:
    rho, mx, _ = moments(f)
    return (mx / rho)[ch.rows[0] : ch.rows[1], 0] + shift * g


def initial_state(ch: Channel, u1d: torch.Tensor, g: float):
    """Equilibrium at the analytic profile, stored momentum ``u + g/2``, fluid cells only."""
    u = torch.zeros((ch.ny, ch.nx), dtype=DT)
    u[ch.rows[0] : ch.rows[1], :] = (u1d + 0.5 * g).unsqueeze(1)
    rho = torch.ones((ch.ny, ch.nx), dtype=DT)
    zero = torch.zeros_like(u)
    f = equilibrium(rho, u, zero)
    return torch.where(ch.fluid.unsqueeze(0), f, WI)


def reference_profile(ch: Channel, *, u_wall: float, g: float, nu: float) -> torch.Tensor:
    yj = ch.y_fluid
    H = ch.height
    u = u_wall * (1.0 - yj / H)
    if g:
        u = u + g / (2.0 * nu) * yj * (H - yj)
    return u


def mass_probe(ch: Channel, alpha, wv, f0: torch.Tensor, f_pre: torch.Tensor) -> float:
    """Mass change of a single operator application (no collision / streaming involved)."""
    f_st = stream(f_pre)
    m0 = float(f_st.sum())
    out = interpolated_bounce_back_2d(f_st, f_pre, ch.sdf, ch.solid, alpha=alpha, wall_velocity=wv)
    return float(out.sum()) - m0


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def run_case(
    *,
    n_fluid: int,
    delta_b: float,
    delta_t: float,
    closure: str,
    carrier: str,
    arm: str,
    nu: float = 0.1,
    u_wall: float = 0.0,
    u_ref_max: float = 0.02,
    forced: bool = False,
    steps: int = 2000,
    probe_every: int = 1000,
    tol: float = 1e-14,
    stop_on_steady: bool = True,
) -> dict:
    ch = Channel(n_fluid, delta_b, delta_t)
    g = 8.0 * nu * u_ref_max / ch.height**2 if forced else 0.0
    src = (3.0 * WI * CX) * g
    src = torch.where(ch.fluid.unsqueeze(0), src, torch.zeros_like(src))
    af = ALPHAS[closure]
    lam_m = None if carrier == "bgk" else 3.0 / 16.0

    wv = None
    if arm == "term":
        wv = ch.wall_velocity(u_wall, 0.0)
    elif arm == "couette_poiseuille":
        wv = ch.wall_velocity(u_wall + 0.5 * g, 0.5 * g)
    elif arm == "half":
        wv = ch.wall_velocity(0.5 * g, 0.5 * g)
    elif arm == "lam":
        lm = lam_m if lam_m is not None else 3.0 * nu
        wv = ch.wall_velocity(lm * g, lm * g)
    elif arm == "zero":
        wv = ch.wall_velocity(0.0, 0.0)
    elif arm != "none":
        raise ValueError(arm)

    u0 = reference_profile(ch, u_wall=u_wall, g=g, nu=nu)
    f = initial_state(ch, u0, g)
    m0 = float(f.sum())
    probe = mass_probe(ch, af, wv, None, f) if wv is not None else mass_probe(ch, af, None, None, f)

    t0 = time.time()
    marks: dict[int, float] = {}
    steady_at = None
    prof_prev = None
    for n in range(1, steps + 1):
        rho, mx, my = moments(f)
        f_post = collide(f, rho, mx / rho, my / rho, carrier, nu, lam_m)
        f_post = torch.where(ch.fluid.unsqueeze(0), f_post + src, f)
        f = interpolated_bounce_back_2d(
            stream(f_post), f_post, ch.sdf, ch.solid, alpha=af, wall_velocity=wv,
        )
        if n % probe_every == 0:
            prof = profile(f, g, ch)
            u_ref = reference_profile(ch, u_wall=u_wall, g=g, nu=nu)
            marks[n] = float(torch.linalg.norm(prof - u_ref) / torch.linalg.norm(u_ref))
            if stop_on_steady and prof_prev is not None:
                if torch.max(torch.abs(prof - prof_prev)).item() < tol:
                    steady_at = n
                    break
            prof_prev = prof

    prof = profile(f, g, ch)
    u_ref = reference_profile(ch, u_wall=u_wall, g=g, nu=nu)
    scale = float(torch.max(torch.abs(u_ref)))
    offset = float(torch.mean(prof - u_ref))
    n_done = n
    return {
        "bench": "couette" if not forced else "couette_poiseuille",
        "n_fluid": n_fluid, "delta_b": delta_b, "delta_t": delta_t,
        "closure": closure, "carrier": carrier, "arm": arm,
        "nu": nu, "g": g, "u_wall": u_wall, "H": ch.height,
        "carrier_param": lam_m,
        "steps": n_done, "steady_at": steady_at,
        "l2_rel": float(torch.linalg.norm(prof - u_ref) / torch.linalg.norm(u_ref)),
        "l2_rel_1000": marks.get(1000),
        "max_rel": float(torch.max(torch.abs(prof - u_ref))) / scale,
        "mean_offset": offset,
        "mean_offset_over_scale": offset / scale,
        "mean_offset_over_g": (offset / g) if g else None,
        "dM_per_step": (float(f.sum()) - m0) / m0 / n_done,
        "mass_probe_1step": probe,
        "wall_time_s": time.time() - t0,
        "y": [float(v) for v in ch.y_fluid],
        "u": [float(v) for v in prof],
        "u_ref": [float(v) for v in u_ref],
    }


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #
COUETTE = [
    dict(delta_b=d, delta_t=d, closure=c, carrier=k, arm=a)
    for d in (0.1, 0.25, 0.5)
    for (k, c) in (("bgk", "bfl"), ("bgk", "cli"), ("bgk", "halfway"), ("trt_c", "bfl"), ("trt_c", "cli"))
    for a in ("none", "term")
] + [
    dict(delta_b=0.3, delta_t=0.7, closure=c, carrier="trt_c", arm=a)
    for c in ("bfl", "cli") for a in ("none", "term")
]

FORCED = [
    dict(delta_b=d, delta_t=d, closure=c, carrier=k, arm=a, forced=True)
    for d in (0.1, 0.25)
    for (k, c) in (("bgk", "bfl"), ("bgk", "cli"), ("trt_c", "bfl"))
    for a in ("none", "half", "lam", "zero")
]

CP = [
    dict(delta_b=d, delta_t=d, closure="bfl", carrier="bgk", arm=a, forced=True, u_wall=0.02)
    for d in (0.25, 0.7)
    for a in ("none", "couette_poiseuille")
]

PARTS = {"couette": COUETTE, "forced": FORCED, "couette_poiseuille": CP}


def run_part(name: str, out_dir: Path, steps: int, skip: int = 0, count: int | None = None) -> None:
    t0 = time.time()
    rows = []
    todo = PARTS[name][skip:] if count is None else PARTS[name][skip : skip + count]
    for i, kw in enumerate(todo):
        kwargs = dict(n_fluid=8, steps=steps, **kw)
        if name == "couette":
            kwargs["u_wall"] = 0.02        # sliding wall speed of the Couette benchmark
        row = run_case(**kwargs)
        rows.append(row)
        print(f"[{name} {i + 1}/{len(todo)}] {row['closure']:>7} {row['carrier']:>6} "
              f"db={row['delta_b']:.2f} dt={row['delta_t']:.2f} {row['arm']:>18} "
              f"l2={row['l2_rel']:.3e} max={row['max_rel']:.3e} steady={row['steady_at']}",
              flush=True)
    payload = {"anchor": ANCHOR, "part": name, "steps": steps,
               "wall_time_s": time.time() - t0, "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_{skip:02d}"
    (out_dir / f"results_{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"[{tag}] done in {payload['wall_time_s']:.1f}s -> results_{tag}.json")


def merge(out_dir: Path) -> int:
    files = sorted(out_dir.glob("results_*.json"))
    if not files:
        raise SystemExit(f"no results_*.json in {out_dir}")
    all_rows = {}
    for fp in files:
        d = json.loads(fp.read_text())
        all_rows.setdefault(d["part"], []).extend(d["rows"])
    (out_dir / "results.json").write_text(json.dumps({"anchor": ANCHOR, "sweeps": all_rows}, indent=1))

    for part, rows in all_rows.items():
        print(f"\n== {part} ==")
        print(f"{'closure':>8} {'carrier':>6} {'db':>5} {'dt':>5} {'arm':>19} "
              f"{'l2':>11} {'l2@1k':>11} {'max':>11} {'dM/step':>11}")
        for r in rows:
            l2k = "n/a" if r["l2_rel_1000"] is None else f"{r['l2_rel_1000']:.3e}"
            print(f"{r['closure']:>8} {r['carrier']:>6} {r['delta_b']:>5.2f} {r['delta_t']:>5.2f} "
                  f"{r['arm']:>19} {r['l2_rel']:>11.3e} {l2k:>11} {r['max_rel']:>11.3e} "
                  f"{r['dM_per_step']:>11.3e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(6)
    out_dir = Path(args.out)
    if args.merge:
        return merge(out_dir)
    if not args.part:
        raise SystemExit("need --part or --merge")
    run_part(args.part, out_dir, args.steps, skip=args.skip, count=args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())