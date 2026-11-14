"""V4 verifier: wall position and solid-slot bookkeeping of the alpha-scheme operator.

Anchor ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``.  Adversarial follow-up of
``verify_v3_moving_wall.py`` (mission m20669) and of the numpy verifier ``verify_v2.py``.

Three things the earlier bundles leave untested
-----------------------------------------------
Q1 **Container independence.**  ``verify_v3_moving_wall.py`` streams periodically in
   y with only two solid rows and never collides the solid; ``verify_v2.py`` guarded
   the wrap (``f_st[:, 0, :] = WI``), V3 did not.  If the residual depends on the
   container, it cannot be attributed to the boundary closure.  Tested here by
   running the *same* physics in a 2-row periodic container and in an absorbing-solid
   container (solid populations destroyed after every streaming step).

Q2 **What the residual is made of.**  Reported ``l2`` mixes two different errors:
   a wrong *wall position* (geometry) and a wrong *amplitude* (a uniform velocity
   scale, i.e. a mean-density offset).  They are separated here by a least-squares
   parabola fit of the numerical profile:

       u(y) = A y (H_eff - y),   A = -a2,  H_eff = y_high - y_low

   ``w_bot = y_fluid[0] - y_low``  -- measured distance from the first fluid node to
   the no-slip plane; exact placement means ``w_bot = delta_b``.
   ``A / A_ref``                   -- ``A_ref = g / (2 nu)`` for the analytic
   Poiseuille solution; exact momentum balance means ``A / A_ref = 1``.

Q3 **Solid-slot initialization contract.**  The operator writes only the *reflected*
   populations into the solid cell (the "solid-side write" contract); everything else
   inside the solid is dead storage.  ``verify_v3_moving_wall.py`` initialises the
   solid to the rest equilibrium ``W``; this driver can instead seed the reflection
   slots by applying the operator once at ``t = 0`` (``init="reflect"``).  The two
   conventions differ by a *one-time* mass injection at the first step, which is
   redistributed as a uniform density offset and then frozen (nothing removes mass
   from a periodic-in-x channel).  ``init`` sweeps therefore measure how much of the
   reported velocity error is bookkeeping rather than physics.

Container / initialization modes
--------------------------------
``mode="iso"``  solid populations destroyed after streaming (absorbing wall);
``mode="per"``  bit-faithful V3 container: periodic streaming, solid untouched.
``init="reflect"`` (default) zero solid + one operator application before the loop;
``init="rest"``  solid filled with the lattice rest equilibrium ``W`` (= V3 driver);
``init="zero"``  solid zeroed and *not* seeded -- reproduces the first-step deficit.

Usage
-----
    PYTHONPATH=src python3 verify_v4.py --part sweep     # Q2 (wall position vs delta)
    PYTHONPATH=src python3 verify_v4.py --part init      # Q3 (bookkeeping)
    PYTHONPATH=src python3 verify_v4.py --part wrap      # Q1 (container)
    PYTHONPATH=src python3 verify_v4.py --merge          # tables
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ANCHOR = "XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME"
sys.path.insert(0, str(REPO / "src"))

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
OPP = OPP9.to(torch.int64)

ALPHAS = {
    "halfway": lambda q: torch.full_like(q, 2.0),
    "bfl": alpha_bfl,
    "cli": alpha_cli,
}


class Channel:
    """Two straight walls at sub-grid distances below/above the fluid rows."""

    def __init__(self, n_fluid: int, delta_b: float, delta_t: float,
                 nx: int = 2, n_wall: int = 3):
        self.n_fluid, self.db, self.dt, self.nx, self.n_wall = n_fluid, delta_b, delta_t, nx, n_wall
        ny = n_fluid + 2 * n_wall
        y = torch.arange(ny, dtype=DT) - n_wall + delta_b   # y = 0 is the lower wall
        self.height = (n_fluid - 1) + delta_b + delta_t
        self.sdf = torch.minimum(y, self.height - y)
        fluid1d = torch.zeros(ny, dtype=torch.bool)
        fluid1d[n_wall : n_wall + n_fluid] = True
        self.fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
        self.sdf = self.sdf.view(ny, 1).expand(ny, nx).contiguous()
        self.solid = ~self.fluid
        self.ny, self.nx = ny, nx
        self.y_fluid = y[n_wall : n_wall + n_fluid]
        self.rows = (n_wall, n_wall + n_fluid)


# --------------------------------------------------------------------------- #
# LBM primitives (rewritten for this driver; float64)
# --------------------------------------------------------------------------- #
def equilibrium(rho, ux, uy):
    cu = CX * ux + CY * uy
    usq = ux * ux + uy * uy
    return WI * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def moments(f):
    return f.sum(0), (CX * f).sum(0), (CY * f).sum(0)


def collide(f, rho, ux, uy, carrier: str, nu: float, lam_m: float | None):
    feq = equilibrium(rho, ux, uy)
    if carrier == "bgk":
        return f - (1.0 / (0.5 + 3.0 * nu)) * (f - feq)
    sp, sm = 1.0 / (0.5 + 3.0 * nu), 1.0 / (0.5 + lam_m)
    fp, fm = 0.5 * (f + f[OPP]), 0.5 * (f - f[OPP])
    ep, em = 0.5 * (feq + feq[OPP]), 0.5 * (feq - feq[OPP])
    return f - sp * (fp - ep) - sm * (fm - em)


def stream(f):
    out = torch.empty_like(f)
    for i in range(Q):
        out[i] = torch.roll(f[i], shifts=(int(CI[i, 1]), int(CI[i, 0])), dims=(0, 1))
    return out


def reference_profile(ch: Channel, g: float, nu: float) -> torch.Tensor:
    y = ch.y_fluid
    return g / (2.0 * nu) * y * (ch.height - y)


def report_profile(f: torch.Tensor, g: float, ch: Channel) -> torch.Tensor:
    rho, mx, _ = moments(f)
    return (mx / rho)[ch.rows[0] : ch.rows[1], 0] - 0.5 * g


def parabola_fit(y: np.ndarray, u: np.ndarray) -> tuple[float, float, float, float]:
    """Return (A, y_low, y_high, rel_residual) of u = A (y - y_low)(y_high - y)."""
    A = np.vstack([np.ones_like(y), y, y * y]).T
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    a0, a1, a2 = (float(c) for c in coef)
    fit = A @ coef
    res = float(np.linalg.norm(fit - u) / max(np.linalg.norm(u), 1e-300))
    roots = np.roots([a2, a1, a0])
    real = np.sort(roots[np.isreal(roots)].real)
    if real.size < 2 or a2 == 0.0:
        return float("nan"), float("nan"), float("nan"), res
    return -a2, float(real[0]), float(real[-1]), res


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def run_case(*, n_fluid: int = 8, delta_b: float = 0.25, delta_t: float | None = None,
             closure: str = "bfl", carrier: str = "bgk", mode: str = "iso",
             init: str = "reflect", n_wall: int = 3, nx: int = 2,
             nu: float = 0.1, u_max: float = 0.02, steps: int = 1500) -> dict:
    delta_t = delta_b if delta_t is None else delta_t
    ch = Channel(n_fluid, delta_b, delta_t, nx=nx, n_wall=n_wall)
    g = 8.0 * nu * u_max / ch.height**2
    lam_m = None
    if carrier == "trt16":                     # classic magic product Lambda+ Lambda- = 3/16
        lam_m = (3.0 / 16.0) / (3.0 * nu)
    elif carrier == "trtd2":                   # PREREG recipe Lambda+ Lambda- = delta^2 / 2
        lam_m = (0.5 * delta_b**2) / (3.0 * nu)
    elif carrier != "bgk":
        raise ValueError(carrier)

    af = ALPHAS[closure]
    u_ref = reference_profile(ch, g, nu)
    A_ref = g / (2.0 * nu)

    u0 = torch.zeros((ch.ny, ch.nx), dtype=DT)
    u0[ch.rows[0] : ch.rows[1], :] = (u_ref + 0.5 * g).unsqueeze(1)
    f = equilibrium(torch.ones_like(u0), u0, torch.zeros_like(u0))
    f = torch.where(ch.fluid.unsqueeze(0), f, torch.zeros_like(f))
    src = torch.where(ch.fluid.unsqueeze(0), 3.0 * WI * CX * g, torch.zeros_like(WI * CX))

    def boundary(f_new, f_pre):
        return interpolated_bounce_back_2d(f_new, f_pre, ch.sdf, ch.solid, alpha=af)

    if init == "rest":
        f = torch.where(ch.fluid.unsqueeze(0), f, WI.expand_as(f).contiguous())
    elif init == "reflect":
        # seed the reflection slots exactly as the operator will use them, dead
        # storage stays zero -> no spurious mass injection on the first step
        f = boundary(stream(f), f)
    elif init != "zero":
        raise ValueError(init)

    m0 = float(f.sum())
    t0 = time.time()
    for _ in range(steps):
        rho, mx, my = moments(f)
        f_post = collide(f, rho, mx / rho, my / rho, carrier, nu, lam_m)
        f_post = torch.where(ch.fluid.unsqueeze(0), f_post + src, f)  # solid keeps reflections
        f_new = stream(f_post)
        if mode == "iso":
            f_new = torch.where(ch.fluid.unsqueeze(0), f_new, torch.zeros_like(f_new))
        elif mode != "per":
            raise ValueError(mode)
        f = boundary(f_new, f_post)

    prof = report_profile(f, g, ch)
    u_ref_t = u_ref
    y = ch.y_fluid.numpy()
    A_fit, y_low, y_high, fit_res = parabola_fit(y, prof.numpy())
    rho_all, _, _ = moments(f)
    rho_fluid = rho_all[ch.rows[0] : ch.rows[1], :]
    scale = float(torch.max(torch.abs(u_ref_t)))
    amp = float(A_fit / A_ref) if np.isfinite(A_fit) else float("nan")
    # shape-only residual: reference renormalised to the fitted amplitude
    l2_shape = float(torch.linalg.norm(prof - amp * u_ref_t) / torch.linalg.norm(u_ref_t))
    return {
        "n_fluid": n_fluid, "delta_b": delta_b, "delta_t": delta_t, "H": ch.height,
        "closure": closure, "carrier": carrier, "mode": mode, "init": init,
        "n_wall": n_wall, "lattice": carrier, "lam_m": lam_m, "nu": nu, "g": g,
        "steps": steps,
        "l2_rel": float(torch.linalg.norm(prof - u_ref_t) / torch.linalg.norm(u_ref_t)),
        "l2_shape": l2_shape,
        "max_rel": float(torch.max(torch.abs(prof - u_ref_t))) / scale,
        "A_over_ref": amp,
        "w_bot": float(y[0] - y_low), "w_top": float(y_high - y[-1]),
        "H_eff": float(y_high - y_low),
        "w_bot_err": float(y[0] - y_low - delta_b),
        "w_top_err": float(y_high - y[-1] - delta_t),
        "fit_res": fit_res,
        "rho_fluid_mean": float(rho_fluid.mean()),
        "dM_total_rel": (float(f.sum()) - m0) / m0,
        "wall_time_s": time.time() - t0,
        "y": [float(v) for v in y], "u": [float(v) for v in prof],
        "u_ref": [float(v) for v in u_ref_t],
    }


# --------------------------------------------------------------------------- #
# case lists
# --------------------------------------------------------------------------- #
def sweep_cases() -> list[dict]:
    """Q2: wall position vs delta on the production operator, clean container."""
    out = []
    for d in (0.1, 0.25, 0.5, 0.75, 0.9):
        for clo in ("halfway", "bfl", "cli"):
            out.append(dict(delta_b=d, closure=clo, carrier="bgk"))
    for car in ("trtd2", "trt16"):
        for d in (0.25, 0.5, 0.75):
            out.append(dict(delta_b=d, closure="bfl", carrier=car))
    return out


def init_cases() -> list[dict]:
    """Q3: how much of the velocity error is solid-slot bookkeeping?"""
    return [dict(delta_b=d, closure=clo, carrier="bgk", init=i)
            for d in (0.25, 0.5) for clo in ("bfl", "halfway")
            for i in ("reflect", "rest", "zero")]


def wrap_cases() -> list[dict]:
    """Q1: same physics, different container."""
    return [dict(delta_b=0.25, closure=clo, carrier="bgk", mode=m,
                 n_wall=2 if m == "per" else 3)
            for clo in ("bfl", "cli") for m in ("per", "iso")]


PARTS = {"sweep": sweep_cases, "init": init_cases, "wrap": wrap_cases}


def run_part(name: str, out_dir: Path, steps: int) -> None:
    t0 = time.time()
    todo = PARTS[name]()
    rows = []
    for i, kw in enumerate(todo):
        row = run_case(steps=steps, **kw)
        rows.append(row)
        print(f"[{name} {i + 1}/{len(todo)}] {row['closure']:>7} {row['carrier']:>6} "
              f"{row['mode']}/{row['init']:>7} db={row['delta_b']:.2f} "
              f"l2={row['l2_rel']:.3e} shape={row['l2_shape']:.2e} "
              f"amp={row['A_over_ref']:.5f} w_err=({row['w_bot_err']:+.4f},{row['w_top_err']:+.4f}) "
              f"rho={row['rho_fluid_mean']:.6f} {row['wall_time_s']:.1f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"anchor": ANCHOR, "part": name, "steps": steps,
               "wall_time_s": time.time() - t0, "rows": rows}
    (out_dir / f"results_{name}.json").write_text(json.dumps(payload, indent=1))
    print(f"[{name}] {len(rows)} arms in {payload['wall_time_s']:.1f}s")
    (out_dir / "summary_v4.md").write_text(render_summary(load(out_dir)))


def load(out_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for p in sorted(out_dir.glob("results_*.json")):
        out[p.stem.replace("results_", "")] = json.loads(p.read_text())["rows"]
    return out


def render_summary(data) -> str:
    L = ["# V4 measurement summary", ""]
    if "sweep" in data:
        L += ["## Q2 wall position vs delta (init=reflect, iso container, N=8)", "",
              "| δ | closure | carrier | l2 % | shape l2 % | A/A_ref | w_bot | w_bot−δ | w_top−δ | ρ̄ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in data["sweep"]:
            L.append(f"| {r['delta_b']:.2f} | {r['closure']} | {r['carrier']} | "
                     f"{100 * r['l2_rel']:.4f} | {100 * r['l2_shape']:.4f} | "
                     f"{r['A_over_ref']:.5f} | {r['w_bot']:.4f} | {r['w_bot_err']:+.4f} | "
                     f"{r['w_top_err']:+.4f} | {r['rho_fluid_mean']:.6f} |")
        L.append("")
    if "init" in data:
        L += ["## Q3 solid-slot initialization (δ, closure on BGK)", "",
              "| δ | closure | init | l2 % | A/A_ref | ρ̄ | w_bot−δ |", "|---|---|---|---|---|---|---|"]
        for r in data["init"]:
            L.append(f"| {r['delta_b']:.2f} | {r['closure']} | {r['init']} | "
                     f"{100 * r['l2_rel']:.4f} | {r['A_over_ref']:.5f} | "
                     f"{r['rho_fluid_mean']:.6f} | {r['w_bot_err']:+.4f} |")
        L.append("")
    if "wrap" in data:
        L += ["## Q1 container sensitivity (δ=0.25)", "",
              "| closure | mode | l2 % | A/A_ref | w_bot−δ |", "|---|---|---|---|---|"]
        for r in data["wrap"]:
            L.append(f"| {r['closure']} | {r['mode']} | {100 * r['l2_rel']:.4f} | "
                     f"{r['A_over_ref']:.5f} | {r['w_bot_err']:+.4f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts_v4")
    args = ap.parse_args(argv)

    if args.part:
        run_part(args.part, args.out, args.steps)
    if args.merge or not args.part:
        text = render_summary(load(args.out))
        (args.out / "summary_v4.md").write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())