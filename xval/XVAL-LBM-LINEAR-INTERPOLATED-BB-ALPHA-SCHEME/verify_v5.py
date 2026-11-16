"""V5 verifier: close the OSS-exactness gap — transplant pylbm's *actual* recipe.

Anchor ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``.  Follow-up of
``verify_v4.py`` (mission m22795), which recorded an honest FAIL:

    "OSS 严格精确性 l2<1e-10 未复现（最好臂 7.5e-04）"

and attributed the residual to the closure.  This driver tests the alternative
hypothesis that the residual is a **contract gap**, not a closure defect.

What was read from the OSS test (``~/oss-cache/pylbm/tests/test_linear_bounce_back.py``,
``poiseuille``):

* carrier: d2Q9 **TRT** with ``Lambda+ = 3 nu`` and the scheme magic product
  ``Lambda+ Lambda- = delta^2 / 2`` (i.e. ``Lambda- = delta^2 / (6 nu)``);
* force: post-collision, isotropic weights, ``S_q = 3 w_q c_qx F``;
* reported velocity: ``sum f c / rho - F/2`` (the population set stores ``F/2``);
* wall Dirichlet value: ``value`` callback sets ``m[qx] = lam_m * F``, i.e. the
  velocity used to build ``e^-_q(x_w)`` is **``u_w = Lambda- F``**, not ``0`` and
  not ``F/2``.  This is the ``-alpha e^-_q(x_w)`` term of
  :class:`pylbm.boundary.LinearBounceBack`;
* assertion: ``atol = 1e-12 * exact.max()`` over the fluid nodes.

``verify_v4.py`` never passed ``wall_velocity`` (V3 added the term, the V4 loop
does not use it), so the exactness claim was being tested against a *different*
boundary condition.  Arms below isolate the two candidate ingredients
(Dirichlet wall value, post-collision correction ``K_hat``) on the same
container, TRT carrier and convergence criterion.

Parts
-----
``--part wall``    Dirichlet-value ablation (zero / F/2 / Lambda- F) x closure x correction
``--part family``  full closure x correction x delta x nu grid at the pylbm wall value
``--part carrier`` BGK control at the pylbm wall value (expect parametrization to break)
``--merge``        tables -> ``artifacts_v5/``

Usage
-----
    PYTHONPATH=src python3 verify_v5.py --part wall
    PYTHONPATH=src python3 verify_v5.py --part family
    PYTHONPATH=src python3 verify_v5.py --part carrier
    PYTHONPATH=src python3 verify_v5.py --merge
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    alpha_bfl,
    alpha_cli,
    alpha_yli,
    interpolated_bounce_back_2d,
)

from verify_v4 import (  # noqa: E402
    CX,
    CY,
    DT,
    WI,
    Channel,
    collide,
    equilibrium,
    moments,
    reference_profile,
    stream,
)

OUT = HERE / "artifacts_v5"
ALPHAS = {
    "halfway": lambda q: torch.full_like(q, 2.0),   # production half-way operator
    "bfl": alpha_bfl,
    "cli": alpha_cli,
    "yli": alpha_yli,
}
# pylbm asserts atol = 1e-12 * max(exact); same tolerance here
Pylbm_ATOL = 1e-12


def wall_value(kind: str, g: float, lam_m: float | None) -> tuple[float, float]:
    """Dirichlet wall velocity of the cut links (x component, y component)."""
    if kind == "zero":
        return (0.0, 0.0)
    if kind == "half":                      # naive "no-slip stored momentum"
        return (0.5 * g, 0.0)
    if kind == "laminus":                   # pylbm ``value``: m[qx] = lam_m * F
        if lam_m is None:
            raise ValueError("laminus needs the TRT anti-symmetric rate")
        return (lam_m * g, 0.0)
    raise ValueError(kind)


def run_case(*, delta: float = 0.375, nu: float = 0.3, closure: str = "cli",
             correction: bool = True, wall: str = "laminus", carrier: str = "trt",
             n_fluid: int = 8, nx: int = 2, n_wall: int = 3, u_max: float = 0.05,
             max_steps: int = 120000, tol: float = 1e-14, check_every: int = 200,
             init: str = "reflect") -> dict:
    """One arm: force-driven plane Poiseuille, pylbm convention, march to steady state."""
    ch = Channel(n_fluid, delta, delta, nx=nx, n_wall=n_wall)
    g = 8.0 * nu * u_max / ch.height**2

    lam_m = None
    if carrier == "trt":
        lam_m = 0.5 * delta**2 / (3.0 * nu)      # Lambda+ Lambda- = delta^2 / 2
    elif carrier != "bgk":
        raise ValueError(carrier)

    uw = wall_value(wall, g, lam_m)
    wall_velocity = torch.tensor(uw, dtype=DT).expand(ch.ny, ch.nx, 2).contiguous()

    af = ALPHAS[closure]
    u_ref = reference_profile(ch, g, nu)
    rows = ch.rows

    u0 = torch.zeros((ch.ny, ch.nx), dtype=DT)
    u0[rows[0]:rows[1], :] = (u_ref + 0.5 * g).unsqueeze(1)
    f = equilibrium(torch.ones_like(u0), u0, torch.zeros_like(u0))
    f = torch.where(ch.fluid.unsqueeze(0), f, torch.zeros_like(f))
    src = torch.where(ch.fluid.unsqueeze(0), 3.0 * WI * CX * g, torch.zeros_like(WI * CX))

    def boundary(f_new, f_pre, f_old):
        kw = dict(alpha=af, wall_velocity=wall_velocity, wall_density=1.0)
        if correction:
            kw.update(post_collision_correction=True, f_old=f_old)
        return interpolated_bounce_back_2d(f_new, f_pre, ch.sdf, ch.solid, **kw)

    if init == "reflect":
        f = boundary(stream(f), f, f)
    elif init != "zero":
        raise ValueError(init)

    def one_step(f_cur):
        rho, mx, my = moments(f_cur)
        f_post = collide(f_cur, rho, mx / rho, my / rho, carrier, nu, lam_m)
        f_post = torch.where(ch.fluid.unsqueeze(0), f_post + src, f_cur)
        return boundary(stream(f_post), f_post, f_cur)

    t0 = time.time()
    steps, converged = 0, False
    prev = None
    while steps < max_steps:
        f = one_step(f)
        steps += 1
        if steps % check_every == 0:
            rho, mx, _ = moments(f)
            u = (mx / rho)[rows[0]:rows[1], 0] - 0.5 * g
            if prev is not None:
                d = float(torch.max(torch.abs(u - prev)))
                if d <= tol * float(torch.max(torch.abs(u))):
                    converged = True
                    break
            prev = u

    rho, mx, _ = moments(f)
    u = (mx / rho)[rows[0]:rows[1], 0] - 0.5 * g
    u_ref_t = u_ref
    scale = float(torch.max(torch.abs(u_ref_t)))
    l2 = float(torch.linalg.norm(u - u_ref_t) / torch.linalg.norm(u_ref_t))
    max_rel = float(torch.max(torch.abs(u - u_ref_t))) / scale
    # fixed-point drift: one further step and re-measure
    f2 = one_step(f)
    rho2, mx2, _ = moments(f2)
    u2 = (mx2 / rho2)[rows[0]:rows[1], 0] - 0.5 * g
    drift = float(torch.max(torch.abs(u2 - u))) / scale

    return {
        "delta": delta, "nu": nu, "H": ch.height, "closure": closure,
        "correction": bool(correction), "wall": wall, "carrier": carrier,
        "lam_plus": 3.0 * nu, "lam_minus": lam_m, "u_w": uw[0], "g": g,
        "u_max_ref": scale, "n_fluid": n_fluid, "steps": steps,
        "converged": converged, "l2_rel": l2, "max_rel": max_rel,
        "fixed_point_drift": drift,
        "passes_pylbm_atol": bool(max_rel <= Pylbm_ATOL),
        "passes_owner_15pct": bool(l2 < 0.15),
        "rho_fluid_mean": float(rho[rows[0]:rows[1], :].mean()),
        "wall_time_s": time.time() - t0,
        "y": [float(v) for v in ch.y_fluid],
        "u": [float(v) for v in u], "u_ref": [float(v) for v in u_ref_t],
    }


# --------------------------------------------------------------------------- #
# case lists
# --------------------------------------------------------------------------- #
def wall_cases() -> list[dict]:
    """Dirichlet-value ablation on the *same* closure/carrier/grid."""
    out = []
    for d in (0.375, 0.625):
        for clo in ("bfl", "cli"):
            for corr in (True, False):
                for w in ("zero", "half", "laminus"):
                    out.append(dict(delta=d, closure=clo, correction=corr, wall=w))
    return out


def family_cases() -> list[dict]:
    """pylbm's own sweep shape: closure x correction x delta x nu at u_w = Lambda- F."""
    out = []
    for d in (0.125, 0.375, 0.625, 0.875):
        for nu in (0.05, 0.3):
            for clo in ("bfl", "cli", "yli"):
                for corr in (True, False):
                    out.append(dict(delta=d, nu=nu, closure=clo, correction=corr,
                                    wall="laminus"))
    return out


def carrier_cases() -> list[dict]:
    """Controls: BGK carrier and the delta=1/2 collapse at the pylbm wall value."""
    out = [dict(delta=d, closure=clo, correction=True, wall="laminus", carrier="bgk",
                nu=0.3)
           for d in (0.375, 0.625) for clo in ("bfl", "cli")]
    out += [dict(delta=0.5, closure=clo, correction=True, wall="laminus")
            for clo in ("bfl", "cli", "yli")]
    out += [dict(delta=0.5, closure="bfl", correction=False, wall="half")]
    return out


PARTS = {"wall": wall_cases, "family": family_cases, "carrier": carrier_cases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS) + ["merge"], required=True)
    ap.add_argument("--key", default=None, help="only this case key (debug)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.part == "merge":
        merge()
        return

    cases = PARTS[args.part]()
    res = []
    for i, c in enumerate(cases):
        t0 = time.time()
        r = run_case(**c)
        res.append(r)
        print(f"[{i + 1}/{len(cases)}] {c} -> l2={r['l2_rel']:.3e} "
              f"max={r['max_rel']:.3e} steps={r['steps']} "
              f"conv={r['converged']} ({time.time() - t0:.1f}s)", flush=True)
    (OUT / f"results_{args.part}.json").write_text(json.dumps(res, indent=1))


def merge() -> None:
    rows: list[dict] = []
    for p in PARTS:
        fp = OUT / f"results_{p}.json"
        if fp.exists():
            rows += json.loads(fp.read_text())
    if not rows:
        print("no results")
        return
    best = min(rows, key=lambda r: r["l2_rel"])
    lines = ["| closure | corr | wall u_w | carrier | delta | nu | l2_rel | max_rel | pylbm 1e-12 | owner 15% | steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['closure']} | {'on' if r['correction'] else 'off'} | {r['wall']} "
            f"| {r['carrier']} | {r['delta']} | {r['nu']} | {r['l2_rel']:.3e} "
            f"| {r['max_rel']:.3e} | {'PASS' if r['passes_pylbm_atol'] else 'fail'} "
            f"| {'PASS' if r['passes_owner_15pct'] else 'FAIL'} | {r['steps']} |")
    (OUT / "summary_v5.md").write_text(
        f"# V5 summary ({len(rows)} arms)\n\nbest l2_rel = {best['l2_rel']:.3e} "
        f"({best['closure']}, corr={best['correction']}, wall={best['wall']}, "
        f"delta={best['delta']}, nu={best['nu']}, carrier={best['carrier']})\n\n"
        + "\n".join(lines) + "\n")
    print(f"wrote {OUT / 'summary_v5.md'} ({len(rows)} arms, best {best['l2_rel']:.3e})")


if __name__ == "__main__":
    main()