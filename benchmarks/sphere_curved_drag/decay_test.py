#!/usr/bin/env python3
"""Wall-consistency control: staircase vs BFL curved wall, identical box.

Both walls are imposed on the *same* periodic box (``L = 24``, ``R = 6``,
``nu = 0.6``) holding the *same* initial uniform stream ``u0 = 0.05`` with no
body force.  The mean fluid momentum then decays at a rate set by the wall's
momentum removal; if the BFL wall removes momentum like the certified
staircase wall, the two decay curves coincide.

This is a *relative* control (wall vs wall), not an absolute drag check: the
unforced spin-down of a periodic cell is not a steady state, so no drag
coefficient is claimed from it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19  # noqa: E402
from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask  # noqa: E402
from tensorlbm.d3q19 import C, W  # noqa: E402
from tensorlbm.interpolated_bc import compute_q_sphere  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

L, R, NU, U0, STEPS = 24, 6.0, 0.6, 0.05, 300
TAU = 0.5 + 3.0 * NU


def run(scheme: str, threads: int = 8) -> dict:
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    cvec = C.to(dev).float().view(19, 3, 1, 1, 1)
    wvec = W.to(dev).float().view(19, 1, 1, 1)
    rr = torch.arange(L, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(rr, rr, rr, indexing="ij")
    c = 0.5 * L
    solid_sdf = ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) <= R * R
    solid_vox = sphere_mask(L, L, L, c, c, c, R, dev)
    mask, q = compute_q_sphere(L, L, L, c, c, c, R, dev)

    solid = solid_sdf if scheme == "bfl" else solid_vox
    fluid = ~solid
    v_fluid = float(fluid.sum())

    rho = torch.ones((L, L, L))
    ux = torch.full((L, L, L), U0)
    z = torch.zeros_like(rho)
    cu = cvec[:, 0] * ux
    f = wvec * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * ux * ux)
    f_rest = wvec.expand(19, L, L, L).contiguous()

    hist = []
    for i in range(1, STEPS + 1):
        rho = f.sum(dim=0)
        ir = 1.0 / rho
        ux = (f * cvec[:, 0]).sum(dim=0) * ir
        uy = (f * cvec[:, 1]).sum(dim=0) * ir
        uz = (f * cvec[:, 2]).sum(dim=0) * ir
        cuv = cvec[:, 0] * ux + cvec[:, 1] * uy + cvec[:, 2] * uz
        feq = wvec * rho * (1 + 3 * cuv + 4.5 * cuv * cuv - 1.5 * (ux * ux + uy * uy + uz * uz))
        fn = f - (f - feq) / TAU
        fn[:, solid] = f_rest[:, solid]
        if scheme == "bfl":
            fp = fn
            fs = stream3d(fp)
            f, force = bouzidi_bounce_back_d3q19(fs, fp, mask, q, return_force=True)
        else:
            f = stream3d(bounce_back_cells_3d(fn, solid_vox))
            force = (torch.tensor(0.0),) * 3
        if i % 25 == 0:
            hist.append({"step": i, "u_mean": float(ux[fluid].mean()), "F_bfl": float(force[0])})

    m0 = hist[0]["u_mean"]
    m1 = hist[-1]["u_mean"]
    rate = math.log(m0 / m1) / (hist[-1]["step"] - hist[0]["step"])
    return {
        "scheme": scheme,
        "L": L,
        "R": R,
        "nu": NU,
        "tau": TAU,
        "u0": U0,
        "steps": STEPS,
        "v_fluid": v_fluid,
        "n_solid": int(solid.sum()),
        "hist": hist,
        "u_end": m1,
        "decay_rate_per_step": rate,
        "F_equiv_initial": rate * (L**3) * U0,
        "cd_equiv_initial": 2 * rate * (L**3) * U0 / (U0 * U0 * math.pi * R * R),
    }


def main() -> int:
    out = {"protocol": "identical periodic box, same IC, no body force; wall vs wall",
           "cases": [run("bfl"), run("stair")]}
    a, b = out["cases"]
    out["rel_diff_decay_pct"] = 100.0 * (a["decay_rate_per_step"] - b["decay_rate_per_step"]) / b[
        "decay_rate_per_step"
    ]
    p = Path(__file__).resolve().parent / "out" / "decay_test.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"decay rate BFL {a['decay_rate_per_step']:.6e} vs staircase "
          f"{b['decay_rate_per_step']:.6e} -> {out['rel_diff_decay_pct']:+.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())