#!/usr/bin/env python3
"""Run the *reference* implementation (ebinan92/Fingering_dynamics) as an XVAL arm.

The reference file is vendored verbatim at ``oss_src/validation_ebinan92.py``
(repo ``https://github.com/ebinan92/Fingering_dynamics``, commit ``150a5739``,
``lattice_boltzmann/validation.py``; the repository ships no LICENSE file --
this copy is included only as the cross-validation reference source).

Two *documented* patches are applied by this runner; no physics formula is
touched:

P1 (required to run at all): in the published file the order-parameter stencils
   ``getNabla_psix`` / ``getNabla_psiy`` / ``getNabla_psi2`` are defined at
   module level while ``Compute.getMu`` calls them as methods, so the file as
   committed raises ``AttributeError: 'Compute' object has no attribute
   'getNabla_psi2'``.  The runner binds them onto the class (no formula change).

P2 (required to have a contact line): the published ``main`` initialises the
   drop at the *centre* of the box (``create_circle``), so the drop touches no
   wall and no contact angle exists.  The runner re-instantiates the same
   scheme with the drop resting on the lower wall, i.e. the geometry of the
   xval case.

Everything else (equilibria, forcing, stencils, bounce-back, parameters) is the
reference's own code.  The contact angle is measured with the TensorLBM
estimator so that both arms are read by the same ruler.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE / "oss_src" / "validation_ebinan92.py"

sys.path.insert(0, str(HERE.parents[1] / "src"))
from tensorlbm.wetting.wall_order_parameter_2d import contact_angle_from_field  # noqa: E402


def load_reference() -> dict:
    """Exec the reference source (without its ``__main__`` block)."""
    text = SRC.read_text()
    text = text.split("if __name__ == '__main__':")[0]
    import matplotlib

    matplotlib.use("Agg")
    mod: dict = {"__name__": "oss_validation"}
    exec(compile(text, str(SRC), "exec"), mod)  # noqa: S102 - vendored reference
    cls = mod["Compute"]
    for name in (
        "getNabla_psix",
        "getNabla_psiy",
        "getNabla_psi2",
        "getMix_tau",
        "power_law",
        "updatePsi",
        "updateF",
        "updateG",
    ):
        if name in mod:  # only the helpers the published file left at module level
            setattr(cls, name, mod[name])  # patch P1
    return mod


def run_case(psi_wall: float, H: int, W: int, steps: int, radius: float, outdir: Path) -> dict:
    mod = load_reference()
    mod["H"] = H
    mod["W"] = W
    mod["psi_wall"] = float(psi_wall)
    mod["MAX_T"] = steps
    # patch P2: rest the drop on the lower wall instead of the box centre
    def create_circle(n: int, r: float):  # noqa: ANN001 - mirror of the reference signature
        # the reference returns a mask that its caller transposes (``.T``) before
        # slicing to ``(H, W)``; keep that contract and only move the centre down
        # to the lower wall (y = r + 1) instead of the box centre.
        xx, yy = np.ogrid[0:n, 0:H]
        return (xx - 0.5 * n) ** 2 + (yy - (r + 1.0)) ** 2 <= r**2

    mod["create_circle"] = create_circle

    t0 = time.time()
    cm = mod["Compute"]()
    for _ in range(steps):
        for j in range(9):
            cm.feq[j] = cm.getfeq(j)
            cm.geq[j] = cm.getgeq(j)
        cm.mix_tau = cm.getMix_tau()
        for j in range(9):
            cm.F[j] = cm.getLarge_F(j)
        cm.updateF()
        cm.updateG()
        f_behind = copy.deepcopy(cm.f)
        g_behind = copy.deepcopy(cm.g)
        mod["stream"](cm.f, cm.g)
        mod["halfway_bounceback"](f_behind, g_behind, cm.f, cm.g)
        cm.updateRho()
        cm.updatePsi()
        cm.updateMu()
        cm.updateU()
        cm.updateP()
    wall = time.time() - t0

    psi = torch.tensor(np.array(cm.psi), dtype=torch.float64)
    finite = bool(np.isfinite(np.array(cm.psi)).all())
    meas = {}
    if finite:
        try:
            meas = contact_angle_from_field(
                psi, psi_wall=float(psi_wall), interface_width=float(mod["xi"])
            )
        except Exception as exc:  # pragma: no cover - degenerate reference fields
            meas = {"error": str(exc)}
    result = {
        "arm": "oss_reference",
        "psi_wall": float(psi_wall),
        "H": H,
        "W": W,
        "steps": steps,
        "radius": radius,
        "finite": finite,
        "psi_min": float(psi.min()) if finite else float("nan"),
        "psi_max": float(psi.max()) if finite else float("nan"),
        "psi_mass": float(psi.sum()) if finite else float("nan"),
        "max_u": float(np.abs(np.array(cm.ux)).max() + np.abs(np.array(cm.uy)).max()) if finite else float("nan"),
        "wall_seconds": wall,
        "measurement": meas,
        "reference_constants": {
            "xi": float(mod["xi"]),
            "sigma": float(mod["sigma"]),
            "kappa": float(mod["kappa"]),
            "a": float(mod["a"]),
            "gamma": float(mod["gamma"]),
            "tau_g": float(mod["tau"]),
            "Pe": float(mod["Pe"]),
            "C_W": float(mod["C_W"]),
        },
        "patches": ["P1 stencil bindings", "P2 drop on lower wall"],
        "source": "ebinan92/Fingering_dynamics@150a5739 lattice_boltzmann/validation.py",
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    np.save(outdir / "psi.npy", np.array(cm.psi))
    return result


def main(argv: list[str]) -> int:
    psi_wall = float(argv[1]) if len(argv) > 1 else 0.0
    H = int(argv[2]) if len(argv) > 2 else 56
    W = int(argv[3]) if len(argv) > 3 else 80
    steps = int(argv[4]) if len(argv) > 4 else 2000
    radius = float(argv[5]) if len(argv) > 5 else 16.0
    outdir = HERE / "runs" / f"oss_psi{psi_wall:+.2f}"
    res = run_case(psi_wall, H, W, steps, radius, outdir)
    print(json.dumps({k: res[k] for k in ("finite", "psi_min", "psi_max", "wall_seconds")}))
    print(json.dumps(res["measurement"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))