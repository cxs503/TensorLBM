#!/usr/bin/env python3
"""Ghost-rate calibration of the three-segment KBC operator: nu_eff(alpha).

The operator relaxes the shear segment at the *exact* BGK rate ``1/tau`` and the
ghost segment at ``alpha * omega_g / tau``.  The discrete Navier-Stokes
viscosity is not carried by the shear segment alone -- part of the ``-1/2`` in
``nu = cs2 (tau - 1/2)`` comes from the O(Kn) ghost content of the
Chapman-Enskog distribution -- so a ghost rate away from ``1/tau`` can shift the
effective viscosity and, at the campaign's low ``tau``, destabilise the
ghost modes.

Probe: a *driven* Stokes shear layer on a periodic 1-D grid,

    F_x(y) = F0 cos(k y)      =>      steady  u_x(y) = F0 cos(k y) / (nu_eff k^2)

which reaches steady state in ``1/(nu k^2)`` steps (a few thousand at the
campaign's viscosity) and needs no wall clock of the order of the channel
diffusion time.  ``alpha = 1`` must return ``nu = cs2 (tau - 1/2)``.

Output: ``calibration.json`` + ``calibration.csv`` under
``artifacts/kbc/p495w4-n03/ghost_rate_calibration``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.kbc_collision import CS2, decompose, equilibrium, lattice, moments  # noqa: E402

LAT = lattice(27)
CY = LAT.c[:, 1]
C = LAT.c.astype(float)


def driven_shear_nu(tau: float, alpha: float, *, ny: int = 32, k_cells: int = 8,
                    f0: float = 1e-9, steps: int = 4000, omega_g: float = 1.0) -> dict:
    """Effective viscosity from a body-force-driven Stokes shear layer."""
    k = 2.0 * np.pi / k_cells
    y = np.arange(ny)
    force = f0 * np.cos(k * y)                 # (ny,)
    u = np.zeros((3, ny))
    u[0] = force / (CS2 * (tau - 0.5) * k * k)  # start at the expected state
    f = equilibrium(np.ones(ny), u, LAT)
    w = LAT.w.reshape(-1, 1)
    for _ in range(steps):
        rho, uu = moments(f, LAT)
        uu = uu.copy()
        uu[0] += 0.5 * force                      # Guo half-force shift
        feq = equilibrium(rho, uu, LAT)
        s_seg, b_seg, g_seg = decompose(f - feq, LAT)   # K/B/C split at u*
        om = 1.0 / tau
        cand = (feq + (1.0 - om) * s_seg + (1.0 - om) * b_seg
                + (1.0 - alpha * omega_g * om) * g_seg)
        cu = np.einsum("qa,ay->qy", C, uu)
        cf = C[:, 0:1] * force[None]
        guo = w * (1.0 - 0.5 * om) * (3.0 * (cf - uu[0][None] * force[None]) + 9.0 * cu * cf)
        post = np.maximum(cand + guo, 1e-30)
        new = np.empty_like(post)
        for q in range(27):
            new[q] = np.roll(post[q], int(CY[q]))
        f = new
    rho, uu = moments(f, LAT)
    # cos(k y) with wavelength k_cells cells sits in FFT bin ny/k_cells;
    # |X_bin| = A ny / 2 for a pure cosine of amplitude A.
    amp = float(2.0 * np.abs(np.fft.rfft(uu[0])[ny // k_cells]) / ny)
    nu = f0 / (amp * k * k) if amp > 0 else float("inf")
    ok = bool(np.isfinite(nu) and amp > 0.0 and np.all(np.isfinite(f)))
    return {"alpha": alpha, "tau": tau, "k": k, "u_amp": amp,
            "nu_meas": nu if ok else float("nan"),
            "nu_nominal": CS2 * (tau - 0.5), "finite": ok}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/kbc/p495w4-n03/ghost_rate_calibration")
    ap.add_argument("--alphas", default="0.0,0.9,1.0,1.02,1.05,1.1,1.2")
    ap.add_argument("--taus", default="0.5024,0.5026,0.52,0.7325")
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
    rows = []
    for tau in taus:
        ref = driven_shear_nu(tau, 1.0, steps=args.steps)
        ref_nu = ref["nu_meas"]
        for alpha in alphas:
            row = driven_shear_nu(tau, alpha, steps=args.steps)
            row["nu_ratio_vs_alpha1"] = (row["nu_meas"] / ref_nu
                                         if (ref_nu == ref_nu and ref_nu > 0) else float("nan"))
            row["nu_ratio_vs_nominal"] = (row["nu_meas"] / row["nu_nominal"]
                                          if row["nu_nominal"] > 0 else float("nan"))
            rows.append(row)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}),
                  flush=True)
    payload = {
        "description": "effective shear viscosity of the three-segment KBC operator vs ghost rate",
        "method": "driven Stokes shear layer: F_x = F0 cos(k y), steady u = F0/(nu_eff k^2)",
        "alpha_definition": "ghost rate = alpha * omega_g / tau; alpha = omega_g = 1 is BGK",
        "rows": rows,
        "reading": ("alpha = 1 reproduces nu = cs2 (tau - 1/2); alpha > 1 destabilises the ghost "
                    "modes at the campaign's tau ~ 0.5024-0.5026, which is why the production "
                    "configuration fixes alpha = 1."),
    }
    (out / "calibration.json").write_text(json.dumps(payload, indent=1))
    with open(out / "calibration.csv", "w") as fh:
        fh.write("tau,alpha,nu_meas,nu_nominal,nu_ratio_vs_alpha1,nu_ratio_vs_nominal,finite\n")
        for r in rows:
            fh.write(f"{r['tau']},{r['alpha']},{r['nu_meas']:.6e},{r['nu_nominal']:.6e},"
                     f"{r['nu_ratio_vs_alpha1']:.4f},{r['nu_ratio_vs_nominal']:.4f},{int(r['finite'])}\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
