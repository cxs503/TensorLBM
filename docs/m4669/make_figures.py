"""Figures for the m4669 CI-red fix (TensorLBM PR #669).

Two real, programmatically generated panels — nothing hand-copied:

(a) ``wss_from_fneq_3d`` returned field on a mid-plane, for a state that is
    equilibrium + a deterministic xy-shear perturbation and a spherical solid
    inclusion (the scenario used by the rewritten regression guard).  Solid
    cells must be masked to zero, fluid cells must carry |sigma_xy|.
(b) Verification of the same call path against the closed-form stress of the
    imposed perturbation: ``sigma_xy = -(1 - 1/(2 tau)) * A * sum_q c_x^2 c_y^2``
    with ``sum_q c_x^2 c_y^2 = 4`` for D3Q19 (the four in-plane diagonals).  Relative error vs tau.

Run: python docs/m4669/make_figures.py  (writes docs/m4669/fig_wss_regression.png)
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.boundaries3d import sphere_mask  # noqa: E402
from tensorlbm.d3q19 import W, C, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.wall_shear import wss_from_fneq_3d  # noqa: E402

OUT = pathlib.Path(__file__).with_name("fig_wss_regression.png")
SXXYY = sum(float(cx) ** 2 * float(cy) ** 2 for (cx, cy, cz) in C)  # = 4 for D3Q19


def sheared_state(shape: tuple[int, int, int], amp: float = 1e-4):
    rho = torch.ones(shape, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f_eq = equilibrium3d(rho, zero, zero, zero)
    cx_cy = (C[:, 0] * C[:, 1]).view(-1, 1, 1, 1).to(f_eq.dtype)
    f = f_eq + amp * cx_cy * torch.ones_like(f_eq)
    return f, rho, zero, zero


def main() -> None:
    # ---- panel (a): field on a mid-plane with a solid inclusion -------------
    n = 16
    shape = (n, n, n)
    f, rho, ux, uy = sheared_state(shape)
    uz = torch.zeros(shape, dtype=torch.float64)
    mask = sphere_mask(n, n, n, 8.0, 8.0, 8.0, 3.5, device=torch.device("cpu"))
    wss = wss_from_fneq_3d(f, rho, ux, uy, uz, tau=0.8, mask=mask)
    plane = wss[n // 2].numpy()
    solid_plane = mask[n // 2].numpy()

    # ---- panel (b): closed-form verification over the relaxation time -------
    taus = np.array([0.55, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5])
    amp = 1e-4
    errs = []
    nums = []
    for tau in taus:
        w = wss_from_fneq_3d(f, rho, ux, uy, uz, tau=float(tau), mask=torch.zeros(shape, dtype=torch.bool))
        num = float(w[0, 0, 0])
        ana = abs(-(1.0 - 0.5 / float(tau)) * amp * SXXYY)
        nums.append(num)
        errs.append(abs(num - ana) / ana)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    im = ax.imshow(plane, cmap="viridis", origin="lower")
    ax.contour(solid_plane, levels=[0.5], colors="w", linewidths=1.0)
    ax.set_title("(a) wss_from_fneq_3d, mid-plane\nuniform xy-shear + solid inclusion")
    ax.set_xlabel("x (lattice)")
    ax.set_ylabel("y (lattice)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\tau_w$ (lattice units)")

    ax = axes[1]
    ax.semilogy(taus, np.maximum(errs, 1e-12), "o-", label="relative error vs closed form")
    ax.axhline(3e-2, color="r", ls="--", lw=1.0, label="3 % acceptance band")
    ax.set_xlabel(r"relaxation time $\tau$ (-)")
    ax.set_ylabel("relative error (-)")
    ax.set_title("(b) verification vs analytic $\sigma_{xy}$\n"
                 r"$\sigma_{xy}=-(1-1/2\tau)\,A\sum_q c_x^2c_y^2$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print("fluid max wss = %.6e ; solid max wss = %.6e" % (float(wss[~mask].max()), float(wss[mask].abs().max())))
    print("sum_q cx^2 cy^2 = %.17g" % SXXYY)
    for t, nu, e in zip(taus, nums, errs):
        print("tau=%.2f numeric=%.8e analytic=%.8e relerr=%.3e" % (t, nu, abs(-(1.0 - 0.5 / t) * amp * SXXYY), e))


if __name__ == "__main__":
    main()