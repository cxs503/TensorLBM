"""Analytic verification campaign for ``tensorlbm.vortex_identification``.

Generates the numbers and figures quoted in
``docs/vortex_identification_verification.md``:

* grid-convergence table for Q / lambda2 / Omega against the closed-form
  Lamb-Oseen vortex,
* scalar acceptance quantities (Q = 0 core radius, Omega = 0.52 core radius),
* field / contour figure and radial-profile figure,
* unit-scaling check for the physical grid spacing ``dx``.

Usage::

    PYTHONPATH=src python3 scripts/verify_vortex_identification.py

Writes ``artifacts/vortex_identification_verification/`` (table.json + PNGs).
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.vortex_identification import (  # noqa: E402
    lambda2_criterion_2d,
    omega_criterion_2d,
    q_criterion_2d,
)

RC = 1.0
L_DOMAIN = 8.0 * RC
OUT = os.path.join("artifacts", "vortex_identification_verification")


def lob_velocity_2d(n: int, length: float = L_DOMAIN, rc: float = RC):
    h = length / n
    c = (np.arange(n) + 0.5) * h - 0.5 * length
    xg, yg = np.meshgrid(c, c)
    r2 = xg**2 + yg**2
    r = np.sqrt(r2)
    r_safe = np.where(r < 1e-30, 1.0, r)
    x = r2 / rc**2
    u_theta = rc * (1.0 - np.exp(-x)) / r_safe
    return h, xg, yg, x, r, -u_theta * yg / r_safe, u_theta * xg / r_safe


def lob_q(x):
    ex = np.exp(-x)
    s = ((x + 1.0) * ex - 1.0) / np.where(x == 0, 1.0, x)
    return ex * ex - s * s


def lob_s(x):
    ex = np.exp(-x)
    return ((x + 1.0) * ex - 1.0) / np.where(x == 0, 1.0, x)


def lob_omega(x, eps=0.0):
    ex = np.exp(-x)
    s = lob_s(x)
    return ex * ex / (ex * ex + s * s + eps)


def bisect(f, lo, hi, n=200):
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def crossing(r, f, level):
    for i in range(1, len(f)):
        if (f[i - 1] - level) * (f[i] - level) < 0:
            return r[i - 1] + (level - f[i - 1]) * (r[i] - r[i - 1]) / (f[i] - f[i - 1])
    return float("nan")


def main() -> dict:
    os.makedirs(OUT, exist_ok=True)
    xq = bisect(lambda x: (2 * x + 1) * math.exp(-x) - 1.0, 0.5, 2.0)
    r_q = math.sqrt(xq)
    r_om = math.sqrt(bisect(lambda x: float(lob_omega(np.array([x]))[0]) - 0.52, 0.01, 5.0))

    rows = []
    fields = {}
    for n in (32, 64, 128, 256, 512):
        h, xg, yg, x, r, ux, uy = lob_velocity_2d(n)
        ux_t, uy_t = torch.tensor(ux), torch.tensor(uy)
        q = q_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
        l2 = lambda2_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
        om = omega_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
        qe, ome = lob_q(x), lob_omega(x, eps=1e-5)
        inner = (slice(1, -1), slice(1, -1))
        scale_q = float(np.abs(qe[inner]).max())
        j = int(np.argmin(np.abs(yg[:, 0])))
        i0 = int(np.searchsorted(xg[0], 0.0)) - 1
        rows.append(
            {
                "n": n,
                "h_over_rc": h / RC,
                "q_linf_rel": float(np.abs(q[inner] - qe[inner]).max() / scale_q),
                "q_l2_rel": float(np.sqrt(((q[inner] - qe[inner]) ** 2).mean()) / scale_q),
                "l2_linf_rel": float(np.abs(l2[inner] + q[inner]).max() / scale_q),
                "omega_linf_abs": float(np.abs(om - ome)[inner].max()),
                "r_q_meas": float(crossing(xg[0, i0:], q[j, i0:], 0.0) / RC),
                "r_omega_meas": float(crossing(xg[0, i0:], om[j, i0:], 0.52) / RC),
            }
        )
        if n == 256:
            fields = dict(h=h, xg=xg, yg=yg, x=x, q=q, qe=qe, om=om, ome=ome)

    for row in rows:
        row["r_q_err_rel"] = abs(row["r_q_meas"] / r_q - 1.0)
        row["r_omega_err_rel"] = abs(row["r_omega_meas"] / r_om - 1.0)

    # consecutive refinement orders for q_linf_rel
    orders = []
    for a, b in zip(rows, rows[1:]):
        orders.append(math.log2(a["q_linf_rel"] / b["q_linf_rel"]))
    for i, o in enumerate(orders):
        rows[i]["p_q_linf"] = o

    # ---- unit-scaling (dx) check ----------------------------------------
    h, _xg, _yg, _x, _r, ux, uy = lob_velocity_2d(128)
    ux_t, uy_t = torch.tensor(ux), torch.tensor(uy)
    q_unit = q_criterion_2d(ux_t, uy_t).numpy()
    q_phys = q_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    inner = (slice(1, -1), slice(1, -1))
    ratio = float((q_unit[inner] / q_phys[inner]).mean())
    dx_check = {
        "h": h,
        "expected_ratio_h2": h**2,
        "measured_ratio": ratio,
        "rel_dev": abs(ratio / h**2 - 1.0),
        "q_max_unit_grid": float(np.abs(q_unit).max()),
        "q_max_physical": float(np.abs(q_phys).max()),
    }

    # ---- 3-D axisymmetric consistency -----------------------------------
    from tensorlbm.vortex_identification import (
        lambda2_criterion_3d,
        omega_criterion_3d,
        q_criterion_3d,
    )

    n3, nz = 96, 8
    dz = 2.0 * RC / nz
    h3, _x3, _y3, x3, _r3, ux3n, uy3n = lob_velocity_2d(n3)
    ux3 = torch.tensor(np.broadcast_to(ux3n, (nz, n3, n3)).copy())
    uy3 = torch.tensor(np.broadcast_to(uy3n, (nz, n3, n3)).copy())
    uz3 = torch.zeros_like(ux3)
    q3 = q_criterion_3d(ux3, uy3, uz3, dx=h3, dy=h3, dz=dz).numpy()
    l23 = lambda2_criterion_3d(ux3, uy3, uz3, dx=h3, dy=h3, dz=dz).numpy()
    om3 = omega_criterion_3d(ux3, uy3, uz3, dx=h3, dy=h3, dz=dz).numpy()
    qe3 = np.broadcast_to(lob_q(x3), (nz, n3, n3))
    ome3 = np.broadcast_to(lob_omega(x3, eps=1e-5), (nz, n3, n3))
    inner3 = (slice(1, -1), slice(1, -1), slice(1, -1))
    scale3 = float(np.abs(qe3[inner3]).max())
    three_d = {
        "n": n3,
        "nz": nz,
        "q_linf_rel": float(np.abs(q3 - qe3)[inner3].max() / scale3),
        "l2_plus_q_rel": float(np.abs(l23 + q3)[inner3].max() / scale3),
        "omega_linf_abs": float(np.abs(om3 - ome3)[inner3].max()),
    }

    # ---- exact flows: rigid rotation and pure strain ---------------------
    n2 = 64
    h2 = 2.0 / n2
    c2 = (np.arange(n2) + 0.5) * h2 - 1.0
    xg2, yg2 = np.meshgrid(c2, c2)
    inner2 = (slice(1, -1), slice(1, -1))
    w0, a0 = 1.7, 1.3
    q_rot = q_criterion_2d(torch.tensor(-w0 * yg2), torch.tensor(w0 * xg2), dx=h2, dy=h2).numpy()
    l_rot = lambda2_criterion_2d(torch.tensor(-w0 * yg2), torch.tensor(w0 * xg2), dx=h2, dy=h2).numpy()
    q_str = q_criterion_2d(torch.tensor(a0 * xg2), torch.tensor(-a0 * yg2), dx=h2, dy=h2).numpy()
    l_str = lambda2_criterion_2d(torch.tensor(a0 * xg2), torch.tensor(-a0 * yg2), dx=h2, dy=h2).numpy()
    om_str = omega_criterion_2d(torch.tensor(a0 * xg2), torch.tensor(-a0 * yg2), dx=h2, dy=h2).numpy()
    exact = {
        "rigid_rotation": {
            "omega0": w0,
            "q_exact": w0**2,
            "q_rel_dev": float(np.abs(q_rot[inner2] / w0**2 - 1.0).max()),
            "l2_exact": -(w0**2),
            "l2_rel_dev": float(np.abs(l_rot[inner2] / (-(w0**2)) - 1.0).max()),
        },
        "pure_strain": {
            "a": a0,
            "q_exact": -(a0**2),
            "q_rel_dev": float(np.abs(q_str[inner2] / (-(a0**2)) - 1.0).max()),
            "l2_exact": a0**2,
            "l2_rel_dev": float(np.abs(l_str[inner2] / (a0**2) - 1.0).max()),
            "omega_max": float(om_str[inner2].max()),
        },
    }

    table = {
        "reference": "Lamb-Oseen vortex, Gamma/(2*pi) = 1, r_c = 1, domain 8 r_c",
        "analytic": {
            "r_q_over_rc": r_q,
            "r_omega_052_over_rc": r_om,
            "q_max": 1.0,
        },
        "convergence": rows,
        "3d_axisymmetric": three_d,
        "exact_flows": exact,
        "dx_scaling": dx_check,
        "acceptance": {
            "q_linf_rel_at_n128": [r for r in rows if r["n"] == 128][0]["q_linf_rel"],
            "threshold": 0.03,
            "pass": bool([r for r in rows if r["n"] == 128][0]["q_linf_rel"] < 0.03),
        },
    }
    with open(os.path.join(OUT, "table.json"), "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)

    # ---- figure 1: field + error map + contours -------------------------
    q, qe, om = fields["q"], fields["qe"], fields["om"]
    err = q - qe
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    extent = (-L_DOMAIN / 2, L_DOMAIN / 2, -L_DOMAIN / 2, L_DOMAIN / 2)
    im0 = axes[0].imshow(q, origin="lower", extent=extent, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    axes[0].contour(fields["xg"], fields["yg"], q, levels=[0.0], colors="k", linewidths=1.2)
    axes[0].contour(
        fields["xg"], fields["yg"], om, levels=[0.52], colors="lime", linestyles="--", linewidths=1.2
    )
    axes[0].set_title("Q-criterion, Oseen vortex (n=256)\nblack: Q=0, dashed: $\\Omega$=0.52")
    axes[0].set_xlabel("x / $r_c$")
    axes[0].set_ylabel("y / $r_c$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="Q / ($\\Gamma/2\\pi r_c^{-2}$)$^2$")

    im1 = axes[1].imshow(err, origin="lower", extent=extent, cmap="PuOr", vmin=-2e-3, vmax=2e-3)
    axes[1].set_title("Q error (numerical - analytic), n=256\n$\\|e\\|_\\infty/\\max|Q|$ = %.2e"
                      % rows[2]["q_linf_rel"])
    axes[1].set_xlabel("x / $r_c$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="Q error")

    # radial profiles along y ~ 0
    j = int(np.argmin(np.abs(fields["yg"][:, 0])))
    i0 = int(np.searchsorted(fields["xg"][0], 0.0)) - 1
    rr = fields["xg"][0, i0:] / RC
    axes[2].plot(rr, q[j, i0:], "o", ms=3, mfc="none", label="Q numerical")
    axes[2].plot(rr, lob_q(fields["x"][j, i0:]), "-", label="Q analytic")
    axes[2].plot(rr, om[j, i0:], "s", ms=3, mfc="none", label="$\\Omega$ numerical")
    axes[2].plot(rr, lob_omega(fields["x"][j, i0:], eps=1e-5), "--", label="$\\Omega$ analytic")
    axes[2].axhline(0.0, color="0.6", lw=0.8)
    axes[2].axvline(r_q, color="k", lw=0.8, ls=":")
    axes[2].annotate("$r_Q$=%.4f$r_c$" % r_q, (r_q, 0.35), fontsize=8)
    axes[2].axhline(0.52, color="g", lw=0.8, ls=":")
    axes[2].set_xlim(0, 4)
    axes[2].set_ylim(-0.25, 1.05)
    axes[2].set_xlabel("r / $r_c$")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].set_title("radial profiles through the core")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_q_field_profiles.png"), dpi=140)
    plt.close(fig)

    # ---- figure 2: convergence ------------------------------------------
    hs = [r["h_over_rc"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(hs, [r["q_linf_rel"] for r in rows], "o-", label="$\\|Q-Q_{ex}\\|_\\infty/\\max|Q|$")
    axes[0].loglog(hs, [r["q_l2_rel"] for r in rows], "s-", label="L2 norm")
    axes[0].loglog(hs, [r["l2_linf_rel"] for r in rows], "^-", label="$\\|\\lambda_2+Q\\|_\\infty/\\max|Q|$")
    axes[0].loglog(hs, [r["omega_linf_abs"] for r in rows], "d-", label="$\\|\\Omega-\\Omega_{ex}\\|_\\infty$")
    ref = [rows[0]["q_linf_rel"] * (h / hs[0]) ** 2 for h in hs]
    axes[0].loglog(hs, ref, "k--", lw=1.0, label="$O(h^2)$ reference")
    axes[0].axhline(0.03, color="r", ls=":", lw=1.0, label="3 % acceptance")
    axes[0].set_xlabel("h / $r_c$")
    axes[0].set_ylabel("relative error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("grid convergence of the criteria")

    axes[1].plot(hs, [100 * r["r_q_err_rel"] for r in rows], "o-", label="$r_Q$ (Q=0 contour)")
    axes[1].plot(hs, [100 * r["r_omega_err_rel"] for r in rows], "s-", label="$r$ at $\\Omega$=0.52")
    axes[1].axhline(3.0, color="r", ls=":", label="3 % acceptance")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("h / $r_c$")
    axes[1].set_ylabel("core-radius error [%]")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("scalar acceptance: vortex-core radius")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_convergence.png"), dpi=140)
    plt.close(fig)

    print(json.dumps({k: v for k, v in table.items() if k not in ("convergence",)}, indent=2))
    hdr = "  n     h/rc     Q_Linf_rel  Q_L2_rel   |l2+Q|_rel  |Om-Om_ex|  p(Q)   r_Q/rc    err%    r_Om/rc   err%"
    print(hdr)
    for r in rows:
        print(
            f"{r['n']:4d}  {r['h_over_rc']:8.5f}  {r['q_linf_rel']:10.3e}  {r['q_l2_rel']:9.2e}  "
            f"{r['l2_linf_rel']:10.2e}  {r['omega_linf_abs']:9.2e}  "
            f"{r.get('p_q_linf', float('nan')):5.2f}  {r['r_q_meas']:8.5f}  {100*r['r_q_err_rel']:6.3f}  "
            f"{r['r_omega_meas']:8.5f}  {100*r['r_omega_err_rel']:6.3f}"
        )
    return table


if __name__ == "__main__":
    main()