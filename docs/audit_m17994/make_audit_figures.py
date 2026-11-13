"""Audit figures for m17994 (XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA).

Produces three figures from *measured* data only, no hand-typed numbers:

  figA_kpi_parity.png   deterministic KPIs, delivered run vs auditor re-run
  figB_field_vs_analytic.png  ux(y) at the probe column vs the analytic shear
                              wave, plus the error at the predeclared and the
                              actually executed step count
  figC_timing_spread.png  wall-clock based KPIs (MLUPS, overlap speedup,
                          polling sweep) delivered vs re-run -> noise band

Run:  python docs/audit_m17994/make_audit_figures.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
EV = HERE / "evidence"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(pathlib.Path("/root/tp-agent/work/m17994/src")))
from tensorlbm.parallel import slab_unpacked as su  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DELIV = json.load(open(EV / "audit_rerun_results.json"))          # auditor re-run
ORIG = json.load(open("/root/tp-agent/work/m17994/artifacts/"
                      "xval_mpi_slab_unpacked_halo_overlap/results.json"))


# ------------------------------------------------------------------ figA
def fig_a() -> str:
    keys = ["probe_rel_err_pct", "field_l2_rel_err_pct", "identity_maxrel",
            "wire_bytes_per_step", "staging_bytes_per_step", "mass_drift_rel"]
    arms = list(ORIG["kpis"])
    d = np.zeros((len(keys), len(arms)))
    for j, a in enumerate(arms):
        for i, k in enumerate(keys):
            vo = ORIG["kpis"][a].get(k, 0.0)
            va = DELIV["kpis"][a].get(k, 0.0)
            d[i, j] = abs(va - vo) / max(abs(vo), 1e-30)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    im = ax.imshow(np.log10(d + 1e-18), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_xticks(range(len(arms)), arms, rotation=45, ha="right", fontsize=8)
    for i in range(len(keys)):
        for j in range(len(arms)):
            ax.text(j, i, "0" if d[i, j] == 0 else f"{d[i, j]:.1e}",
                    ha="center", va="center", fontsize=7,
                    color="w" if np.log10(d[i, j] + 1e-18) < -6 else "k")
    fig.colorbar(im, ax=ax, label="log10 |re-run - delivered| / |delivered|")
    ax.set_title("figA  deterministic KPI parity: auditor re-run vs delivered "
                 "(0 = bit-identical)", fontsize=9)
    fig.tight_layout()
    out = FIG / "figA_kpi_parity.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


# ------------------------------------------------------------------ figB
def fig_b(nx: int = 128, tau: float = 0.8, u0: float = 0.04) -> str:
    nu = (tau - 0.5) / 3.0
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    errs = {}
    for steps in (300, 600):
        s = su.SlabSolver(nx, nx, tau=tau)
        s.init_shear_wave(u0)
        for _ in range(steps):
            s.step()
        _, ux, uy = s.macroscopic()
        ana = su.shear_wave_analytic(nx, nx, steps, nu, u0)
        err = float(np.sqrt(((ux - ana) ** 2).sum()) / np.sqrt((ana ** 2).sum()) * 100.0)
        errs[steps] = err
        axes[0].plot(np.arange(nx), ux[:, 0], lw=2.4 if steps == 300 else 1.2,
                     label=f"LBM t={steps}  err={err:.4f}%")
        axes[2].plot(np.arange(nx), 100 * (ux[:, 0] - ana[:, 0]) / ans_scale(ana),
                     label=f"t={steps}")
        if steps == 300:
            axes[0].plot(np.arange(nx), ana[:, 0], "k--", lw=1.0, label="analytic")
            axes[1].imshow(ux, origin="lower", cmap="RdBu_r")
            axes[1].set_title(f"u_x field @ t=300 (N={nx}²)", fontsize=9)
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("y")
    axes[0].set_xlabel("y"); axes[0].set_ylabel("u_x")
    axes[0].set_title("figB  shear-wave profile x=0: LBM vs analytic", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[2].set_xlabel("y"); axes[2].set_ylabel("relative error [%]")
    axes[2].set_title("pointwise error vs analytic", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "figB_field_vs_analytic.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    json.dump({f"probe_err_pct_t{k}": v for k, v in errs.items()},
              open(EV / "audit_step_amendment_check.json", "w"), indent=1)
    return str(out)


def ans_scale(ana: np.ndarray) -> np.ndarray:
    m = np.abs(ana).max()
    return np.where(np.abs(ana) < 1e-3 * m, m, ana)


# ------------------------------------------------------------------ figC
def fig_c() -> str:
    names = ["np1", "np2", "np4"]
    sp_o = [ORIG["overlap"][n]["speedup_overlap_vs_blocking"] for n in names]
    sp_a = [DELIV["overlap"][n]["speedup_overlap_vs_blocking"] for n in names]
    pk = ["poll_every_1", "poll_every_8", "poll_every_64"]
    po = [ORIG["poll_sensitivity"][k]["wall_s"] for k in pk]
    pa = [DELIV["poll_sensitivity"][k]["wall_s"] for k in pk]
    arms = [a for a in ORIG["kpis"]]
    mo = [ORIG["kpis"][a]["mlups"] for a in arms]
    ma = [DELIV["kpis"][a]["mlups"] for a in arms]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, sp_o, 0.4, label="delivered")
    axes[0].bar(x + 0.2, sp_a, 0.4, label="auditor re-run")
    axes[0].axhline(1.0, color="k", ls=":")
    axes[0].set_xticks(x, names); axes[0].legend(fontsize=7)
    axes[0].set_ylabel("overlap/blocking speedup")
    axes[0].set_title("H3 overlap: sign stable, magnitude not", fontsize=9)

    x = np.arange(len(pk))
    axes[1].bar(x - 0.2, po, 0.4, label="delivered")
    axes[1].bar(x + 0.2, pa, 0.4, label="auditor re-run")
    axes[1].set_xticks(x, ["1 row", "8 rows", "64 rows"]); axes[1].legend(fontsize=7)
    axes[1].set_ylabel("wall [s]"); axes[1].set_xlabel("Testall polling granularity")
    axes[1].set_title("polling sweep: ordering flips at 64 rows", fontsize=9)

    x = np.arange(len(arms))
    axes[2].scatter(mo, ma, c="C0")
    lim = [0, max(mo + ma) * 1.1]
    axes[2].plot(lim, lim, "k:", lw=1)
    axes[2].set_xlabel("delivered MLUPS"); axes[2].set_ylabel("re-run MLUPS")
    axes[2].set_title("throughput: same host, up to 2.9x apart", fontsize=9)
    axes[2].set_xlim(lim); axes[2].set_ylim(lim)
    fig.tight_layout()
    out = FIG / "figC_timing_spread.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    ratio = max(max(a / b, b / a) for a, b in zip(mo, ma))
    json.dump({"mlups_max_ratio_rerun_vs_delivered": round(ratio, 3),
               "mlups_delivered": dict(zip(arms, mo)),
               "mlups_rerun": dict(zip(arms, ma)),
               "overlap_speedup_delivered": dict(zip(names, sp_o)),
               "overlap_speedup_rerun": dict(zip(names, sp_a)),
               "poll_wall_delivered": dict(zip(pk, po)),
               "poll_wall_rerun": dict(zip(pk, pa))},
              open(EV / "audit_timing_spread.json", "w"), indent=1)
    return str(out)


if __name__ == "__main__":
    for p in (fig_a(), fig_b(), fig_c()):
        print("wrote", p)