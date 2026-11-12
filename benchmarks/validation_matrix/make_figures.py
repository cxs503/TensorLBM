"""Figures for the validation-matrix report (matplotlib, no hand-typed numbers).

Two panels per figure, all numbers come from the run itself:

1. ``dam_break_ritter_ci.png``
   (a) depth profile h(x) at t=5 sqrt(h0/g) for nx=200/400/800 vs the analytic
       Ritter profile (the field-level comparison);
   (b) |front error| and |profile L2 error| vs nx on log axes (the convergence
       ladder that decides which resolution the CI gate needs).

2. ``static_droplet_laplace_ci.png``
   (a) centreline / radial density profile of an R=16 droplet (interface
       resolution, the field-level comparison);
   (b) measured Laplace pressure jump vs 1/R_eq with the fitted line
       (the analytic law comparison).

Usage::

    python benchmarks/validation_matrix/make_figures.py [--out DIR] [--skip-droplet]

Figure provenance: the script prints the sha256 of every artifact it reads and
writes a ``figures/provenance.json`` next to the PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CASE_ROOT.parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dam_break_profile(nx: int, t_end: float = 5.0):
    from tensorlbm.shallow_water import (
        ShallowWaterConfig,
        ritter_depth,
        run_ritter_dam_break,
    )

    cfg = ShallowWaterConfig(nx=nx, t_end=t_end, sample_every=max(1, nx // 4))
    run = run_ritter_dam_break(cfg)
    t = run.time[-1]
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    h = np.asarray(run.profiles[-1][1], dtype=float)
    h_ref = np.array([ritter_depth(float(xi), t, cfg.h0, cfg.g) for xi in x])
    return x, h, h_ref, t, run


def figure_dam_break(out_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ladder = [200, 400, 800]
    metrics = {}
    x_ref = None
    h_ref = None
    for nx in ladder:
        x, h, h_r, t, run = _dam_break_profile(nx)
        x_ref, h_ref = x, h_r
        axes[0].plot(x, h, lw=1.4, label=f"nx={nx}")
        # front error from the run itself (integral form of the Ritter law)
        from tensorlbm.shallow_water import front_from_mass, ritter_front

        front_num = front_from_mass(run.volume_downstream[-1], 1.0)
        front_ana = ritter_front(t, 1.0, 1.0)
        mask = (x >= 0) & (x <= front_ana)
        l2 = float(np.linalg.norm(h[mask] - h_r[mask]) / np.linalg.norm(h_r[mask])) * 100
        metrics[nx] = {
            "front_err_pct": (front_num - front_ana) / front_ana * 100,
            "profile_l2_pct": l2,
        }
    axes[0].plot(x_ref, h_ref, "k--", lw=1.2, label="Ritter analytic")
    axes[0].set_xlim(0, 11)
    axes[0].set_xlabel("x / h0")
    axes[0].set_ylabel("h / h0")
    axes[0].set_title("dam break: depth profile at t = 5 sqrt(h0/g)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    nxs = sorted(metrics)
    axes[1].loglog(nxs, [abs(metrics[n]["front_err_pct"]) for n in nxs], "o-",
                   label="|front error| [%] (5% gate)")
    axes[1].loglog(nxs, [metrics[n]["profile_l2_pct"] for n in nxs], "s-",
                   label="profile L2 error [%] (5% gate)")
    axes[1].axhline(5.0, color="r", ls="--", lw=1.0, label="CI gate 5%")
    axes[1].set_xlabel("nx")
    axes[1].set_ylabel("error [%]")
    axes[1].set_title("convergence ladder (first order)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "dam_break_ritter_ci.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"figure": str(path.relative_to(REPO_ROOT)), "ladder": metrics}


def figure_droplet(out_dir: Path, radius: float = 16.0, max_steps: int = 16000):
    import importlib.util

    laplace = REPO_ROOT / "benchmarks" / "verified" / "laplace_droplet" / "run.py"
    spec = importlib.util.spec_from_file_location("laplace_droplet_case_fig", laplace)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    device = torch.device("cpu")
    L = int(4 * radius)
    final, _ = mod.run_droplet(2, radius, L, device, max_steps, 6000, 1000,
                               compile_mode="eager")
    result_path = CASE_ROOT / "cases" / "static_droplet_laplace_ci" / "result.json"
    payload = json.loads(result_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) field: radial density profile of the R=16 droplet
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.multiphase import collide_sc_single_component, psi_exp
    from tensorlbm.solver import stream

    rho0 = mod.init_rho(2, L, radius, device)
    rr = mod._radius_field(2, L, device)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    for _ in range(max_steps):
        f = stream(collide_sc_single_component(f, G=5.0, tau=1.0, psi_fn=psi_exp))
    rho = f.sum(dim=0)
    bins = np.arange(0, L / 2, 0.5)
    r = rr.flatten().numpy()
    rv = rho.flatten().numpy()
    idx = np.digitize(r, bins) - 1
    prof = np.array([rv[idx == i].mean() for i in range(len(bins) - 1)])
    axes[0].plot(bins[:-1], prof, ".", ms=3, label=f"radial mean rho, R={radius:g}")
    axes[0].axvline(final["R_eq"], color="k", ls="--", lw=1,
                    label=f"R_eq = {final['R_eq']:.2f}")
    axes[0].set_xlabel("r [lattice]")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_title("static droplet: density interface (SCMP, G_lib=+5)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # (b) Laplace law from the judged artifact
    rows = []
    for lv in payload["levels"]:
        rows.extend(lv["detail"]["per_radius"])
    inv_r = np.array([1.0 / r["R_eq"] for r in rows])
    dp = np.array([r["dp"] for r in rows])
    sigma = float(np.sum(inv_r * dp) / np.sum(inv_r * inv_r))
    axes[1].plot(inv_r, dp, "o", label="measured $\\Delta p$ (all levels)")
    xx = np.linspace(0, inv_r.max() * 1.05, 50)
    axes[1].plot(xx, sigma * xx, "k--", label=f"$\\Delta p = \\sigma/R$, $\\sigma$={sigma:.4f}")
    axes[1].set_xlabel("1 / R_eq")
    axes[1].set_ylabel("$\\Delta p$")
    axes[1].set_title("Young-Laplace law (2-D)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "static_droplet_laplace_ci.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "figure": str(path.relative_to(REPO_ROOT)),
        "sigma_fit_all_levels": sigma,
        "R_eq_field_run": final["R_eq"],
        "dp_field_run": final["dp"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(CASE_ROOT / "figures"))
    ap.add_argument("--skip-droplet", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = {"figures": [], "artifacts": []}
    art = CASE_ROOT / "cases" / "dam_break_ritter_ci" / "result.json"
    prov["artifacts"].append({"path": str(art.relative_to(REPO_ROOT)),
                              "sha256": _sha256(art)})
    prov["figures"].append(figure_dam_break(out))
    if not args.skip_droplet:
        dr = CASE_ROOT / "cases" / "static_droplet_laplace_ci" / "result.json"
        if dr.exists():
            prov["artifacts"].append({"path": str(dr.relative_to(REPO_ROOT)),
                                      "sha256": _sha256(dr)})
            prov["figures"].append(figure_droplet(out))
        else:
            prov["figures"].append({"figure": None,
                                    "reason": f"{dr} missing; droplet panel skipped"})
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    for f in prov["figures"]:
        print(f.get("figure"), f.get("reason", ""))
    print(f"provenance -> {out / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())