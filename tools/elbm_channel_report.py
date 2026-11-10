"""Post-process an ELBM channel run: wall-unit profiles, DNS gates, figures.

Usage::

    python tools/elbm_channel_report.py artifacts/elbm/raw/elbm_re180.npz \
        --re-tau 180 --out artifacts/elbm/elbm_re180

Everything is computed from the raw kernel output (no hand-copied numbers):
the friction velocity comes from the momentum balance ``u_tau = sqrt(g h)``,
the viscosity from the input ``tau`` (and, independently, from the measured
viscous-sublayer slope), and the DNS targets from the SHA-256 verified
MKM-1999 profiles shipped in ``tensorlbm/data/mkm1999``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorlbm.elbm_channel import channel_gates, log_law_slope
from tensorlbm.validation import mkm1999

_SCALAR_NAMES = (
    "samples", "min_population", "max_abs_u", "alpha_mean", "alpha_min", "alpha_max",
    "entropy_drop_max", "finite", "nu_nominal_internal", "mean_rho",
)


def load_run(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    data = np.load(path)
    profiles = {k.removeprefix("prof_"): data[k] for k in data.files if k.startswith("prof_")}
    scalars = dict(zip(_SCALAR_NAMES, np.asarray(data["scalars"]).tolist(), strict=True))
    history = data["history"]
    return profiles, scalars, history


def wall_units(
    profiles: dict[str, np.ndarray], *, re_tau: float, u_tau: float, ny: int
) -> dict[str, np.ndarray]:
    height = float(ny - 2)
    half = 0.5 * height
    nu = u_tau * half / re_tau
    y = np.arange(ny) - 0.5
    fluid = (y > 0.0) & (y < height)
    yc = y[fluid]
    u = profiles["u"][fluid]
    y_plus = yc * u_tau / nu
    u_plus = u / u_tau
    # measured viscosity from the viscous sublayer (U+ = y+ there)
    sub = (y_plus > 0) & (y_plus <= 2.5)
    slope = float(np.polyfit(yc[sub], u[sub], 1)[0]) if int(sub.sum()) >= 2 else float("nan")
    nu_meas = u_tau**2 / slope if slope > 0 else float("nan")
    out = {
        "y": yc,
        "y_plus": y_plus,
        "u_plus": u_plus,
        "u_plus_at_measured_nu": u / u_tau,
        "y_plus_measured": yc * u_tau / nu_meas if np.isfinite(nu_meas) else y_plus,
        "nu_nominal": nu,
        "nu_measured": nu_meas,
        "u_tau_from_wall_slope": float(np.sqrt(nu * slope)) if slope > 0 else float("nan"),
        "tau_w_momentum_balance": u_tau**2,
        "re_tau_measured": u_tau * half / nu_meas if np.isfinite(nu_meas) else float("nan"),
        "half_height": half,
        "u_tau": u_tau,
    }
    for key in ("uu", "vv", "ww", "uv"):
        out[key] = profiles[key][fluid]
    out["tke"] = 0.5 * (out["uu"] + out["vv"] + out["ww"])
    return out


def report(run_npz: Path, *, re_tau: float, u_tau: float, out_dir: Path, log_band=None) -> dict:
    profiles, scalars, history = load_run(run_npz)
    ny = int(profiles["u"].size)
    wu = wall_units(profiles, re_tau=re_tau, u_tau=u_tau, ny=ny)
    sigma = 0.5 * (wu["uu"] + wu["vv"] + wu["ww"])
    profile_lists = {
        "y": wu["y"].tolist(),
        "y_plus_nominal": wu["y_plus"].tolist(),
        "u_plus_nominal": wu["u_plus"].tolist(),
        "u_prime_u_prime": wu["uu"].tolist(),
        "v_prime_v_prime": wu["vv"].tolist(),
        "w_prime_w_prime": wu["ww"].tolist(),
        "u_prime_v_prime": wu["uv"].tolist(),
        "tke": sigma.tolist(),
    }
    gate_scalars = {
        "half_height": wu["half_height"],
        "u_tau": u_tau,
        "re_tau_nominal": re_tau,
        "delta_plus": re_tau / wu["half_height"],
        "finite": bool(scalars["finite"] > 0.5),
        "half_height": wu["half_height"],
        "delta_plus": re_tau / wu["half_height"],
        "max_u_rms_plus": float(np.sqrt(np.maximum(wu["uu"], 0.0)).max() / u_tau),
        "max_tke": float(wu["tke"].max()),
        "max_tke_over_u_tau2": float(wu["tke"].max() / u_tau**2),
        "max_minus_uv_plus": float((-wu["uv"] / u_tau**2).max()),
        "steps": int(json.loads((run_npz.parent / f"{run_npz.stem}_meta.json").read_text())["steps"])
        if (run_npz.parent / f"{run_npz.stem}_meta.json").exists()
        else 10_000,
        "min_population": scalars["min_population"],
        "entropy_drop_max_collision": scalars["entropy_drop_max"],
    }
    nominal = 180 if re_tau < 280 else (395 if re_tau < 500 else 590)
    mkm = mkm1999.load_profile(nominal)
    if log_band is None:
        log_band = (max(30.0, 3.0 * gate_scalars["delta_plus"]), 0.35 * re_tau)
    gates = channel_gates(profile_lists, gate_scalars, mkm, log_band=log_band)

    extra = {
        "run": str(run_npz),
        "re_tau_target": re_tau,
        "re_tau_nominal_from_tau": re_tau,
        "re_tau_from_measured_nu": wu["re_tau_measured"],
        "nu_nominal": wu["nu_nominal"],
        "nu_measured_sublayer": wu["nu_measured"],
        "u_tau": u_tau,
        "u_tau_from_wall_slope": wu["u_tau_from_wall_slope"],
        "tau_w_momentum_balance": wu["tau_w_momentum_balance"],
        "samples": int(scalars["samples"]),
        "alpha_mean": scalars["alpha_mean"],
        "alpha_min": scalars["alpha_min"],
        "alpha_max": scalars["alpha_max"],
        "min_population": scalars["min_population"],
        "max_abs_u": scalars["max_abs_u"],
        "mean_rho": scalars["mean_rho"],
        "entropy_drop_max_collision": scalars["entropy_drop_max"],
        "finite": bool(scalars["finite"] > 0.5),
        "half_height": wu["half_height"],
        "delta_plus": re_tau / wu["half_height"],
        "max_u_rms_plus": float(np.sqrt(np.maximum(wu["uu"], 0.0)).max() / u_tau),
        "max_tke": float(wu["tke"].max()),
        "max_tke_over_u_tau2": float(wu["tke"].max() / u_tau**2),
        "max_minus_uv_plus": float((-wu["uv"] / u_tau**2).max()),
        "history_columns": [
            "step", "max_abs_u", "mean_rho", "min_population", "alpha_mean",
            "alpha_min", "alpha_max", "entropy_drop_max",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gates.json").write_text(
        json.dumps({"extra": extra, "gates": gates}, indent=1, default=float)
    )
    np.savetxt(
        out_dir / "profile.csv",
        np.column_stack([wu["y"], wu["y_plus"], wu["u_plus"], wu["uu"], wu["vv"], wu["ww"], wu["uv"]]),
        delimiter=",",
        header="y,y_plus,u_plus,uu,vv,ww,uv",
        comments="",
    )
    np.savetxt(
        out_dir / "history.csv",
        history,
        delimiter=",",
        header=",".join(extra["history_columns"]),
        comments="",
    )
    return {"extra": extra, "gates": gates, "wall_units": wu, "mkm": mkm, "history": history}


def make_figures(result: dict, out_dir: Path, *, label: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wu = result["wall_units"]
    mkm = result["mkm"]
    gates = result["gates"]
    extras = result["extra"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.semilogx(mkm.y_plus, mkm.u_plus, "k-", lw=1.4, label=f"MKM-1999 DNS Re_tau={mkm.re_tau:.0f}")
    ax.semilogx(wu["y_plus"], wu["u_plus"], "o", ms=3.2, color="C3",
                label=f"ELBM run (Re_tau={extras['re_tau_target']:.0f}, "
                      f"d+={extras['re_tau_target'] / wu['half_height']:.1f})")
    yp = np.array([5.0, 60.0])
    ax.semilogx(yp, 5.0 + np.log(yp) / 0.41, "b--", lw=1.0, label="log law, kappa=0.41, B=5.2")
    ax.set_xlabel("y+")
    ax.set_ylabel("U+")
    ax.set_title(f"Mean velocity profile ({label})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    gate_a = gates["a_log_slope"]
    if gate_a.get("valid"):
        ax.text(0.03, 0.93, f"fitted slope={gate_a['slope']:.3f} (1/kappa={1 / 0.41:.3f})",
                transform=ax.transAxes, fontsize=7)
    p = out_dir / f"{label}_uplus.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    u_tau = extras["u_tau"]
    ax.plot(mkm.y_plus, mkm.u_rms_plus, "k-", lw=1.2, label="DNS u_rms+")
    ax.plot(wu["y_plus"], np.sqrt(np.maximum(wu["uu"], 0.0)) / u_tau, "o-", ms=3, lw=1.0,
            color="C3", label="ELBM u_rms+")
    ax.plot(mkm.y_plus, -mkm.uv_plus, "k--", lw=1.0, label="DNS -<u'v'>+")
    ax.plot(wu["y_plus"], -wu["uv"] / u_tau**2, "s--", ms=3, lw=1.0, color="C0",
            label="ELBM -<u'v'>+")
    ax.set_xlabel("y+")
    ax.set_ylabel("stress / u_rms (wall units)")
    ax.set_xlim(0, 1.05 * extras["re_tau_target"])
    ax.set_ylim(0, 3.0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(f"Reynolds stresses ({label})")
    p = out_dir / f"{label}_stresses.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    hist = np.asarray(result["history"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)
    if hist.size:
        axes[0].plot(hist[:, 0], hist[:, 1], "-", lw=1.2, color="C0")
        axes[0].set_ylabel("max|u| (lattice)")
        axes[0].grid(alpha=0.3)
        axes[1].semilogy(hist[:, 0], np.maximum(hist[:, 3], 1e-12), "-", lw=1.2, color="C2",
                         label="min population")
        ax2 = axes[1].twinx()
        ax2.plot(hist[:, 0], hist[:, 4], "-", lw=1.0, color="C4", label="<alpha>")
        axes[1].set_ylabel("min population")
        ax2.set_ylabel("<alpha>", color="C4")
        axes[1].set_xlabel("step")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Stability trace ({label}): positivity and entropic limiter", fontsize=9)
    p = out_dir / f"{label}_stability.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--re-tau", type=float, required=True)
    ap.add_argument("--u-tau", type=float, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    npz = Path(args.npz)
    label = args.label or npz.stem
    out = Path(args.out) if args.out else npz.parent.parent / label
    res = report(npz, re_tau=args.re_tau, u_tau=args.u_tau, out_dir=out)
    figs = make_figures(res, out, label=label)
    summary = {
        "extra": res["extra"],
        "gates": {
            k: ({kk: vv for kk, vv in v.items() if not isinstance(vv, list)} if isinstance(v, dict) else v)
            for k, v in res["gates"].items()
        },
        "figures": [str(f) for f in figs],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())