"""Figures for the NWT acceptance report (matplotlib, Agg backend).

Every figure is regenerated from the recorded artefacts (``case.json`` and
``probes.npz``); no number is transcribed by hand into the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KH_LEVELS = (0.7, 1.5, 3.0)
EPS_LEVELS = (0.02, 0.08)


def _load_probes(out_dir: Path, name: str):
    path = out_dir / name / "probes.npz"
    if not path.exists():
        return None
    return np.load(path)


def fig_dispersion(result: dict, out_dir: Path, path: Path) -> str:
    cases = result["cases"]
    names = [n for n in cases if n.startswith("kh") and "_nosponge" not in n and "_dx_" not in n]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {0.7: "#1b6ca8", 1.5: "#e07b39", 3.0: "#3a7d44"}
    for n in names:
        d = cases[n]
        disp = d["dispersion"]
        kh = d["case"]["kh"]
        c_m = disp["phase_speed_measured"]
        c_c = disp["phase_speed_closed_form_at_k_measured"]
        ax[0].scatter(
            c_c,
            c_m,
            s=60,
            color=colors.get(round(kh, 1), "k"),
            marker="o" if d["case"]["steepness"] < 0.05 else "s",
            label=f"kh={kh:.1f}, ka={d['case']['steepness']:.2f}",
        )
    lim = ax[0].get_xlim()
    lo, hi = min(lim) * 0.98, max(lim) * 1.02
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="c = sqrt((g/k) tanh(kh))")
    ax[0].fill_between([lo, hi], [lo * 0.99, hi * 0.99], [lo * 1.01, hi * 1.01], color="0.85", zorder=0, label="+/-1 % band")
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("closed-form phase speed  c_th(k_meas)  [lattice]")
    ax[0].set_ylabel("measured phase speed  c_num  [lattice]")
    ax[0].set_title("C1 dispersion: measured vs closed form")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    labels, errs = [], []
    kh_worst = result["criteria"]["C1_dispersion"]["worst_error_by_kh"]
    for key, val in kh_worst.items():
        labels.append(f"kh={key}")
        errs.append(100.0 * val)
    bars = ax[1].bar(labels, errs, color="#4c72b0")
    ax[1].axhline(1.0, color="r", ls="--", lw=1.2, label="1 % acceptance")
    ax[1].set_ylabel("|c_num - c_th| / c_th  [%]")
    ax[1].set_title("C1 worst-case dispersion error per kh")
    for b, e in zip(bars, errs):
        ax[1].text(b.get_x() + b.get_width() / 2, e, f"{e:.2f}%", ha="center", va="bottom", fontsize=8)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def fig_waveform(result: dict, out_dir: Path, path: Path) -> str:
    cases = result["cases"]
    picks = [("kh3.0_eps020", 0.02), ("kh3.0_eps080", 0.08)]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    for ax, (name, eps) in zip(axes, picks):
        d = cases.get(name)
        pr = _load_probes(out_dir, name)
        if d is None or pr is None:
            ax.text(0.5, 0.5, f"{name}: missing", ha="center", transform=ax.transAxes)
            continue
        eta = pr["eta"]
        omega = float(pr["omega"])
        mid = eta.shape[0] // 2
        t = np.arange(eta.shape[1], dtype=float)
        amp = d["case"]["amplitude_a"]
        a2 = d["case"]["second_harmonic_amplitude"]
        analytic = amp * np.cos(omega * t) + a2 * np.cos(2.0 * omega * t)
        # align the analytic phase with the measured record at the mid probe
        win = slice(int(0.2 * len(t)), int(0.2 * len(t)) + int(2 * 2 * math.pi / omega))
        measured = eta[mid, win]
        tt = t[win]
        best = None
        for shift in np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False):
            ref = amp * np.cos(omega * tt + shift) + a2 * np.cos(2.0 * omega * tt + 2.0 * shift)
            err = float(np.sqrt(np.mean((measured - ref) ** 2)))
            if best is None or err < best[0]:
                best = (err, shift, ref)
        ax.plot(tt, measured, lw=1.2, label=f"LBM probe {mid} (kh={d['case']['kh']:.2f}, ka={eps})")
        ax.plot(tt, best[2], "r--", lw=1.0, label=f"Stokes-2nd analytic (rms misfit {best[0]:.2e})")
        ax.set_ylabel("eta  [lattice]")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_title("C2 waveform: recorded elevation vs closed-form Stokes-2nd elevation")
    axes[-1].set_xlabel("time step (analysis window)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def fig_harmonics(result: dict, out_dir: Path, path: Path) -> str:
    rows = result["criteria"]["C2_waveform"]["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    labels = [f"kh{r['kh']:.1f}\nka={r['eps']:.2f}" for r in rows]
    idx = np.arange(len(rows))
    a1m = [r["A1_measured"] for r in rows]
    a1a = [r["A1_analytic"] for r in rows]
    a2m = [r["A2_measured"] for r in rows]
    a2a = [r["A2_analytic"] for r in rows]
    ax[0].bar(idx - 0.2, a1a, 0.4, label="analytic A1", color="#999999")
    ax[0].bar(idx + 0.2, a1m, 0.4, label="measured A1", color="#4c72b0")
    ax[0].set_xticks(idx, labels, fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("A1  [lattice]")
    ax[0].set_title("C2 first harmonic (tolerance 3 %)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    for i, r in enumerate(rows):
        ax[0].text(i, max(a1a[i], a1m[i]) * 1.15, f"{100 * r['A1_error']:.2f}%", ha="center", fontsize=8)

    ax[1].bar(idx - 0.2, a2a, 0.4, label="analytic A2", color="#999999")
    ax[1].bar(idx + 0.2, a2m, 0.4, label="measured A2", color="#c44e52")
    ax[1].set_xticks(idx, labels, fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("A2  [lattice]")
    ax[1].set_title("C2 second harmonic (tolerance 10 %)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    for i, r in enumerate(rows):
        ax[1].text(i, max(a2a[i], a2m[i]) * 1.15, f"{100 * r['A2_error']:.2f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def fig_reflection(result: dict, out_dir: Path, path: Path) -> str:
    rows = result["criteria"]["C3_absorption"]["rows"]
    neg = result["criteria"]["C3_absorption"]["negative_control_KR"]
    labels = [r["case"].replace("kh", "").replace("_eps", "\nka=").replace("_dx_", " ") for r in rows]
    vals = [100.0 * (r["KR_two_point"] if r["KR_two_point"] is not None else float("nan")) for r in rows]
    lsq = [100.0 * (r["KR_least_squares"] if r["KR_least_squares"] is not None else float("nan")) for r in rows]
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, vals, 0.4, label="Goda-Suzuki two-point", color="#4c72b0")
    ax.bar(idx + 0.2, lsq, 0.4, label="least squares (all probes)", color="#55a868")
    if neg is not None:
        ax.axhline(100.0 * neg, color="#c44e52", ls=":", lw=1.5, label=f"negative control (absorber off): {100 * neg:.1f} %")
    ax.axhline(5.0, color="r", ls="--", lw=1.2, label="5 % acceptance")
    ax.axhline(30.0, color="0.4", ls="-.", lw=1.0, label="30 % negative-control floor")
    ax.set_xticks(idx, labels, fontsize=8)
    ax.set_ylabel("reflection coefficient  KR  [%]")
    ax.set_title("C3 absorption: two-point reflection coefficient per case")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def fig_surface(result: dict, out_dir: Path, path: Path) -> str:
    name = "kh3.0_eps080" if "kh3.0_eps080" in result["cases"] else next(iter(result["cases"]))
    d = result["cases"][name]
    pr = _load_probes(out_dir, name)
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    if pr is not None and "eta_snap" in pr:
        snap = pr["eta_snap"]
        steps = pr["snap_steps"]
        x = np.arange(snap.shape[1])
        for i in range(snap.shape[0]):
            ax.plot(x[1:-1], snap[i, 1:-1], lw=0.8, color="0.55")
        ax.plot(x[1:-1], snap[-1, 1:-1], lw=1.6, color="#4c72b0", label=f"LBM free surface (step {int(steps[-1])})")
        # analytic profile, phase-aligned to the last snapshot by a global fit
        omega = float(pr["omega"])
        k = 2.0 * math.pi / d["case"]["wavelength"]
        amp = d["case"]["amplitude_a"]
        a2 = d["case"]["second_harmonic_amplitude"]
        xs = x[1:-1].astype(float)
        best = None
        for shift in np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False):
            ref = amp * np.cos(k * xs - shift) + a2 * np.cos(2.0 * (k * xs - shift))
            err = float(np.sqrt(np.mean((snap[-1, 1:-1] - ref) ** 2)))
            if best is None or err < best[0]:
                best = (err, ref, shift)
        ax.plot(xs, best[1], "r--", lw=1.2, label=f"Stokes-2nd analytic (rms misfit {best[0]:.2e})")
        geom = d["diagnostics"]
        for x0, col, lab in (
            (geom.get("x_generator_end", 0), "#f0c419", "end of wavemaker zone"),
            (geom.get("x_sponge_start", 0), "#7fb069", "start of absorber"),
        ):
            ax.axvline(x0, color=col, lw=1.2, ls=":", label=lab)
        for px in d["probe_x"]:
            ax.plot([px], [amp * 1.6], "k^", ms=4)
        ax.set_xlabel("x  [lattice]")
        ax.set_ylabel("eta  [lattice]")
        ax.set_title(f"Free-surface field, case {name} (kh={d['case']['kh']:.2f}, ka={d['case']['steepness']:.2f}); triangles = probes")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no snapshot recorded", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def fig_budget(result: dict, out_dir: Path, path: Path) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    rows = result["criteria"]["C4_mass"]["rows"]
    labels = [r["case"] for r in rows]
    vals = [100.0 * r["max_per_period_drift"] for r in rows]
    idx = np.arange(len(rows))
    ax[0].bar(idx, vals, color="#4c72b0")
    ax[0].axhline(0.5, color="r", ls="--", lw=1.2, label="0.5 %/period acceptance")
    ax[0].set_xticks(idx, labels, rotation=60, fontsize=7, ha="right")
    ax[0].set_ylabel("max relative liquid-inventory drift [%/period]")
    ax[0].set_title("C4 mass conservation")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    c5 = result["criteria"]["C5_grid_convergence"]
    if "dx_base" in c5:
        b = c5["dx_base"]
        f = c5["dx_fine"]
        ax[1].bar([0, 1], [b["A1"], f["A1"]], color=["#4c72b0", "#55a868"], width=0.55)
        ax[1].axhline(b["A1"] * 0.97, color="r", ls="--", lw=1.0)
        ax[1].axhline(b["A1"] * 1.03, color="r", ls="--", lw=1.0, label="+/-3 % of coarse")
        ax[1].set_xticks([0, 1], [f"A1 at dx (lam={b['wavelength']:.0f})", f"A1 at dx/2 (lam={f['wavelength']:.0f})"], fontsize=8)
        ax[1].set_ylabel("A1  [lattice]")
        ax[1].set_title(f"C5 grid convergence: change {100 * c5['relative_change']:.2f} %")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis="y")
    else:
        ax[1].text(0.5, 0.5, "grid-convergence pair missing", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def make_all_figures(result: dict, out_dir: Path, figures_dir: Path) -> list[dict]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    made = [
        ("fig1_dispersion.png", "C1: measured phase speed vs the closed-form dispersion relation, and the worst-case error per kh.", fig_dispersion),
        ("fig2_waveform.png", "C2: recorded elevation at a mid-flume probe vs the closed-form second-order Stokes elevation.", fig_waveform),
        ("fig3_harmonics.png", "C2: first- and second-harmonic amplitudes, measured vs analytic, for every kh/ka pair.", fig_harmonics),
        ("fig4_reflection.png", "C3: Goda-Suzuki two-point and least-squares reflection coefficients, with the absorber-off negative control.", fig_reflection),
        ("fig5_surface.png", "Free-surface field along the flume (last snapshot) vs the analytic profile, with the wavemaker and absorber zones marked.", fig_surface),
        ("fig6_budget.png", "C4 inventory drift per period and C5 grid convergence of the first-harmonic amplitude.", fig_budget),
    ]
    out = []
    for fname, caption, fn in made:
        p = figures_dir / fname
        try:
            fn(result, out_dir, p)
            out.append({"name": fname, "caption": caption, "path": str(p)})
        except Exception as exc:  # pragma: no cover
            out.append({"name": fname, "caption": caption, "error": str(exc)})
    return out