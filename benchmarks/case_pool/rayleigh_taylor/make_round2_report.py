"""Round-2 report builder: acceptance table + figure, numbers injected, no hand copy."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_case as rc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = [REPO / "runs" / "m18081" / s for s in ("suite1", "suite2", "suite3")]


def slope(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def collect():
    rows = []
    for root in RUNS:
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            cj, hj = d / "case.json", d / "history.csv"
            if not (cj.exists() and hj.exists()):
                continue
            case = json.loads(cj.read_text()); spec = case["spec"]
            rs = list(csv.DictReader(hj.open()))
            if "h_bubble_zc" in rs[0]:          # round-2 runs: h_bubble = column moment
                hb, hs = "h_bubble", "h_spike"
            elif "h_bubble_col" in rs[0]:       # pre-swap runs: column moment under *_col
                hb, hs = "h_bubble_col", "h_spike_col"
            else:
                hb, hs = "h_bubble", "h_spike"
            win = [r for r in rs if rc.TAU_LO <= float(r["tau"]) <= rc.TAU_HI]
            unit = math.sqrt(spec["atwood"] * spec["g"] / spec["lam"])
            ref = rc.boussinesq_velocity(spec["atwood"], spec["g"], spec["lam"])
            if len(win) >= 5:
                ts = [float(r["tau"]) for r in win]
                ub = slope(ts, [float(r[hb]) for r in win]) * unit
                us = slope(ts, [float(r[hs]) for r in win]) * unit
            else:
                ub = us = float("nan")
            rows.append(dict(name=case["name"], scheme=spec.get("phase_scheme", "bgk"),
                             lam=spec["lam"], A=spec["atwood"], g=spec["g"], n=len(rs),
                             u_b=ub, u_s=us, ref=ref,
                             err_b=100 * (ub / ref - 1), err_s=100 * (us / ref - 1),
                             sigma=case.get("sigma_sim"), sigma_err=case.get("sigma_rel_error_vs_viscous_pct"),
                             ma=case.get("ma_peak"), phi_max=(case.get("phi_range_peak") or [None, None])[1],
                             hist=[(float(r["tau"]), float(r[hb])) for r in rs],
                             wall=case.get("wall_seconds")))
    return rows


def main():
    rows = collect()
    rows.sort(key=lambda r: (r["scheme"], -r["A"], r["lam"], r["g"]))
    out = HERE / "artifacts" / "round2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in rows:
        if r["scheme"] == "allen_cahn":
            ts = [h[0] for h in r["hist"]]; hs = [h[1] for h in r["hist"]]
            lab = f"A={r['A']:.2f} lam={r['lam']} g={r['g']:g}"
            ax[0].plot(ts, hs, label=lab)
    ax[0].axvspan(rc.TAU_LO, rc.TAU_HI, color="0.85", label="terminal window")
    ax[0].set_xlabel(r"$\tau = t\sqrt{Ag/\lambda}$"); ax[0].set_ylabel(r"$h_{bubble}/\Delta$ (cells)")
    ax[0].set_title("bubble front (column moment), Allen-Cahn interface")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    ac = [r for r in rows if r["scheme"] == "allen_cahn"]
    labels = [f"A={r['A']:.2f}\nlam={r['lam']}" for r in ac]
    x = range(len(ac))
    ax[1].bar([i - .2 for i in x], [r["u_b"] for r in ac], .4, label="u_bubble sim")
    ax[1].bar([i + .2 for i in x], [r["ref"] for r in ac], .4, label="ref sqrt(2Ag/(Ck)), C=1")
    for i, r in enumerate(ac):
        ax[1].text(i, max(r["u_b"], r["ref"]) * 1.02, f"{r['err_b']:+.1f}%", ha="center", fontsize=8)
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("terminal velocity (lattice)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_title("terminal velocity vs potential-flow reference (10% gate)")
    fig.tight_layout()
    figs = REPO / "figures"; figs.mkdir(exist_ok=True)
    fig.savefig(figs / "CASE-RAYLEIGH-TAYLOR_round2.png", dpi=140)
    print("figure written")

    lines = []
    for r in rows:
        lines.append(f"| {r['name']} | {r['scheme']} | {r['A']:.3f} | {r['lam']} | {r['g']:g} | "
                     f"{r['u_b']:.5f} | {r['u_s']:.5f} | {r['ref']:.5f} | {r['err_b']:+.1f}% | {r['err_s']:+.1f}% | "
                     f"{r['sigma']:.5f} | {r['sigma_err']:+.1f}% | {r['ma']:.3f} | {r['phi_max']} |")
    table = "\n".join(lines)
    print(table)
    (out / "acceptance_table.md").write_text(
        "| case | scheme | A | lam | g | u_b | u_s | ref | err_b | err_s | sigma | sigma_err | Ma_peak | phi_max |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + table + "\n")
    return rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
