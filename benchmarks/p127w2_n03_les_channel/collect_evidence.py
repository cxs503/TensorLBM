#!/usr/bin/env python3
"""Collect the raw run logs of the P127W2-N03 campaign into a machine-readable record.

The campaign hit the host's compute wall: several runs were stopped by the
solver's own divergence guard (``FloatingPointError`` in ``run_les``), so they
never wrote a result JSON.  Their *logs* are still first-hand evidence — the
per-250-step trace of Mach number, bulk velocity and resolved TKE — and this
script turns them into ``artifacts/campaign/experiments.json`` plus a figure, so
the report can cite numbers instead of prose.

Run register (label -> launch parameters) is the only hand-entered part; every
number below it is parsed from the log files.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "artifacts" / "campaign"
LOG_DIR = ROOT / "logs"

# label -> (Re_tau, h, cs, u_tau, strain_source, wall_model, steps, warmup)
RUNS = {
    "R180h08": (180, 8, 0.12, 0.01, "fneq", "spalding", 4000, 800),
    "R180h08nowm": (180, 8, 0.12, 0.01, "fneq", "none", 4000, 800),
    "R180h10": (180, 10, 0.12, 0.01, "fneq", "spalding", 4000, 800),
    "R180h12": (180, 12, 0.12, 0.01, "fneq", "spalding", 4000, 800),
    "R395h12": (395, 12, 0.12, 0.01, "fneq", "spalding", 4000, 800),
    "R590h12": (590, 12, 0.12, 0.01, "fneq", "spalding", 4000, 800),
    "R180h08vg": (180, 8, 0.12, 0.01, "velocity_gradient", "none", 4000, 800),
    "V180h08": (180, 8, 0.12, 0.01, "velocity_gradient", "none", 2500, 800),
    "V180h10": (180, 10, 0.12, 0.01, "velocity_gradient", "none", 2500, 800),
    "V395h08": (395, 8, 0.12, 0.01, "velocity_gradient", "none", 2000, 700),
    "V590h08": (590, 8, 0.12, 0.01, "velocity_gradient", "none", 2000, 700),
    "V180h08lo": (180, 8, 0.12, 0.005, "velocity_gradient", "none", 3500, 1200),
    "case_re180_L1": (180, 16, 0.12, 0.02, "fneq", "none", 14000, 0),
    "case_re180_L2": (180, 16, 0.12, 0.02, "fneq", "none", 16000, 0),
    "case_re180_L3": (180, 24, 0.12, 0.02, "fneq", "none", 22000, 0),
    "case_re395_L1": (395, 32, 0.12, 0.02, "fneq", "none", 14000, 0),
    "case_re590_L1": (590, 32, 0.12, 0.02, "fneq", "none", 14000, 0),
}

LINE = re.compile(
    r"\[(?P<label>[^\]]+)\]\s+(?P<step>\d+)/(?P<total>\d+)\s+Ma=(?P<ma>[0-9.eE+-]+)\s+"
    r"u_bulk=(?P<ub>[0-9.eE+-]+)\s+tke=(?P<tke>[0-9.eE+-]+)\s+(?P<rate>[0-9.eE+-]+)\s+steps/s"
)
DONE = re.compile(r"\[(?P<label>[^\]]+)\]\s+done in (?P<secs>[0-9.]+)s")
DIVERGED = re.compile(r"divergence at step (?P<step>\d+)")


def parse_log(path: Path) -> dict:
    history = []
    diverged_at = None
    done = False
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            history.append(
                {
                    "step": int(m.group("step")),
                    "ma": float(m.group("ma")),
                    "u_bulk": float(m.group("ub")),
                    "tke": float(m.group("tke")),
                }
            )
        d = DIVERGED.search(line)
        if d:
            diverged_at = int(d.group("step"))
        if DONE.search(line):
            done = True
    return {"history": history, "diverged_at_step": diverged_at, "completed": done}


def main() -> int:
    records = []
    for label, (re_tau, h, cs, u_tau, strain, wall_model, steps, warmup) in RUNS.items():
        path = LOG_DIR / f"camp_{label}.log"
        if not path.exists():
            path = LOG_DIR / f"{label}.log"
        if not path.exists():
            continue
        parsed = parse_log(path)
        hist = parsed["history"]
        tke = [h_["tke"] for h_ in hist]
        t_e = h / u_tau if u_tau > 0 else float("nan")
        records.append(
            {
                "label": label,
                "log": str(path.relative_to(ROOT)),
                "re_tau": re_tau,
                "h": h,
                "dy_plus": re_tau / h,
                "cs": cs,
                "u_tau": u_tau,
                "strain_source": strain,
                "wall_model": wall_model,
                "planned_steps": steps,
                "warmup_steps": warmup,
                "eddy_turnover_steps": t_e,
                "last_step_reached": hist[-1]["step"] if hist else 0,
                "completed": parsed["completed"],
                "diverged_at_step": parsed["diverged_at_step"],
                "max_mach_reached": max((h_["ma"] for h_ in hist), default=None),
                "tke_first": tke[0] if tke else None,
                "tke_last": tke[-1] if tke else None,
                "tke_ratio_last_over_first": (tke[-1] / tke[0]) if tke and tke[0] > 0 else None,
                "eddy_turnovers_reached": (hist[-1]["step"] / t_e) if hist and t_e > 0 else None,
                "history": hist,
            }
        )

    payload = {
        "schema": "p127w2-n03-experiment-register-v1",
        "note": (
            "Every number is parsed from the run logs in logs/; the register maps labels to the "
            "launch parameters. Runs stopped by the solver's divergence guard never wrote a result "
            "JSON, so their record is the log trace."
        ),
        "runs": records,
        "summary": {
            "n_runs": len(records),
            "n_completed": sum(1 for r in records if r["completed"]),
            "n_diverged": sum(1 for r in records if r["diverged_at_step"]),
            "labels_diverged": [r["label"] for r in records if r["diverged_at_step"]],
            "labels_decayed": [
                r["label"] for r in records
                if r["tke_ratio_last_over_first"] is not None and r["tke_ratio_last_over_first"] < 0.1
            ],
        },
    }
    (CAMPAIGN / "experiments.json").write_text(json.dumps(payload, indent=1))

    # ---- figure: resolved TKE against eddy turnovers, all runs -------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = CAMPAIGN / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for rec in records:
        if not rec["history"]:
            continue
        t = [h_["step"] / rec["eddy_turnover_steps"] for h_ in rec["history"]]
        tke = [max(h_["tke"], 1e-16) for h_ in rec["history"]]
        style = "-" if rec["strain_source"] == "fneq" else "--"
        ax.semilogy(
            t, tke, style, lw=1.1, marker="", 
            label=f"{rec['label']} (Re{rec['re_tau']}, dy+={rec['dy_plus']:.0f}, {rec['strain_source']})",
        )
        if rec["diverged_at_step"]:
            ax.axvline(rec["diverged_at_step"] / rec["eddy_turnover_steps"], color="tab:red",
                       lw=0.6, alpha=0.4)
    ax.set_xlabel(r"$t\,u_\tau/h$  (eddy turnovers)")
    ax.set_ylabel("resolved TKE (lattice units)")
    ax.set_title("Resolved turbulence: fneq strain decays, velocity-gradient strain bursts then diverges")
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figdir / "fig_all_runs_tke.png", dpi=140)
    plt.close(fig)

    print(json.dumps(payload["summary"], indent=1))
    for rec in records:
        print(
            "{:<14} Re={:>3} h={:>2} dy+={:5.1f} {:<17} steps={:>6}/{:<6} T_e={:5.1f} "
            "Ma_max={:6.2f} tke {:.2e} -> {:.2e} ({})".format(
                rec["label"], rec["re_tau"], rec["h"], rec["dy_plus"], rec["strain_source"],
                rec["last_step_reached"], rec["planned_steps"],
                rec["eddy_turnovers_reached"] or float("nan"),
                rec["max_mach_reached"] or float("nan"),
                rec["tke_first"] or float("nan"), rec["tke_last"] or float("nan"),
                "DIVERGED@" + str(rec["diverged_at_step"]) if rec["diverged_at_step"] else
                ("completed" if rec["completed"] else "stopped"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())