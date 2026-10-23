#!/usr/bin/env python
"""Plot u+ vs y+ for B26 channel cases (semilog) with DNS + log-law reference."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KAPPA = 0.41
B_LOG = 5.0
HERE = Path(__file__).resolve().parent


def load_dns() -> tuple[np.ndarray, np.ndarray]:
    yp, up = [], []
    with open(HERE / "dns_ref" / "mkm180.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            yp.append(float(r["y_plus"]))
            up.append(float(r["u_plus"]))
    return np.array(yp), np.array(up)


def main() -> None:
    cases = json.loads((HERE / "result.json").read_text(encoding="utf-8"))["results"]
    dns_yp, dns_up = load_dns()
    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)

    yp_log = np.logspace(0, 2.6, 200)
    ax.plot(
        yp_log,
        np.log(yp_log) / KAPPA + B_LOG,
        "--",
        color="tab:gray",
        label=f"log law u+=(1/{KAPPA})ln(y+)+{B_LOG}",
    )
    ax.plot(yp_log, yp_log, ":", color="tab:gray", label="u+=y+")
    ax.plot(dns_yp, dns_up, "-", color="black", lw=1.6, label="DNS MKM1999 Re_tau=178.12")

    colors = ["tab:red", "tab:blue", "tab:green"]
    for i, res in enumerate(cases):
        prof = res["profile"]
        yp = np.array(prof["y_plus"])
        up = np.array(prof["u_plus"])
        label = f"sim nx{res['config']['nx']}x{res['config']['ny']}"
        if "run_dir" in res:
            label = Path(res["run_dir"]).name
        ax.plot(yp, up, "o-", ms=4, lw=1.2, color=colors[i % len(colors)], label=label)
        ax.axvspan(30, 100, color="orange", alpha=0.12)
        err = res["errors"]["rms_vs_dns"]
        ax.text(
            0.02,
            0.98 - 0.05 * i,
            f"{label}: RMS(DNS)={err:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color=colors[i % len(colors)],
        )

    ax.set_xscale("log")
    ax.set_xlim(1, 400)
    ax.set_ylim(0, 28)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$u^+$")
    ax.set_title("B26: turbulent channel Re_tau=180, u+ profile")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(HERE / "uplus_profile.png", dpi=150)
    print("saved", HERE / "uplus_profile.png")


if __name__ == "__main__":
    sys.exit(main())
