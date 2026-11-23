"""DE-X-GAS1-GASVISC: depth-time comparison figure (ctrl vs arm vs experiment).

Reads the two run JSONs and the locked experimental reference, plots the
post-contact crater-depth curve for both arms against the experimental
markers, and saves a PNG.  All numbers come from the files (zero hand-copy).
"""

import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import reference  # noqa: E402

CASE = json.loads((HERE / "protocol.json").read_text())


def load_history(path):
    run = json.loads(Path(path).read_text())
    rows = run["history"]
    t0 = None
    ts, hs = [], []
    for r in rows:
        if r.get("depth_over_D") is None or r.get("time_ms") is None:
            continue
        if t0 is None:
            t0 = r["time_ms"]
        ts.append(r["time_ms"] - t0)
        hs.append(r["depth_over_D"])
    return np.array(ts), np.array(hs), run


def main():
    ctrl_path, arm_path, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    t_ref, h_ref = reference(CASE)
    tc, hc, rc = load_history(ctrl_path)
    ta, ha, ra = load_history(arm_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(
        tc, hc, "-", color="tab:blue", lw=1.6, label=f"ctrl (gas_nu_ratio={rc.get('gas_nu_ratio')})"
    )
    ax.plot(
        ta, ha, "--", color="tab:red", lw=1.6, label=f"arm (gas_nu_ratio={ra.get('gas_nu_ratio')})"
    )
    ax.plot(
        t_ref, h_ref, "o", color="k", ms=5, mfc="none", mew=1.4, label="experiment (Santini case 3)"
    )
    # experimental peak annotation
    ipk = int(np.argmax(h_ref))
    ax.annotate(
        f"exp peak {h_ref[ipk]:.4f}D @ {t_ref[ipk]:.3f}ms",
        xy=(t_ref[ipk], h_ref[ipk]),
        xytext=(t_ref[ipk] + 1.5, h_ref[ipk] + 0.15),
        arrowprops=dict(arrowstyle="->", color="k"),
        fontsize=8,
    )
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("crater depth (D)")
    ax.set_title("DE-X-GAS1-GASVISC: gas-viscosity arm vs ctrl (cells=8)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"saved {out_png}")
    if len(sys.argv) > 4 and sys.argv[4] == "--b64":
        print(base64.b64encode(Path(out_png).read_bytes()).decode())


if __name__ == "__main__":
    main()
