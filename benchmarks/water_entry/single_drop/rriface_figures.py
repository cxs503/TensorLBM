"""DE-X-RELX3-RRIFACE: report figures (depth-time vs experiment, cavity
geometry time series).  All data read from runs/<arm>/result.json and
runs/relx3_rriface/cav_series.json; nothing typed by hand."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_PEAK_D = 1.857434
EXP_PEAK_PC_MS = 10.39446


def load(tree: Path, arm: str) -> dict:
    return json.loads((tree / "runs" / arm / "result.json").read_text())


def pc_series(res: dict):
    h = res["history"]
    contact = res.get("contact_step")
    dt = h[1]["time_ms"] - h[0]["time_ms"]
    t = [r["time_ms"] - contact * dt for r in h]
    d = [r["depth_over_D"] for r in h]
    return t, d


def main() -> None:
    tree = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/relx3_rriface")
    out.mkdir(parents=True, exist_ok=True)

    ctrl = load(tree, "relx3_rriface_ctrl")
    arm = load(tree, "relx3_rriface")
    tc, dc = pc_series(ctrl)
    ta, da = pc_series(arm)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.plot(tc, dc, label="ctrl (BGK, this work)", lw=1.6)
    ax.plot(ta, da, label="rr_iface (RR on INTERFACE cells only)", lw=1.6)
    ax.axhline(EXP_PEAK_D, color="k", ls=":", lw=1)
    ax.axvline(EXP_PEAK_PC_MS, color="k", ls=":", lw=1)
    ax.plot([EXP_PEAK_PC_MS], [EXP_PEAK_D], "k*", ms=14,
            label=f"experiment {EXP_PEAK_D}D @ {EXP_PEAK_PC_MS} ms (pc)")
    pkc = max(dc)
    pka = max(da)
    ic = dc.index(pkc)
    ia = da.index(pka)
    ax.plot([tc[ic]], [pkc], "o", ms=5)
    ax.plot([ta[ia]], [pka], "s", ms=5)
    ax.annotate(f"ctrl {pkc:.4f}D", (tc[ic], pkc), textcoords="offset points", xytext=(8, -12))
    ax.annotate(f"rr_iface {pka:.4f}D", (ta[ia], pka), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("time since contact (ms)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("Single-drop water entry (Santini case-3, cells=8): interface-local RR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_depth_time.png")
    plt.close(fig)

    cav_path = tree / "runs" / "relx3_rriface" / "cav_series.json"
    if cav_path.exists():
        rows = json.loads(cav_path.read_text())["rows"]
        t = [r["time_ms"] for r in rows]
        area = [r["cav_area_cells"] for r in rows]
        width = [r["cav_width_cells"] for r in rows]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), dpi=140, sharex=True)
        a1.plot(t, area, lw=1.4)
        a1.set_ylabel("cavity area (cells, mid-plane)")
        a1.grid(alpha=0.3)
        a2.plot(t, width, lw=1.4)
        a2.set_ylabel("cavity width (cells)")
        a2.set_xlabel("time (ms, absolute)")
        a2.set_title("rr_iface arm: mid-plane cavity cross-section (geometry attribution)")
        fig.tight_layout()
        fig.savefig(out / "fig_cavity_geometry.png")
        plt.close(fig)
        print("wrote fig_cavity_geometry.png")
    else:
        print("cav_series.json missing; geometry figure skipped")

    for name in ("fig_depth_time.png",):
        p = out / name
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            (out / (name + ".b64")).write_text(b64)
            print(name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    import sys

    main()
