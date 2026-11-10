"""p247w1-N10 辅助证据：伪滑移闭式（tau/g 扫描）、Couette、剖面残差。

产出 artifacts/p247w1_n10_evidence.json（报告数字全部从这里注入，零手抄）。
用法：PYTHONPATH=src python scripts/p247w1_n10_evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorlbm.boundary.partial_slip_second_order import (
    channel_poiseuille_2nd,
    couette_slip,
    probe_kappa_table,
    probe_parabola_residual,
    slip_law_theory,
    spurious_slip_coefficient,
)
from tensorlbm.boundary.partial_slip_wall import analytic_plane_profile, invert_slip_length, nu_from_tau

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
G = 1e-5


def main() -> dict:
    ART.mkdir(exist_ok=True)

    # 1) kappa 闭式 vs 模拟（tau 扫描 × g 扫描）
    kap = probe_kappa_table(ny=17, steps=200000, tol=1e-12)

    # 2) 剖面残差结构（σ=0，H=7.5/15.5）：常数平移的证据
    prof = []
    for ny in (17, 33):
        r = channel_poiseuille_2nd(0.0, ny=ny, tau=0.8, g=G, steps=200000, tol=1e-12,
                                   correction=False)
        p = probe_parabola_residual(r)
        p["H"] = (ny - 2) / 2.0
        prof.append(p)

    # 3) Couette：切应力与滑移长度
    cou = []
    for sigma in (0.0, 0.1, 0.25, 0.5, 0.75):
        c = couette_slip(sigma, ny=33, tau=0.8, u_wall=1e-3, steps=40000, tol=1e-12)
        half = (33 - 2) / 2.0
        ana = c["nu"] * 1e-3 / (half + c["b_inverted"])
        c.pop("coord")
        c.pop("u")
        c["shear_stress_analytic"] = float(ana)
        c["shear_stress_rel_err"] = float((c["shear_stress"] - ana) / ana)
        cou.append(c)

    # 4) 剖面逐点误差（σ=0.2，H=15.5，修正后）
    r = channel_poiseuille_2nd(0.2, ny=33, tau=0.8, g=G, steps=300000, tol=1e-12)
    inv = invert_slip_length(r)
    ana = analytic_plane_profile(r.coord, r.length, inv.get("nu", nu_from_tau(0.8)), G, b=0.0)
    got = np.asarray(r.u) - np.mean(r.u)
    ref = ana - np.mean(ana)
    prof_err = {
        "sigma": 0.2,
        "H": (33 - 2) / 2.0,
        "max_rel_err_vs_flipped_analytic": float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))),
        "b": float(inv["b"]),
        "b_theory": float(slip_law_theory(0.2, nu_from_tau(0.8))),
    }

    out = {
        "node": "p247w1-N10",
        "source": "m7963-c1",
        "kappa_table": kap,
        "kappa_at_tau0p8": spurious_slip_coefficient(0.8),
        "parabola_residual": prof,
        "couette": cou,
        "profile_error": prof_err,
        "g": G,
    }
    (ART / "p247w1_n10_evidence.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps({"kappa_max_rel_dev": o["kappa_table"]["max_rel_dev"],
                      "couette_rel_err": [c["shear_stress_rel_err"] for c in o["couette"]],
                      "profile_err": o["profile_error"]}, indent=2))