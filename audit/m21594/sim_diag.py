#!/usr/bin/env python3
"""m21594 诊断腿：用 m21558 交付的**同一个** two_component_sc 求解器做三组对照。

D2 稳态检验      : 交付配置 2000 / 20000 步，看 eps_inf 是否随步数下降到 15% 内
                   （直接检验 STATUS.md 的「Issue 2: 需要 10000+ 步」根因假设）
D3 单组分对照    : G=0、组分 B 置真空、ρ_A=2、τ=0.8 → 该求解器能否解出单组分 Poiseuille
                   （把「有效黏度需标定」与「强迫/边界格式本身错」分开）
D4 质量/壁格账   : 两组分总质量随时间漂移 + 壁邻格密度峰值

只读快照来源: audit/m21594/snapshot（sha256 见 evidence.json E1）
用法: python3 audit/m21594/sim_diag.py [--steps 2000,20000]
"""
from __future__ import annotations

import importlib.util
import os
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D3_STEPS = int(os.environ.get("D3_STEPS", "20000"))
SNAP = HERE / "snapshot"
NY, NX, H, G_FORCE = 32, 4, 30.0, 1e-4
NU_A = (1.0 / 3.0) * (0.8 - 0.5)
NU_B = (1.0 / 3.0) * (1.2 - 0.5)
MARGIN = 3


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass 需要 __module__ 可解析
    spec.loader.exec_module(mod)
    return mod


tcs = load("_txc_two_component_sc", SNAP / "src/tensorlbm/verification/layered_poiseuille/two_component_sc.py")
ana = load("_txc_analytic", SNAP / "src/tensorlbm/verification/layered_poiseuille/analytic.py")


def run_case(steps: int, G: float = 2.0, single: bool = False) -> dict:
    import torch
    cfg = tcs.TwoComponentSCConfig(ny=NY, nx=NX, tau_a=0.8, tau_b=1.2, G=G,
                                   g_x=G_FORCE, g_y=0.0, steps=steps, forcing="edm")
    s = tcs.TwoComponentSCSolver(cfg)
    if single:
        z = torch.full((NY, NX), 1e-8, dtype=torch.float32)
        s.f_b = tcs.equilibrium(z, torch.zeros_like(z), torch.zeros_like(z))
    trace = []
    t0 = time.time()
    for k in range(1, steps + 1):
        s.step()
        if k % 500 == 0 or k == steps:
            m = s.get_macroscopic()
            trace.append({"step": k, "u_max": float(m["ux_total"].abs().max()),
                          "rho_wall_max": float(m["rho_total"][1].max()),
                          "mass_a": float(m["rho_a"].sum()), "mass_b": float(m["rho_b"].sum())})
    p = s.get_x_averaged_profile()
    return {"steps": steps, "G": G, "single": single, "wall_s": round(time.time() - t0, 1),
            "y": p["y"].tolist(), "ux": p["ux_total"].tolist(),
            "rho_a": p["rho_a"].tolist(), "rho_b": p["rho_b"].tolist(), "trace": trace}


def metrics(r: dict, ref) -> dict:
    y = np.asarray(r["y"])
    ux = np.asarray(r["ux"])
    ua = ref.u(y)
    sc = float(np.max(np.abs(ua[MARGIN:-MARGIN])))
    return {
        "u_peak_num": float(np.max(ux)),
        "u_peak_ana": float(np.max(ua)),
        "u_peak_err": float(abs(np.max(ux) - np.max(ua)) / np.max(ua)),
        "eps_inf": float(np.max(np.abs(ux[MARGIN:-MARGIN] - ua[MARGIN:-MARGIN])) / sc),
        "eps_l2": float(np.linalg.norm(ux[MARGIN:-MARGIN] - ua[MARGIN:-MARGIN])
                        / np.linalg.norm(ua[MARGIN:-MARGIN])),
        "u_max_over_g_t": float(np.max(ux) / (G_FORCE * r["steps"])),
        "interior_std": float(np.std(ux[2:-2])),
        "wall_rho_peak": float(max(np.max(np.asarray(r["rho_a"])), np.max(np.asarray(r["rho_b"])))),
    }


def main(steps_list):
    ref_layered = ana.solve_layered(height=H, interfaces=[15.0], nus=[NU_A, NU_B], g=G_FORCE)
    ref_single = ana.solve_layered(height=H, interfaces=[], nus=[NU_A], g=G_FORCE)
    out = {"meta": {"nu_a": NU_A, "nu_b": NU_B, "H": H, "g": G_FORCE,
                    "diffusion_time_H2_over_nu": H * H / NU_A,
                    "solve_layered_u_peak_layered": float(ref_layered.u_peak()),
                    "solve_layered_u_peak_single": float(ref_single.u_peak())},
           "D2_steady_state": {}, "D3_single_component_control": {}}
    for st in steps_list:
        r = run_case(st)
        d = metrics(r, ref_layered)
        d["trace"] = r["trace"]
        d["wall_s"] = r["wall_s"]
        d["profile"] = {"y": r["y"], "ux": r["ux"], "rho_a": r["rho_a"], "rho_b": r["rho_b"]}
        out["D2_steady_state"][str(st)] = d
        print(f"[D2] {st} steps: u_peak_err={d['u_peak_err']:.2%} eps_inf={d['eps_inf']:.2%} "
              f"u_max/g_t={d['u_max_over_g_t']:.3f} wall_rho={d['wall_rho_peak']:.1f} ({r['wall_s']}s)")
    r = run_case(D3_STEPS, G=0.0, single=True)
    d = metrics(r, ref_single)
    d["trace"] = r["trace"]
    d["wall_s"] = r["wall_s"]
    d["profile"] = {"y": r["y"], "ux": r["ux"], "rho_a": r["rho_a"], "rho_b": r["rho_b"]}
    out["D3_single_component_control"] = d
    print(f"[D3] single-comp G=0 {D3_STEPS} steps: u_peak_err={d['u_peak_err']:.2%} "
          f"eps_inf={d['eps_inf']:.2%} u_max/g_t={d['u_max_over_g_t']:.3f} ({r['wall_s']}s)")
    out["pass_15pct"] = {
        "D2_" + k: bool(v["eps_inf"] < 0.15) for k, v in out["D2_steady_state"].items()}
    out["pass_15pct"]["D3_single_component"] = bool(d["eps_inf"] < 0.15)
    (HERE / "sim_diagnostics.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "sim_diagnostics.json")


if __name__ == "__main__":
    steps = [2000, 20000]
    if len(sys.argv) > 2 and sys.argv[1] == "--steps":
        steps = [int(x) for x in sys.argv[2].split(",")]
    main(steps)