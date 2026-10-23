#!/usr/bin/env python3
# NOTE: operates on the raw per-step trace files (formal/case_D*.json)
# retained on the server staging tree, not the repo summary files.
"""W8-B 方柱 Re=100 — verify.py：从 case_*.json 原始迹线独立复算全部统计与判决。

独立性声明：只用 numpy + 标准库重算（不 import torch/tensorlbm/run.py），
几何（掩码/链接数）用逐胞双重循环暴力复算（与 run.py 的向量化实现独立）。

复算项：
 1. 几何：nx/ny、方柱边界 x0..x1/y0..y1、固体胞数（=D²）、流体→固体链接数
    （暴力逐向计数；解析式 4D + 4(2D−1) = 12D−4 交叉核对）
 2. 每列（wet/pair/mem）：Cd 窗均值、12 分块均值（std/min/max）、St 谱峰
    （Hann+抛物线）、St 过零（滞回阈值 0.25max）、cl_amp（max|Cl−mean|）
 3. 误差（对锁定参考 Cd_ref=1.4785/St_ref=0.1456）+ 敏感性参考
 4. A/B（wet−mem）与闭包（pair−mem 均值）
 5. 主门与单调条款（prereg §2 + 修订 1：主列 = pair）
 6. 复现门：mem 列 vs 旧档 D32/D48

输出：逐项 OK/FAIL；全部通过 exit 0。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REF_CD = 1.4785
REF_ST = 0.1456
SENS = {
    "okajima_exp": {"cd": 1.59, "st": 0.141},
    "norberg_exp": {"st": 0.143},
    "przulj_interp_b2.5pct": {"cd": 1.43, "st": 0.142},
    "old_archive_okajima": {"cd": 1.6, "st": 0.14},
}
OLD_ARCH = {32: (1.4816, 0.1482), 48: (1.4844, 0.1480)}
DOMAIN_D = 40.0
SQUARE_X_D = 10.0
N_BLOCKS = 12
# D2Q9 速度集（x, y）：静止 + 8 方向（与库 d2q9.C 同构，此处独立硬编码）
VEL = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "OK  " if ok else "FAIL"
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def geometry(D: int) -> dict:
    nx = int(DOMAIN_D * D)
    ny = nx
    cx = SQUARE_X_D * D
    cy = ny / 2.0
    x0 = int(round(cx - D / 2.0))
    x1 = x0 + int(round(D)) - 1
    y0 = int(round(cy - D / 2.0))
    y1 = y0 + int(round(D)) - 1
    solid = np.zeros((ny, nx), dtype=bool)
    solid[y0 : y1 + 1, x0 : x1 + 1] = True
    n_solid = int(solid.sum())
    n_links = 0
    for q, (dx, dy) in enumerate(VEL):
        if dx == 0 and dy == 0:
            continue
        # 链接 (x_s 固体, x_s − c_q 流体)
        ys, xs = np.where(solid)
        n_links += int((~solid[ys - dy, xs - dx]).sum())
    return {
        "nx": nx,
        "ny": ny,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "n_solid": n_solid,
        "n_links": n_links,
        "n_links_formula": 12 * D - 4,
    }


def recompute(cd: np.ndarray, cl: np.ndarray, w0: int, D: int, u_in: float) -> dict:
    cd_w = cd[w0:]
    cl_w_full = cl[w0:]
    cd_mean = float(cd_w.mean())
    blocks = [float(b.mean()) for b in np.array_split(cd_w, N_BLOCKS)]
    cl_w = cl_w_full - cl_w_full.mean()

    spec = np.abs(np.fft.rfft(cl_w * np.hanning(cl_w.size)))
    k = int(np.argmax(spec[1:])) + 1
    delta = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-30:
            delta = float(0.5 * (a - c) / den)
    st = float((k + delta) / cl_w.size * D / u_in)

    thr = 0.25 * float(np.abs(cl_w).max())
    sig = np.where(cl_w >= thr, 1, np.where(cl_w <= -thr, -1, 0))
    idx = np.where(sig != 0, np.arange(sig.size), -1)
    last = np.maximum.accumulate(idx)
    state = np.where(last >= 0, sig[np.maximum(last, 0)], 0)
    crossings = np.where((state[:-1] < 0) & (state[1:] > 0))[0] + 1
    st_cross = float(D / (u_in * np.median(np.diff(crossings)))) if len(crossings) >= 3 else None

    return {
        "cd": cd_mean,
        "cd_block_std": float(np.std(blocks)),
        "cd_block_min": min(blocks),
        "cd_block_max": max(blocks),
        "st": st,
        "st_crossing": st_cross,
        "cl_amp": float(np.abs(cl_w).max()),
    }


def main() -> None:
    stage = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    cases = {}
    for p in sorted(stage.glob("case_D*.json")):
        d = json.loads(p.read_text())
        cases[d["D"]] = d
    check("case files found", len(cases) >= 2, f"D={sorted(cases)}")

    for D in sorted(cases):
        d = cases[D]
        g = geometry(D)
        check(
            f"D{D}: domain nx/ny", d["nx"] == g["nx"] and d["ny"] == g["ny"], f"{d['nx']}x{d['ny']}"
        )
        check(
            f"D{D}: solid cells = D^2",
            d["force_columns"]["n_solid_cells"] == g["n_solid"] == D * D,
            f"{d['force_columns']['n_solid_cells']} vs {D * D}",
        )
        check(
            f"D{D}: wet links = brute force = 12D-4",
            d["force_columns"]["n_links_wet"] == g["n_links"] == g["n_links_formula"],
            f"json={d['force_columns']['n_links_wet']} brute={g['n_links']} formula={g['n_links_formula']}",
        )

        w0 = d["analyze_from"]
        check(f"D{D}: analyze_from = warmup50%", w0 == int(0.5 * d["steps"]))
        u_in = d["u_in"]
        rec = {}
        for col in ("wet", "pair", "mem"):
            cd = np.asarray(d["traces"][f"cd_{col}"], dtype=np.float64)
            cl = np.asarray(d["traces"][f"cl_{col}"], dtype=np.float64)
            check(f"D{D}: trace lengths {col}", len(cd) == d["steps"] and len(cl) == d["steps"])
            r = recompute(cd, cl, w0, D, u_in)
            rec[col] = r
            s = d[col]
            ok = (
                abs(r["cd"] - s["cd"]) < 5e-4
                and abs(r["st"] - s["st"]) < 5e-4
                and abs(r["cl_amp"] - s["cl_amp"]) < 5e-4
                and abs(r["cd_block_std"] - s["cd_block_std"]) < 5e-4
            )
            cross_ok = (r["st_crossing"] is None and s["st_crossing"] is None) or (
                r["st_crossing"] is not None
                and s["st_crossing"] is not None
                and abs(r["st_crossing"] - s["st_crossing"]) < 5e-4
            )
            check(
                f"D{D}: recompute stats {col}",
                ok and cross_ok,
                f"Cd {r['cd']:.4f}/{s['cd']} St {r['st']:.4f}/{s['st']} "
                f"cross {r['st_crossing']}/{s['st_crossing']} amp {r['cl_amp']:.4f}/{s['cl_amp']:.4f}",
            )
            # 存储误差的自洽性：用存储（4 位舍入）值重算误差，须与存储误差一致
            ecd = (s["cd"] - REF_CD) / REF_CD * 100.0
            est = (s["st"] - REF_ST) / REF_ST * 100.0
            check(
                f"D{D}: stored err {col}",
                abs(ecd - s["err_cd_pct"]) < 0.01 and abs(est - s["err_st_pct"]) < 0.01,
                f"cd {ecd:+.2f}%/{s['err_cd_pct']}% st {est:+.2f}%/{s['err_st_pct']}%",
            )
            # 深层一致性：未舍入重算误差与存储误差差 ≤0.05pp（4 位舍入界内）
            ecd_raw = (r["cd"] - REF_CD) / REF_CD * 100.0
            est_raw = (r["st"] - REF_ST) / REF_ST * 100.0
            check(
                f"D{D}: raw-trace err consistency {col}",
                abs(ecd_raw - s["err_cd_pct"]) < 0.05 and abs(est_raw - s["err_st_pct"]) < 0.05,
                f"cd {ecd_raw:+.2f}% st {est_raw:+.2f}%",
            )

        # 闭包（prereg 修订 1）：pair ≈ mem（均值，稳态动量预算闭合）
        closure = (rec["pair"]["cd"] - rec["mem"]["cd"]) / rec["mem"]["cd"] * 100.0
        check(f"D{D}: closure pair-mem mean |d|<=0.5%", abs(closure) <= 0.5, f"{closure:+.3f}%")
        # A/B：wet 高于 mem 的量（诊断，非门）
        ab = (rec["wet"]["cd"] - rec["mem"]["cd"]) / rec["mem"]["cd"] * 100.0
        print(
            f"[info] D{D}: wet-mem Cd A/B = {ab:+.2f}%  "
            f"(wet cl_amp {rec['wet']['cl_amp']:.3f} vs mem {rec['mem']['cl_amp']:.3f})"
        )

        if D in OLD_ARCH:
            ocd, ost = OLD_ARCH[D]
            dcd = (rec["mem"]["cd"] - ocd) / ocd * 100.0
            dst = (rec["mem"]["st"] - ost) / ost * 100.0
            check(
                f"D{D}: repro vs old archive (mem) |d|<=0.5%",
                abs(dcd) <= 0.5 and abs(dst) <= 0.5,
                f"cd {dcd:+.3f}% st {dst:+.3f}%",
            )

    # ---- 主门与单调（prereg §2 + 修订 1：主列 = pair）----
    ds = sorted(cases)
    rec_all = {}
    for D in ds:
        d = cases[D]
        w0 = d["analyze_from"]
        rec_all[D] = recompute(
            np.asarray(d["traces"]["cd_pair"]),
            np.asarray(d["traces"]["cl_pair"]),
            w0,
            D,
            d["u_in"],
        )
    fine = rec_all[ds[-1]]
    ecd_f = (fine["cd"] - REF_CD) / REF_CD * 100.0
    est_f = (fine["st"] - REF_ST) / REF_ST * 100.0
    check(f"MAIN GATE finest D{ds[-1]} pair: |Cd err|<=3%", abs(ecd_f) <= 3.0, f"{ecd_f:+.2f}%")
    check(f"MAIN GATE finest D{ds[-1]} pair: |St err|<=3%", abs(est_f) <= 3.0, f"{est_f:+.2f}%")

    errs_cd = [abs(rec_all[D]["cd"] - REF_CD) / REF_CD for D in ds]
    errs_st = [abs(rec_all[D]["st"] - REF_ST) / REF_ST for D in ds]
    mono_cd = all(errs_cd[i + 1] <= errs_cd[i] + 1e-12 for i in range(len(errs_cd) - 1)) or all(
        e <= 0.01 for e in errs_cd
    )
    mono_st = all(errs_st[i + 1] <= errs_st[i] + 1e-12 for i in range(len(errs_st) - 1)) or all(
        e <= 0.01 for e in errs_st
    )
    check(
        "MONOTONE Cd (or floor<=1%)", bool(mono_cd), " -> ".join(f"{e * 100:.2f}%" for e in errs_cd)
    )
    check(
        "MONOTONE St (or floor<=1%)", bool(mono_st), " -> ".join(f"{e * 100:.2f}%" for e in errs_st)
    )

    # 敏感性旁列（finest, pair）
    print(f"[info] finest pair Cd={fine['cd']:.4f} St={fine['st']:.4f}; sensitivity:")
    for name, ref in SENS.items():
        for k, v in ref.items():
            simv = fine["cd"] if k == "cd" else fine["st"]
            print(f"[info]   {name} {k}: {(simv - v) / v * 100:+.2f}% vs {v}")

    print()
    if failures:
        print(f"VERIFY RESULT: {len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("VERIFY RESULT: ALL OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
