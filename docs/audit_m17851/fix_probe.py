#!/usr/bin/env python3
"""m17851 根因探针：2x2 分解 25% 误差是否由 halo 角点被覆盖导致。

做法（不改被审对象代码；运行时 monkeypatch）：
  1) 对照组：原样运行 m17720 的 run_decomposed（npx=npy=2）→ 复现 25.35%；
  2) 探针组：把 runners._halo_exchange 换成"角点感知"版本（x/y 面交换只写内部跨距，
     四个对角角块单独从对角邻居交换），其余代码/参数/步数完全不变；
  3) 若探针组误差落到 <1%，则缺陷定位成立：面交换整行/整列写回，用交换前快照
     覆盖了本轮 x 面已写好的角点（D2Q9 对角分布 f5..f8 依赖角点 halo）。

用法：python docs/audit_m17851/fix_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORK = Path("/root/tp-agent/work/m17720")
sys.path.insert(0, str(WORK / "src"))

import torch  # noqa: E402

from tensorlbm.xval_cartesian_mpi import runners as R  # noqa: E402
from tensorlbm.xval_cartesian_mpi.case_spec import PoiseuilleSpec  # noqa: E402
from tensorlbm.xval_cartesian_mpi.metrics import compute_metrics  # noqa: E402


def corner_aware_halo_exchange(decomp, sub_f):
    """面交换只动内部跨距 + 显式对角角块交换（与分解无关节序）。"""
    h = decomp.halo
    src = [f.clone() for f in sub_f]
    new_f = [f.clone() for f in sub_f]
    npx, npy = decomp.npx, decomp.npy

    for sd in decomp.subdomains:
        r = sd.rank
        ix, iy = sd.coords
        left, right, bottom, top = sd.neighbours
        # 面：只写内部跨距，避免覆盖角块
        if left is not None:
            new_f[left][:, h:-h, -h:] = src[r][:, h:-h, h:2 * h]
        if right is not None:
            new_f[right][:, h:-h, :h] = src[r][:, h:-h, -2 * h:-h]
        if bottom is not None:
            new_f[bottom][:, -h:, h:-h] = src[r][:, h:2 * h, h:-h]
        if top is not None:
            new_f[top][:, :h, h:-h] = src[r][:, -2 * h:-h, h:-h]
        # 角块：对角邻居
        diag = {
            (-1, -1): (iy * npx + (ix - 1) % npx if npx > 1 else r),
            (+1, -1): (iy * npx + (ix + 1) % npx if npx > 1 else r),
            (-1, +1): (iy * npx + (ix - 1) % npx if npx > 1 else r),
            (+1, +1): (iy * npx + (ix + 1) % npx if npx > 1 else r),
        }
        if npx > 1 and npy > 1:
            for dx in (-1, +1):
                for dy in (-1, +1):
                    if (dy < 0 and bottom is None) or (dy > 0 and top is None):
                        continue  # 物理壁方向无邻居（y 向非周期）
                    if (dx < 0 and left is None) or (dx > 0 and right is None):
                        continue
                    tgt_ix = (ix + dx) % npx
                    tgt_iy = (iy + dy) % npy if npy > 1 else iy
                    tgt = tgt_iy * npx + tgt_ix
                    if dx < 0:  # 目标在左：本块首个内部列 → 目标右侧 halo
                        scol = slice(h, 2 * h)
                        tcol = slice(-h, None)
                    else:
                        scol = slice(-2 * h, -h)
                        tcol = slice(0, h)
                    if dy < 0:  # 目标在下：本块首个内部行 → 目标上侧 halo
                        srow = slice(h, 2 * h)
                        trow = slice(-h, None)
                    else:
                        srow = slice(-2 * h, -h)
                        trow = slice(0, h)
                    new_f[tgt][:, trow, tcol] = src[r][:, srow, scol]
    return new_f


def main() -> None:
    spec = PoiseuilleSpec(nx=128, ny=32, steps=1000, sample_every=200, warmup_steps=2000,
                          npx=2, npy=2, halo=1)
    out = {}

    print("[control] 原样 halo exchange (2x2, 128x32, 1000 steps)")
    res0 = R.run_decomposed(spec, "/tmp/m17851_probe_control")
    out["control"] = compute_metrics(res0, spec)
    print("  umax_centreline_error_pct =", out["control"]["umax_centreline_error_pct"])

    R._halo_exchange = corner_aware_halo_exchange
    print("[probe] 角点感知 halo exchange（其余完全相同）")
    res1 = R.run_decomposed(spec, "/tmp/m17851_probe_cornerfix")
    out["corner_aware"] = compute_metrics(res1, spec)
    print("  umax_centreline_error_pct =", out["corner_aware"]["umax_centreline_error_pct"])

    out["delta_pct_points"] = (out["control"]["umax_centreline_error_pct"]
                               - out["corner_aware"]["umax_centreline_error_pct"])
    out["criterion_pct"] = 15.0
    out["probe_under_criterion"] = out["corner_aware"]["umax_centreline_error_pct"] < 15.0
    dst = Path(__file__).resolve().parents[2] / "artifacts" / "m17851_audit" / "fix_probe.json"
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()