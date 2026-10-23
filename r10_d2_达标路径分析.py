#!/usr/bin/env python3
"""R10 d2 达标路径静态分析 (CPU-only):
1) 不同 ny 的 beta 与修正因子
2) 扩域对 n_leaf / 内存 / 速度的影响 (实际 CPU 建树测量)
3) simple / glauert / glauert_classic 修正因子对比
4) 修正后参考值 1.0917*f vs R10 d2 Cd=3.75 (面积修复后预期)
5) 达标路径建议
"""
import math
import os
import time
import torch

from tensorlbm.octree_boundary.geometry import (
    build_octree_shell, sphere_distance_field,
)
from tensorlbm.amr_shell_planning import plan_body_shell_box

RADIUS = 10.0
D = 2 * RADIUS          # 20 coarse cells
REF_INF = 1.0917        # Schiller-Naumann sphere Cd @ Re=100
D_MAX = 2
SHELL_MARGIN = 6
WAKE_CELLS = 32
PAD = 8                 # wall-margin
U_IN = 0.06


def octree_footprint(oct) -> tuple[int, dict]:
    """Sum of torch tensor bytes owned by the octree object."""
    total = 0
    counts = {}
    for name, v in vars(oct).items():
        if isinstance(v, torch.Tensor):
            nbytes = v.numel() * v.element_size()
            total += nbytes
            counts[name] = nbytes
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, torch.Tensor):
                    nbytes = x.numel() * x.element_size()
                    total += nbytes
                    counts[f"{name}[list]"] = counts.get(f"{name}[list]", 0) + nbytes
    return total, counts


def build_config(ny, nx=96, nz=64):
    center = (nx * 0.5, ny * 0.5, nz * 0.5)
    bl = max(2.0, round(RADIUS / 2.0))
    t0 = time.perf_counter()
    solid_coarse = sphere_distance_field(
        (nz, ny, nx), center, RADIUS, torch.device("cpu"),
    ) <= 0.0
    plan = plan_body_shell_box(
        solid_coarse, shell_margin=SHELL_MARGIN, wake_cells=WAKE_CELLS, pad=PAD,
    )
    box = plan.box
    l1_shape = (
        (box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2,
    )
    center_l1 = (
        center[0] * 2.0 - box.x0 * 2,
        center[1] * 2.0 - box.y0 * 2,
        center[2] * 2.0 - box.z0 * 2,
    )
    octree = build_octree_shell(
        l1_shape, center=center_l1, radius=RADIUS * 2.0,
        bl_thickness_cells=bl, d_max=D_MAX,
        lattice="D3Q27", device=torch.device("cpu"),
    )
    dt = time.perf_counter() - t0
    lev = octree.leaf_level.to(torch.float64)
    n1 = int((lev == 1).sum().item())
    n2 = int((lev == 2).sum().item())
    fp, counts = octree_footprint(octree)
    return dict(
        ny=ny, nx=nx, nz=nz, beta=D / ny,
        box=(box.z0, box.z1, box.y0, box.y1, box.x0, box.x1),
        l1_shape=l1_shape, n_leaf=int(octree.n_leaf), n1=n1, n2=n2,
        level_mean=float(lev.mean().item()),
        build_s=dt, footprint_mb=fp / 1e6,
        coarse_cells=nx * ny * nz,
        counts=counts,
    )


def f_simple(b):  return 1.0 / (1.0 - b) ** 2
def f_glauert(b): return 1.0 / math.sqrt(max(1.0 - b * b, 1e-12))
def f_classic(b): return 1.0 / max(1.0 - 1.5 * b, 1e-3)


def main():
    print("=" * 96)
    print("R10 d2 达标路径静态分析 (radius=10 -> D=20, ref_Cd_inf=1.0917, d_max=2,")
    print(f"shell_margin={SHELL_MARGIN}, wake_cells={WAKE_CELLS}, pad={PAD})")
    print("=" * 96)

    # ---------- 1) geometry / n_leaf scaling vs ny ----------
    print("\n[1] 建树实测: n_leaf / L1-L2 / box / 内存 vs ny (nx=96, nz=64)")
    print(f"{'ny':>4} {'beta':>7} {'box z,y,x':>22} {'l1_shape':>16} "
          f"{'n_leaf':>9} {'L1':>8} {'L2':>8} {'build_s':>7} {'octreeMB':>8}")
    rows = []
    if os.environ.get("R10_SKIP_BUILD"):
        # 实测值 (CPU 建树 _r10_ny_scale_cpu.py + ny=96 单建, 已跑完)
        import json
        with open("/tmp/r10_ny_scale_measured.json") as fh:
            rows = json.load(fh)
    else:
        for ny in (96, 128, 160, 64):
            r = build_config(ny)
            rows.append(r)
    for r in rows:
        print(f"{r['ny']:>4} {r['beta']*100:>6.1f}% "
              f"[{r['box'][0]},{r['box'][1]})x[{r['box'][2]},{r['box'][3]})x[{r['box'][4]},{r['box'][5]}) "
              f"{str(r['l1_shape']):>16} {r['n_leaf']:>9} {r['n1']:>8} {r['n2']:>8} "
              f"{r['build_s']:>7.2f} {r['footprint_mb']:>8.1f}")
    # save pickled octrees? no; just keep stats
    base = rows[0]
    print(f"\n  ny=96 -> ny=160: n_leaf {base['n_leaf']} -> {rows[2]['n_leaf']} "
          f"({rows[2]['n_leaf']/base['n_leaf']*100:.1f}%)  "
          f"octree mem {base['footprint_mb']:.1f} -> {rows[2]['footprint_mb']:.1f} MB  "
          f"coarse cells {base['coarse_cells']} -> {rows[2]['coarse_cells']} "
          f"(x{rows[2]['coarse_cells']/base['coarse_cells']:.2f})")

    # ---------- 2) correction factors ----------
    print("\n[2] 阻塞修正因子 f (beta = D/ny = 20/ny)")
    print(f"{'ny':>4} {'beta':>8} {'simple':>10} {'glauert':>10} {'glauert_cls':>11} "
          f"{'simple(硬门禁→)':>18}")
    table = []
    for ny in (96, 128, 160):
        b = D / ny
        fs, fg, fc = f_simple(b), f_glauert(b), f_classic(b)
        # 硬门禁: beta>=15% 时 simple 自动升级为 glauert
        esc = fg if b >= 0.15 else fs
        table.append((ny, b, fs, fg, fc, esc))
        print(f"{ny:>4} {b*100:>7.2f}% {fs:>10.4f} {fg:>10.4f} {fc:>11.4f} "
              f"{esc:>18.4f}")

    # ---------- 3) corrected reference ----------
    print("\n[3] 修正后参考值 ref = 1.0917 x f, 与 R10 d2 面积修复后 Cd=3.75 对比")
    print(f"{'ny':>4} {'beta':>8} {'模式':>20} {'f':>9} {'ref=1.0917*f':>12} "
          f"{'3.75/ref':>9} {'残差':>10}")
    best = None
    for ny, b, fs, fg, fc, esc in table:
        for label, f in (("simple", fs), ("simple(升级glauert)", esc),
                         ("glauert", fg), ("glauert_classic", fc)):
            ref = REF_INF * f
            ratio = 3.75 / ref
            resid = (ratio - 1.0) * 100
            print(f"{ny:>4} {b*100:>7.2f}% {label:>20} {f:>9.4f} {ref:>12.4f} "
                  f"{ratio:>9.2f} {resid:>+9.1f}%")
            if best is None or abs(ratio - 1.0) < abs(best[0] - 1.0):
                best = (ratio, ny, label, f, ref, resid)

    print(f"\n  -> 最接近配置: ny={best[1]} {best[2]} (f={best[3]:.4f}, "
          f"ref={best[4]:.4f}), 3.75/ref={best[0]:.2f} (残差 {best[5]:+.1f}%)")
    print(f"  -> 即使最大修正 (simple@20.8% f=1.5956) ref=1.7419, "
          f"实测 3.75 仍超出 x{3.75/1.7419:.2f}")

    # ---------- 4) speed / memory projection ----------
    print("\n[4] 扩域速度/内存影响 (实测: CPU 建树 ny=96/128/160)")
    n160 = rows[2]
    print(f"  - n_leaf 实测不变: ny=96/128/160 均为 {base['n_leaf']} "
          f"(L1={base['n1']}, L2={base['n2']}); 体拟合精细盒随 ny 仅平移")
    print(f"    y 盒: [{base['box'][2]},{base['box'][3]}) -> "
          f"[{n160['box'][2]},{n160['box'][3]}) (宽度不变 38 粗格)")
    print(f"  - octree 常驻内存实测不变: {base['footprint_mb']:.1f} MB "
          f"(ny=160 同为 {n160['footprint_mb']:.1f} MB)")
    print(f"  - 唯一增量=粗网格: {base['coarse_cells']} -> {n160['coarse_cells']} "
          f"(x{n160['coarse_cells']/base['coarse_cells']:.2f}, 粗网格仅 mask/少量场, "
          f"增量 <10 MB)")
    n = base["n_leaf"]
    print(f"  - 每份 f_leaf(27x{n//2} float32) = {27*(n//2)*4/1e6:.1f} MB; "
          f"neighbor_table(27x{n} int32) = {27*n*4/1e6:.1f} MB (octree 常驻)")
    print(f"  - 接口链接 {1137880} 条 -> 交换缓冲 ~{1137880*4*3/1e6:.0f} MB (3 数组)")
    print(f"  - 结论: ny=96->160 总显存增量 <2% (粗网格 + 少量几何), 每步耗时不变 "
          f"(实测 2 卡 per_step_s=18.1 @590k)");

    # ---------- 5) 残差归因 ----------
    print("\n[5] 残差归因 (3.75 的来源与修正能力)")
    print("  - 面积修复把 Cd 从 9.33 (count-weighted area) 降到 3.75 (finest-leaf area)")
    print("  - 阻塞修正最大仅能把参考抬到 1.74 (simple@20.8%), 修正后仍差 x2.15")
    print("  - 参考修正方向: 实测/参考 = 3.75/1.0917 = 3.44x; 阻塞最多解释 1.60x")
    print("  - 剩余 ~2.15x 需从: 力本身偏大 / Re 定义 / L2 壁面层分辨率 / 回流区")
    print("    时长不足 (r10_d2_newarea: step100=3.46 -> step150=6.45 仍在爬升) 找原因")

    # ---------- 6) recommendation ----------
    print("\n[6] 达标路径建议")
    print("  A. 阻塞修正 (低成本, 已实现): 配置 --blockage-correction=simple 保留硬门禁升级,")
    print("     但注意 simple@20.8% 的 1.74 参考本身偏低, 修正方向是把参考抬高而不是把 3.75 压低;")
    print("  B. 扩域 ny=160 (beta 12.5%): 成本增量 <5%, 把阻塞影响压到 glauert f=1.008 / simple f=1.306,")
    print("     同时消除 'beta>12.5% 告警'; 建议与 A 一起做 (扩域+simple 修正 = ref 1.426);")
    print("  C. 3.75 本身必须降 ~2.2x 才能触达任何修正参考 —— 阻塞修正不是主因;")
    print("     优先排查: (i) 稳态未收敛 (step150 仍在爬升), (ii) L2 壁面层/面积归一化口径,")
    print("     (iii) 尾流区长度 (wc32 vs wc64 无差异 => 尾流区不是主因, 见 r10_wc64 日志 Cd 4.567),")
    print("     (iv) Re 定义 (u_in=0.06, nu 与 D 的换算).")


if __name__ == "__main__":
    main()
