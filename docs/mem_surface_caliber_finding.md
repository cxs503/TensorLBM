# MEM 口径发现：Ladd 动量交换力必须只对「壁面相邻 surface 格」求和

English abstract: *The Ladd (1994) momentum-exchange force on a staircase-voxel
solid body must be summed over the **wall-adjacent surface cells only**. Summing
over **all** solid cells adds a spurious, non-self-cancelling *interior*
contribution that inflates Cd. For the z-extruded cylinder Re = 40 this is the
difference between +4.5 % (all-solid, fails) and +1.6 % (surface-only, passes)
against the independent reference; for the sphere Re = 100 it is the difference
between +264 % (all-solid, the historical failure) and a single-digit error.*

Status: 发现（2026-09-30）．适用面：**所有用阶梯体素（staircase voxel）表示曲面/斜面的
几何**上的 Ladd/MEM 力（圆柱、球、SUBOFF 船体、任何非轴对齐 STL 体素化）。

---

## 1. 问题（what was wrong）

`obstacles.compute_obstacle_forces_3d` / `boundaries.compute_obstacle_forces`
实现的是 Ladd (1994) 动量交换：

```
F_α = 2 · Σ_{x_s ∈ solid} Σ_i c_{iα} f_i(x_s)      （post-stream, pre-bounce-back）
```

对**平面/轴对齐**障碍（如 2D 方柱、平板、腔体壁面），body force 的求和域用
「所有 solid 格」与「surface 格」**完全一致**，因为固体没有内部格（厚度≤1 格），
两者结果相同——所以这个口径在简单几何上从来没有暴露问题。

但对**曲面/斜面**（圆柱侧面、球面、船体），体素化后固体是一个厚实的体块：存在
大量**内部格（interior cells）**。这些内部格的 `Σ_i c_ix f_i` **不会自洽抵消**
（曲面上入射/出射方向不对称），于是给「全部 solid 求和」注入一个**伪项**。

## 2. 机制（why）

把 Ladd 求和拆成两部分：

```
Σ_{solid} = Σ_{surface} + Σ_{interior}
```

* **surface**（壁面相邻格，6-邻域内有至少一个流体格）：承载**真实的跨壁面动量流**
  ——bounce-back 在这些格上把法向动量翻转，正是这部分的动量交换构成真实的 Cd。
* **interior**（其余 solid 格）：bounce-back 在这里作用的是**内部网格点间的往返**，
  其净 x 动量在平面时不产生贡献；在曲面上因离散方向集 D3Q19 的对称性被破坏而**残留
  一个非零的伪项**。

实测（z 挤出圆柱 Re=40，nz=1，域 40D，阻塞 2.5%）：

| 口径 | Cd | vs 1.50 |
|---|---|---|
| `cd_mem_all`（全部 solid，旧口径） | 1.5673 | **+4.49 %** |
| **`cd_mem_surface`（仅壁面相邻，采纳口径）** | **1.5244** | **+1.63 %** ✅ |
| `cd_mem_interior`（仅内部，伪项） | 0.0429 | — |

即：**interior 项 = 旧口径的系统性高估**（+0.043 ≈ +2.9 pp）。

## 3. 为何是 surface 而非别的口径（交叉验证）

同一场上三种**独立**力方法对照（圆柱 D20@40D）：

| 方法 | Cd | vs 1.50 | 说明 |
|---|---|---|---|
| MEM surface-only | 1.5244 | +1.63 % | 本发现 |
| MEM all-solid | 1.5673 | +4.49 % | 旧口径（含伪项） |
| 压力/摩擦分解 `Cd_p+Cd_f` | 1.2940 | −13.7 % | 台阶面压力积分**漏掉驻点升压**，独立低估 |
| 参考（2D 数值簇） | 1.50 [1.48,1.52] | — | Dennis&Chang 1.522 / Fornberg 1.498 / Takami&Keller 1.48 |

surface-only 落在参考簇内，其余二者各偏一边。这与物理判据一致：
**只有 surface 求和等于真实的界面动量通量**。

## 4. 直接受益的用例

### 4.1 cylinder_3d（本用例，已验证）
两档 surface-only 均在 3% 内且收敛（见 `benchmarks/verified/cylinder_3d/`）。

### 4.2 sphere_re100（历史失败的真正原因）
球面历史 MEM 报 **+264 %（Cd≈3.98 vs Schiller-Naumann 1.09）**，当年归因为
「弯曲体素 MEM 不可用」。按本发现，**真正原因就是对全部 solid 求和**：球体内部格
极多（R=8 时 interior/total ≈ 0.7），伪项被放大。正确的做法是只取表面壳
（`run_mem_surface.py` 中的 `_surface_mask`）。参见
`benchmarks/pending/sphere_re100/run_mem_surface.py`（三口径并列）。

### 4.3 其他
任何 STL/球/椭球体素化后的 MEM 力（SUBOFF 壳、Ahmed body 等）若曾出现「MEM 显著
偏高、与压力/摩擦分裂互相矛盾」，应首先检查**求和域是否限制在 surface**。

## 5. 复现

```bash
# 从已保存的收敛场分解三口径（与 run.py 内部公式逐位一致）
PYTHONPATH=src python benchmarks/pending/cylinder_3d/_analyze_field_mem.py \
    runs/cyl3d_40D/D20_L40.json.field.pt 20 --nz 1 --lateral 40
# 输出：cd_mem(all)/cd_mem(surface)/cd_mem(interior)

# run.py（已内置三口径，采纳口径 = surface）
PYTHONPATH=src python benchmarks/verified/cylinder_3d/run.py single 20 \
    --out /tmp/d20.json --save-field --steps 40000 --nz 1 --lateral 40 \
    --device sdaa:0 --compile-mode eager
```

## 6. 备注 / 边界条件

* 对 **nz>1 的周期展向**圆柱，solid 格的 z-邻域恒为 solid，故 6-邻域 surface 掩码
  退化为 xy 4-邻域壁环；三口径因此仍是 z-无关的（与 nz=1 逐位一致）。
* interior 伪项随分辨率**每格递减**（D20: 1.6e-4/格；D40: 5.7e-5/格），但格数随 D²
  增长，总量在测得的 D 范围内缓慢上升；它不随收敛而消失，**不能靠加密网格洗掉**，
  只能靠正确的求和域剔除。
* 本发现**不改变** Ladd 公式本身（仍是 `2 Σ c_ix f_i`），只改变**求和域**。