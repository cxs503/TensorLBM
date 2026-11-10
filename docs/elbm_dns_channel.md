# ELBM（熵格点 Boltzmann）槽道湍流 DNS：驱动、缺陷修复与验证口径

本文件记录 `p318w3-n03` 节点的工程与验证口径。算法本体（熵平衡态、熵条件
Newton、H 定理）在 `src/tensorlbm/elbm.py`，D3Q27 融合 OpenMP 内核在
`src/tensorlbm/elbm_channel_kernel.c`；本节点新增的是**可续算的高分辨率槽道
DNS 驱动**（`src/tensorlbm/elbm_channel_dns.py`）与**报告/判据工具**
（`tools/elbm_dns_report.py`），以及沿着这两个工具暴露出来的**两个真实缺陷的
修复**。

---

## 1. 物理与数值设置

* **熵格点 Boltzmann（ELBM）**：碰撞写成

  `f* = f + (α/τ)(f^eq − f)`，`ν = c_s²(τ/α − 1/2)`

  `α` 按胞、按步由熵条件 `H(f*) = H(f)` 解出（Newton + 正性夹逼 + `α ≤ 1`
  可容许上界；`α = 1` 即 BGK）。熵函数用离散 Gauss（Burg）熵
  `H = Σ f_i ln(f_i/w_i)`，平衡态用 Ansumali–Karlin 乘积闭式解。
  H 定理的离散表述：碰撞不使 H 增加 ⇒ 任意 `τ` 下无时间步限制（ELBM 的
  "无条件稳定"），且每个分布函数保持正值。
* **格子**：D3Q27（权重 `8/27, 2/27, 1/54, 1/216`）。
* **驱动**：Guo (2002) 体积力，含半步力速度平移（平衡态在 `u + F/2ρ` 处取），
  保证每步动量注入恰为 `ρF`；因此壁面摩擦速度由动量平衡**解析给出**
  `u_tau = sqrt(g h)`，无需从剖面拟合 —— 这是把 `C_f` 做成硬判据的前提。
* **壁面**：半程反弹（half-way bounce-back），无滑移面在 `y = 0.5` 与
  `y = ny − 1.5`，流道高 `H = ny − 2`，半高 `h = H/2`。
* **壁面单位**：`Δy+ = Re_tau / h`；`ν = u_tau h / Re_tau`；`τ = 3ν + 1/2`。
* **统计**：沿流向/展向两个齐次方向做面平均，再对时间平均（可跨续算累加），
  得到 `U(y)`、四个独立雷诺应力 `⟨u'u'⟩, ⟨v'v'⟩, ⟨w'w'⟩, ⟨u'v'⟩`、`k`、`C_f`。
* **DNS 参照**：Moser–Kim–Mansour (1999) 谱方法数据，随包内置
  （`tensorlbm/data/mkm1999/chan180.means|chan395.means` 等，provenance 带
  SHA-256），由 `tensorlbm.validation.mkm1999` 读取，`C_f = 2/U_b+²`。

## 2. 为什么需要新的驱动（本节点的工程增量）

原有 `native_channel_run` 是"一次调用跑完整个算例"的 C 驱动：**不能中途存盘**。
壁面解析 DNS 需要 `O(10^4–10^5)` 步、`O(10^6–10^7)` 胞，必须跨作业槽分段跑，
否则一次超时全部作废。新增 `elbm_channel_dns.py` 提供：

| 能力 | 实现 | 验证 |
|---|---|---|
| 逐步驱动（不复制场） | `native_channel_step_inplace`（新增，避免每步两次全场拷贝，DNS 规模下 >1 GB/step 的纯带宽浪费） | 与 `native_channel_step` 逐步逐位一致 |
| 续算 | `f`、熵热启动 `alpha`、统计累加器、步数、样本数、熵诊断 → 单一 `.npz` 检查点；`--resume`；配置不符直接报错而不是混网格 | `test_checkpoint_restart_is_bit_identical`：40 步一次跑 == 20+20 分两段跑（场与累加器逐位相同） |
| 作业槽切分 | `--max-wall-clock`：到点存盘退出，下次续跑 | `test_max_wall_clock_stops_and_resumes` |
| 统计 | float64 面平均 + 时间累加（跨续算连续），避免 `⟨u²⟩−⟨u⟩²` 的相消误差 | `test_sample_profiles_matches_a_brute_force_average` |
| 报告口径 | 输出 `tools/elbm_channel_report.py` 直接可读的 `prof_*/scalars/history` + `_meta.json` | `test_raw_npz_is_accepted_by_the_dns_gate_machinery` |

## 3. 两个真实缺陷（修前/修后都有数字）

### 3.1 精确平衡态上 `α` 塌缩到 0（fp32 镜面点）

* **现象**：把场设成**精确熵平衡态**后走一步，全部流体胞得到 `α = 0`
  （即完全不碰撞，等效粘性发散/消失），壁面胞 `α = 1`。
* **根因链**：`τ ≈ 0.5` 时 BGK 松弛率 `1/τ ≈ 2` 恰好落在熵抛物线的**镜面点**，
  该点精确的 `ΔH = 0`；而旧实现把 `f^eq` 存成 `float32`（`float feq[Q]`），
  每个分量的 `1e-9` 舍入经速率 2 放大并对 27 个方向求和，给出**虚假的
  `ΔH = +1.1e-5`**（实测，`τ = 0.5016`），超过容差 `1e-6(|H0|+1) = 1.07e-6`，
  求解器于是走夹逼分支，而夹逼区间内 `H(α)` 全是正增量 ⇒ 返回 `lo = 0`。
* **修复**：所有胞内代数（含 `f^eq`）用 double；`entropy_alpha` 显式接收
  `double *tol` 输出。修后同一状态 `α = 1.000000`（全部流体胞），
  `ΔH` 回到 fp64 量级（`1e-15`）。参考实现 `solve_entropy_condition` 同步修：
  float32 输入内部上采样到 float64 做熵代数，返回仍用调用者精度。
* **回归测试**：`test_kernel_returns_bgk_for_an_exact_equilibrium_state`、
  `test_torch_reference_returns_bgk_for_the_same_state`。

### 3.2 Guo 半步力速度平移污染熵条件

* **现象**：修完 3.1 后，`α` 仍会在**有体力**的算例里塌缩（粗网格/大加速度时）。
* **根因**：Guo 格式把平衡态取在 `u + F/2ρ` 处。沿这条方向 `ΔH` 不再为零：
  平衡态族的熵随 `|u|` 单调增，平移半步相当于对系统做外功，
  `ΔH ≈ ρ|u|F/2 > 0` **在物理上是真实的**，但它不属于 H 定理覆盖的"碰撞"部分。
  实测（h=8, u_tau=0.02）：`ΔH = +5.2e-5`，远超容差 ⇒ `α → 0`。
* **修复**：熵条件用**未平移**的平衡态 `f^eq(ρ,u)` 求解（H 定理的适用对象），
  碰撞仍朝**已平移**的平衡态 `f^eq(ρ,u+F/2ρ)` 松弛（保持 Guo 二阶精度）；
  正性夹逼沿真正的碰撞方向（平移后）计算。H 诊断只统计**无体力碰撞**部分
  （体力做功单列），并另报"扣掉 fp32 噪声底后的熵残差"
  （`entropy_excess_max_collision`）作为 H 定理的诚实残差。
* **回归测试**：`test_limiter_still_engages_on_a_strongly_distorted_state`
  （强畸变态仍被限幅，`α < 1`，`ΔH ≤ 1e-6`），以及 `tests/test_elbm.py`
  全部 H 定理/正性/粘性用例（20 项）在修复后仍全绿。

> 两个缺陷都是"测试绿但物理错"的类型：单元测试只覆盖到位形分布，
> 没有覆盖"精确平衡态 + 体力驱动"这一 DNS 必经状态。

## 4. 判据口径（机器可算）

| 判据 | 定义 | 通过线 |
|---|---|---|
| (a) 对数区斜率 | 对 `30 ≲ y+ ≲ 0.35 Re_tau` 区间拟合 `dU+/d(ln y+)` | 与 `1/κ = 2.439`（κ=0.41）相对误差 ≤ 5% |
| (b) 雷诺应力峰值位置 | `u_rms+ = sqrt(⟨u'u'⟩)/u_tau` 的内层峰 `y+` | 与 DNS 峰值位置偏差 ≤ 2（战役字母 12±2 同时单列；MKM-1999 的 `u_rms+` 峰在 `y+ ≈ 15`，见 `channel_gates` 注释） |
| (c) 摩擦系数 | `C_f = 2/U_b+²` | 与 DNS `C_f` 相对误差 < 5% |
| (d) 长时稳定性 | 有限性 + 正性 + 步数 > 1000 + 熵残差 ≤ 1e-5 | 全满足 |
| (e) pytest | `tests/test_elbm.py` + `tests/test_elbm_channel_dns.py` | 全绿 |

`U_b+` 用 `∫U dy / (h u_tau)` 计算；`Re_tau` 除标称值外，还由**实测粘性底层斜率**
反推（`ν_meas = u_tau²/(dU/dy)_w`）单列，避免把"额外观耗"藏起来 ——
ELBM 的 `ν(α)` 与 BGK 的差别只有用两个 `Re_tau` 才看得见。

## 5. 复现

```bash
# 1) 单元/回归测试（不需要集群）
python -m pytest tests/test_elbm.py tests/test_elbm_channel_dns.py -q

# 2) 一次壁面解析算例（可分段续算：重复执行同一命令加 --resume）
PYTHONPATH=src OMP_NUM_THREADS=64 python -m tensorlbm.elbm_channel_dns \
    --nx 576 --ny 290 --nz 288 --re-tau 180 --u-tau 0.010 \
    --steps 60000 --warmup 15000 --sample-every 25 \
    --ckpt-every 2000 --max-wall-clock 6600 \
    --label re180 --out runs/elbm_re180/raw/elbm_re180.npz --resume

# 3) 判据 + 图 + 报告（数字全部程序化，零手抄）
python tools/elbm_dns_report.py --raw runs/elbm_re180/raw/elbm_re180.npz \
    --out runs/elbm_re180 --label re180 --markdown docs/reports/elbm_dns_re180.md

# 4) 吞吐标定（决定网格/步数预算）
OMP_NUM_THREADS=64 python tools/elbm_dns_bench.py --nx 576 --ny 290 --nz 288
```

## 6. 局限（诚实边界）

* 熵条件是**离散**表述（对截断格子的近似熵面），不是连续统定理。
* 生产算例使用单一的熵条件松弛因子；KBC/ghost-mode 双群体路线
  （`tensorlbm.entropic_kbc`）未在本节点启用。
* Bose–Einstein 熵已实现并测过矩一致性，但生产碰撞用 Gauss 熵。
* 本节点的算例规模受算力约束（见报告"注意点"）；`Δy+`、域尺寸、统计时长
  三项都直接写进报告表，判据按机器算出的结果判定（含 FAIL），不做修饰。