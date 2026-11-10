# 壁模化 LES 的峰值位置诊断与高 Re 稳定性协议

**模块** `tensorlbm.wall_model_les` ｜ **工具** `tools/les_peak_diagnostic.py` ｜
**脚本** `benchmarks/wall_model_les/re590_stability.py`、`stability_attribution.py`、
`peak_offset_study.py`、`reproducibility_check.py`、`acceptance_p372w2.py`、`peak_report.py`
｜ **节点** p372w2-n01-wm-les-stability-peak（m10757）

本文记录三件可复用的东西：(i) 怎么从 LBM 流场把雷诺应力峰值量到亚格子精度并给出它的不确定度；
(ii) 怎么把一个高 Re 壁模化槽道的发散**归因**而不是猜；(iii) 怎么把网格量化误差与壁模型误差分开。

---

## 1. 峰值位置诊断

```python
from tensorlbm.wall_model_les import find_stress_peak, richardson_gci

est = find_stress_peak(y_plus, uv_plus, y_plus_max=0.25 * h_plus)
est.y_peak              # 样条亚格子峰位
est.y_peak_parabola     # 独立三点抛物线估计
est.uncertainty         # 两者之差的一半 = 插值不确定度
est.spacing             # 峰附近的局部间距（壁单位）
```

设计要点（每条都由测试钉住，`tests/test_les_peak_diagnostic.py`）：

* **只在离散最大点周围三点区间上拟合**。壁模化网格在 `y+ < 100` 里往往只有 3–6 个点，
  全局样条会给出与数据无关的伪极值。
* **报两个估计量**（样条 / 抛物线）而不是一个：在 `dy+` 与峰位同阶时，两者之差就是
  插值误差的量级，把它折进 `uncertainty` 比给一个假精度更有用。
* **搜索带限在槽道内层**（默认 `y+ ≤ 0.25 h+`）。整条半槽搜索时，短统计窗的噪声会把
  最大值推到中线（实测：同配置两种子在 600 步窗内峰位从 `y+ = 23` 跳到 `y+ = 345`）。
* **参照链可核对**：`load_channel_means_table` + `reynolds_stress_from_mean_profile` 从
  MKM 平均剖面用精确槽道动量平衡 `-<u'v'>+ = 1 - y/h - dU+/dy+` 复原应力，壁面与中线两端
  恒等式严格成立（表内首行 `dU+/dy+·Re_tau = Re_tau`，故壁面值恰为 0），
  并复现文献值（MKM-180 峰位 `y+ = 31.7`、峰值 `0.723`）。

命令行：

```bash
python tools/les_peak_diagnostic.py --selftest                       # 解析剖面自检
python tools/les_peak_diagnostic.py --dns-means benchmarks/pending/channel_turb_3d/dns_ref/mkm590.means
python tools/les_peak_diagnostic.py --profiles artifacts/wall_model_les_peak/profiles_fine.npz \
    --dns-means benchmarks/pending/channel_turb_3d/dns_ref/mkm590.means
python tools/les_peak_diagnostic.py --compare fine,medium,coarse \
    --dir artifacts/wall_model_les_peak --json /tmp/peak.json
```

**壁单位约定**：流场里的 `uv` 是格点单位，`-<u'v'>+ = -uv / u_tau^2`。归一化用**实测**
`u_tau_model`（不是配置里给的 `u_tau`），并同时记录比值 `u_tau_model/u_tau`——这个比值偏离 1
就说明运行还没到力平衡，任何与同名 `Re_tau` 的 DNS 比较都要打折。

## 2. 高 Re 稳定性归因协议

```bash
python benchmarks/wall_model_les/re590_stability.py --sweep all --steps 600   # 快速轴扫描
python benchmarks/wall_model_les/re590_stability.py --single --ny 49 --nx 32 --nz 32 \
    --steps 2200 --label base --out runs/x                           # 单个生产尺度算例
python benchmarks/wall_model_les/stability_attribution.py --cases runs/x runs/y
```

每个算例的 JSON 带全时程 `u_bulk / u_tau_model / 最小松弛裕度 3(nu+nu_t)_min / SGS 比 /
max_speed / rho_min`、首次非有限步和异常文本。归因表给出 H1–H4 的判别实验与裁决。

**踩过的三个坑（必须写进任何后续记录）**

1. **短盒掩盖发散**：`8×8` 或 `16×16` 横向盒在 Re_tau = 590、`u_tau = 0.02` 下跑 2200 步不 NaN；
   同参数换 `32×32` 盒在 1000–1700 步内发散。没有解析出湍流的盒宽测不出稳定性。
2. **只看 `tau-1/2` 会误判**：全时程 `3(nu+nu_t)_min` 都在 `O(1e-3)`，从未趋零；
   真正的判别量是 `max|u|`（见下）。
3. **欠松弛不是免费的**：`wall_relaxation < 1` 会改变壁模型反演的输入速度，
   耦合系统的不动点随之漂移（实测 `u_tau_model/u_tau` 远超 1），不能用它充当「稳定化」。

**速度包络判据**（本节点的主结论）：在固定壁单位网格上，稳定性由**已解析速度尺度**决定。
扫描里发散算例的最小 `max|u|` 与稳定算例的最大 `max|u|` 之间没有重叠，
分界就在 lattice 稳定包络 `|u| ≈ 1` 附近。实践推论：给定网格与 `dy+`，
必须把 `u_tau`（等价地把 `Ma`）压到让 `max|u| ≲ 0.9`，否则提高 `Cs`、降低 `beta`
只能推迟发散。

## 3. 网格误差与模型误差的分离

```bash
python benchmarks/wall_model_les/peak_offset_study.py --grid fine --steps 4500 --averaging-start 2000
python benchmarks/wall_model_les/peak_offset_study.py --analyse
python benchmarks/wall_model_les/acceptance_p372w2.py
```

* `richardson_gci` 用不等细化比的一般形式解观测阶 `p`，
  返回 `f_ext`、`GCI_fine`、`grid_error_fine = |f_fine - f_ext|`、`model_error = |f_ext - f_ref|`；
  三重值非单调时 `converged = False` 并写明原因（粗壁模化网格的常态）。
* 每套网格还带**平衡性门**：`u_tau_model/u_tau ≈ 1` 且解析应力不超过 `1 - y+/h+`。
  不过门时峰值数字只作诊断，不作判据。
* 修正映射 `fit_offset_correction` 用留一点交叉验证选线性/反比族，
  `spacing_for_tolerance` 反解达到 15% 容差所需的 `dy+`。

## 4. 与既有模块的关系

`wall_model_les.channel` 本轮新增（默认行为不变）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `exchange_offset` | 1 | 壁模型采样行相对壁节点的偏移（交换点行轴） |
| `wall_relaxation` | 1.0 | 壁模型目标速度的欠松弛因子（诊断用；不动点会漂移） |
| `diagnostics` | False | 日志里追加 `max_speed / rho_min / nu_sgs / tau_minus_half_min` |

`stability_diagnostics()` 是这些量的实现，单元测试钉住静止态下 `nu_sgs = 0`、
`tau_minus_half_min = 3 nu`。