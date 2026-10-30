# SDAA 扫荡报告 — `poiseuille_3d_pipe`（D3Q19 圆管 Hagen-Poiseuille）

- **轮次**：m74 round-2（诊断 + 修复绕行）
- **案例目录**：`benchmarks/verified/poiseuille_3d_pipe/`
- **驱动脚本（未修改）**：`benchmarks/verified/poiseuille_3d_pipe/run.py`
- **SDAA 扫荡器**：`scripts/sdaa_sweep_poiseuille_3d_pipe.py`（本轮新写；上一轮 m34 的
  `scripts/sdaa_sweep_poiseuille_3d_pipe.py` / `scripts/_launch_pipe.sh` 在工作树中
  **已不存在**，`git status` 干净 —— 即本轮实际是从零重建，而非"修 m34 存量"）
- **启动器**：`scripts/_launch_pipe.sh`
- **工件**：`artifacts/sdaa_poiseuille_3d_pipe.json`（判据 + 设备证据）、
  `artifacts/sdaa_poiseuille_3d_pipe_case.json`（案例原生 `run_case` 输出）
- **日志**：`runs/sdaa_poiseuille_3d_pipe.log`
- **判定**：**PASS（7/7 门）** — 单一档位 R=20

## 1. 设备证据（脚本采集，零手抄）

```
torch.sdaa.get_device_name(0) = /dev/tcaicard26 | torch_sdaa 3.1.1a3+gite3971b9
| torch 2.7.1 | is_available=True | count=1 | probe tensor on sdaa:0
```

独立张量探针（二次导入 `run.py`，`pipe_setup(20,6,'sdaa:0')` + `equilibrium3d` + 3 个完整
步进链）：`d/fluid2d/wall_mask.device = sdaa:0`，`f.device = sdaa:0`，`f.shape =
(19,43,43,120)`，3 步后 `finite=True`，SDAA 显存占用 20.65 MB，**CUDA 未使用**。
→ 计算张量确实在 SDAA 上，无 CPU/CUDA 冒充。

## 2. 配置（原口径，未改任何阈值）

| 项 | 值 | 来源 |
|---|---|---|
| 档位 | R=20（单档） | README 表 R=20 行 / `case_R20.json` |
| 网格 | `nz,ny,nx = 43,43,120`（=2R+3, 2R+3, 6R），221 880 voxels，流体 1257 格 | `run.py` `pipe_setup` |
| 模型 | D3Q19 + BGK，`tau=0.8` → `nu=0.1`；velocity 驱动 `u_in=0.02`（`U_max=2u_in=0.04`） | 案例默认 |
| 无量纲 | Re = 8.2，Ma = 0.0693 | 脚本算出 |
| 边界 | Zou-He 速度入口 x=0 + Zou-He 压力出口 x=nx−1 + 壁面 post-streaming 半程反弹 | 库函数 |
| 步数 | `min_steps = max_steps = 20000` | **README 下限**，不下探 |
| 比较口径 | `R_eff^Q = sqrt(2Q/(πU_max))`（流量反演，非剖面拟合），绝对归一化 `U_max=2u_in` | README 主判据 |

## 3. 结果

| 指标 | 本轮 SDAA 实测 | README 记录 | 门限 | 判定 |
|---|---|---|---|---|
| **主判据**：径向平均剖面中心区 max 相对误差（R_eff^Q） | **2.357 %** | 2.15 % | ≤ 3 % | **PASS** |
| 剖面 L2 相对误差 | 1.056 % | 1.29 % | ≤ 3 % | **PASS** |
| 中心线 `u_center` vs 名义解析 `2u_in` | **−1.121 %** | −1.12 % | ≤ 3 % | **PASS** |
| `R_eff^Q` | 20.1087 | 20.1088 | — | PASS（G5） |
| `R_eff^Q − R` | +0.1087 | +0.109 | +0.11 ± 0.05 | **PASS** |
| `Q/Q_ana^Q` | 1.000000000 | ≡1 | |ratio−1| ≤ 1e−6 | **PASS** |
| `u_center` vs `2u_in(R/R_eff^Q)²` | −0.043 % | −0.04 % | 诊断 | 一致 |
| 充分发展检查 `fd_max_rel_dev` | 0.641 % | — | 诊断 | < 1 % |
| 稳态（漂移 < 1e−5 / 2000 步） | True @ 20000 步 | True | 必须 | **PASS** |
| 场有限性 | True | True | 必须 | **PASS** |
| 步数 / 墙钟 | 20 400 步（20000 + 末 400 步时间平均）/ **501.8 s**（`elapsed_s`=479.9，≈39.9 步/s） | — | — | — |

### 判据逐条 PASS/FAIL（与 `artifacts` 中 `gates` 数组一一对应）

| 门 | 判据（原口径） | 数值 | 判定 |
|---|---|---|---|
| G1 | 径向平均剖面中心区 max 相对误差 ≤ 3 % | 2.3572 % | **PASS** |
| G2 | `|u_center − 2u_in|/2u_in` ≤ 3 %（不做任何半径反演的最严口径） | −1.1212 % | **PASS** |
| G3 | 场全部有限 | True | **PASS** |
| G4 | 稳态判据成立且 20000 ≤ n_steps ≤ 20000 | True @ 20000 | **PASS** |
| G5 | `R_eff^Q − R` 与文档几何常数（+0.11±0.05）一致 | +0.1087 | **PASS** |
| G6 | 流量恒等式 `Q/Q_ana^Q = 1` | 1.0000000 | **PASS** |
| G7 | 剖面 L2 相对误差 ≤ 3 % | 1.0556 % | **PASS** |

**披露项（非门限，与 README 披露口径一致，不得当作放宽）**

- 逐格（per-cell）中心区 max = **6.449 %**：楼梯阶壁单格散布，README 披露 1 记录 R=20 为
  6.45 % —— 本轮**精确复现**。它不是剖面形状误差，径向平均后消失（G1 = 2.36 %）。
- `u_max_dp_err_pct = −22.94 %`：用平直壁面中点 `R+0.5` 从测得 dp 反推 `u_max` 的诊断量，
  README 明确其为诊断项（不适用圆周阶梯壁），不参与判定。
- `mass_drift_pct = 0.608 %`：Zou-He 速度入口/压力出口组合并非逐步严格守恒；案例未设此门。
- 与 README 的差异：主判据 2.357 % vs 2.15 %（+0.21 pp），`R_fit−R` +0.233 vs +0.227。
  同档位同步数下差异来自末 400 步时间平均窗口与稳态停机记账；两者均远在 3 % 门内，
  已如实写入 `artifacts.reproducibility_vs_readme`。

## 4. 绕行记录（如实）

1. **`--compile-mode eager`**：Torch-SDAA 3.1.1a3 的 `teco_inductor` 后端无法编译完整 LBM
   步进链（`benchmarks/compile_route.py` docstring 记录的 L5 缺陷），`route_step('default')`
   在 SDAA 上本来就自动回退 eager。显式选 `eager` 只省掉一次失败编译的墙钟，**数值路径
   与回退路径完全一致**（`compile_route` 打印：`mode='eager' -> eager (compile_step
   passthrough) [sdaa-teco_inductor]`）。
2. **未使用 `TRITON_INTERPRET=1`**：eager 路径不进入 `torch.compile`/inductor，不产生 triton
   JIT trace，因此不需要解释器模式 —— 本轮**不存在 triton 依赖**。
3. **shell 120 s 单命令上限**：全量运行 501.8 s，故用 `nohup` 后台 + 轮询（见
   `scripts/_launch_pipe.sh`），非算力问题。
4. **未跑多档收敛阶梯**：R=40 档（约 1.6× 步进成本 / 更长墙钟）本轮不跑，符合任务"只做
   1 档最小网格、不追多档收敛"的预算纪律。因此本文件**不主张** README 级别的"网格收敛
   verified"属性，只主张 R=20 单档在 3 % 原判据下 PASS。

## 5. 复现

```bash
# 单档 R=20，README 步数下限，原判据
bash scripts/_launch_pipe.sh            # nohup 后台，日志 runs/sdaa_poiseuille_3d_pipe.log
# 或前台直跑（约 8 分钟）
python scripts/sdaa_sweep_poiseuille_3d_pipe.py --R 20 --min-steps 20000 --max-steps 20000
# 快速设备/路径自检（200 步，~19 s）
python scripts/sdaa_sweep_poiseuille_3d_pipe.py --smoke
```

退出码：0 = 全部门 PASS，1 = 有门 FAIL（honest-FAIL，脚本不降判据）。