# 湍流通道 channel_turb（Re_τ=180 u+ 剖面）（pending）

**状态：❌ 未达标（根因类：2D 通道不存在线性转捩路径——终态 Re_max≈4200 < 2D 线性稳定阈值 5772，流场停在"层流+Smagorinsky 涡粘"，对数律剖面结构性不可达）——湍流类案例不在已验证清单内，本目录为 2026-08-18 遗留档，真实模拟、无外推。**

## 物理问题

周期流向（x）、上下壁无滑移的平面通道，恒定体积力驱动（压力梯度等价）。
目标：稳态湍流壁面律剖面 u+(y+) 对比 DNS（Kim, Moin & Moser 1987 /
Moser, Kim & Mansour 1999 官方数据库，Re_τ=178.12，随附 `dns_ref/mkm180.csv`）
与对数律 u+=(1/0.41)ln(y+)+5.0，在 **y+∈[30,100]** 窗口 RMS 误差 ≤3%
（problems.md B26）。**湍流类至今不在仓库已验证案例清单中。**

## 共性模块

- 库原语（未修改）：`tensorlbm.turbulent_channel` 的 `TurbulentChannelConfig`
  + `run_turbulent_channel`；碰撞 = D2Q9 BGK + Smagorinsky LES
  （`turbulence.collide_smagorinsky_bgk`，C_s=0.1）。
- 边界：上下壁 post-streaming bounce-back（`boundaries.bounce_back_cells`），
  流向周期；驱动 = Guo 体型体力 a_x=2·u_τ²/H（模块内 `_apply_body_force_2d`）。
- 参数：Re_τ=180，u_τ=0.005（Ma≈0.13），n_steps=100000，平均窗 60000–100000。

## 计算结果现状（result.json + runs/，2026-08-18 实测）

| 网格 | Δy+ | 稳态 | RMS vs DNS（y+30–100） | 平均相对误差 vs DNS | 模块自算口径 | 达标 |
|------|-----|------|------------------------|--------------------|--------------|------|
| 128×32 | 12.0 | 准稳态平台（max\|u\|≈0.1167） | **4.586** | 27.68% | 3.409 | ✗ |
| 256×64 | 5.81 | 否（仍在活塞加速，max\|u\|≈0.0801） | **5.611** | 35.36% | 4.794 | ✗ |

- 细网格误差**更大**（层流爬坡更慢，同 10 万步离稳态更远）→ 加密不收敛。
- 湍流统计：vv≈0、uv≈0——无壁法向脉动、无雷诺应力，**流场不是湍流**。
- 壁面模型：post-streaming bounce-back，u+ 折算壁面偏移 wall_offset_assumed=0.1，
  Δy+ 每格 12.0/5.81（两档）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：✓（库 turbulent_channel 原语，未修改）。
- 直接观测量 ≤3%：✗（RMS 4.59/5.61，平均相对误差 27.7%/35.4%）。
- ≥2 档网格单调收敛：✗（两档均不达标且细档更差）。

## 根因/诊断（按重要性）

1. **2D 通道在该参数下无线性转捩路径（根本原因）**：2D 平面 Poiseuille
   线性稳定阈值 Re_c≈5772（Orr–Sommerfeld，基于 U_max·h/ν）；本参数终态
   Re_max = u_max·Re_τ/u_τ ≈ 4200 < 5772，且 Re_max 与网格无关——任何网格都
   不会线性失稳。2D 无涡拉伸，Smagorinsky 只增耗散，终态=亚临界层流+涡粘
   （中心 u+ = u_max/u_τ ≈23.3，显著高于 DNS 湍流剖面中枢，剖面偏"满"）。
   2D LES 无法复现 3D 湍流对数律。
2. **层流爬坡极慢 + 模块默认预算不足**：u_τ 受 Ma 上限约束，动量扩散时间
   H²/ν 为百万步量级（ν=4.17e-4/8.61e-4、H=30/62，估算），而模块默认
   n_steps=50000、averaging_start=20000——平均窗落在爬坡段。
3. **Smagorinsky 涡粘在层流中照样激活**，软化近壁剪切并移动有效壁面。
4. 模块 Guo 力为一阶形式且施加在含壁面行的全场：质量漂移 ~0.08–0.13%/1e5 步。

## 晋级路径

- 达标需要 **3D 槽道湍流**（3D 有涡拉伸，转捩可自持，DNS 参考同源可用）
  ——等于新案例（3D 网格 + 展向统计平均 + 初始场湍流化 + 数百万步预算，
  成本高）；或放弃"2D LES 复现对数律"这一判据（物理上不成立）。
- 模块侧独立可修：湍流种子初始场、按 H²/ν 定步数预算、壁面行力处理与
  质量漂移。这些修复不改变 2D 无转捩的根本结论。
- 达标即归档 PR 移 `pending/channel_turb` → `verified/`。

## 复现

```
cd benchmarks/pending/channel_turb
PYTHONPATH=../../../src python run.py scan --out-root /tmp/channel_runs \
  --grids 128x32 256x64 --re-tau 180 --u-tau 0.005 --cs 0.1 \
  --steps 100000 --avg-start 60000
python plot_profile.py     # -> uplus_profile.png
```

原始数据：`runs/{128x32,256x64}/`（velocity_profile.csv + run_metadata.json +
result.json）；DNS 参考：`dns_ref/mkm180.csv`（Moser 组官方数据库 65 点）。

<!-- PROVENANCE
[{"v": 4.586411042880767, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "errors.rms_vs_dns"},
 {"v": 0.27683109074794704, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "errors.mean_rel_vs_dns"},
 {"v": 3.8825853199190905, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "errors.rms_vs_loglaw"},
 {"v": 3.4091250896453857, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "log_law_rms_error_module_window"},
 {"v": 0.11674826592206955, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "steady_state.last_max_speed"},
 {"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "steady_state.plateaued"},
 {"v": 12.000000000000002, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "wall_model.delta_y_plus_per_cell"},
 {"v": 5.610730718550807, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "errors.rms_vs_dns"},
 {"v": 0.35355867901491006, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "errors.mean_rel_vs_dns"},
 {"v": 4.819611741164345, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "errors.rms_vs_loglaw"},
 {"v": 4.793848991394043, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "log_law_rms_error_module_window"},
 {"v": 0.08007322996854782, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "steady_state.last_max_speed"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "steady_state.plateaued"},
 {"v": 5.806451612903226, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "wall_model.delta_y_plus_per_cell"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "config.n_steps"},
 {"v": 60000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "config.averaging_start"},
 {"v": 30.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "comparison_window.y_plus_min"},
 {"v": 100.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "comparison_window.y_plus_max"},
 {"v": 180.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "config.re_tau"},
 {"v": 0.005, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "config.u_tau"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/result.json", "k": "verdict.passed"},
 {"v": 27.683109074794704, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "errors.mean_rel_vs_dns", "k2": "errors.mean_rel_vs_dns", "op": "scale100"},
 {"v": 35.355867901491006, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "errors.mean_rel_vs_dns", "k2": "errors.mean_rel_vs_dns", "op": "scale100"},
 {"v": 0.1, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "wall_model.wall_offset_assumed"},
 {"v": 23.34965318441391, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "steady_state.last_max_speed", "k2": "config.u_tau", "op": "ratio"},
 {"v": 0.00041666666666666664, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "derived.nu"},
 {"v": 0.0008611111111111111, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "derived.nu"},
 {"v": 30.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json", "k": "derived.H"},
 {"v": 62.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json", "k": "derived.H"}]
-->
