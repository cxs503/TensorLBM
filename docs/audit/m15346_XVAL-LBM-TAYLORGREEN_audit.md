# m15346 交差审计 · XVAL-LBM-TAYLORGREEN（m15290 交付核验）

审计席位：board-audit（交差核验单，非重做；未修改被审交付代码）
被审单：m15290（succeeded）→ 交付 commit `9e4694fc`，合并提交 `04bb92e6`（Merge #1794）
被审平台报告：tensor_reports id=14387（六段齐全，1 图）
审计基线：ab 审计工作树 `/root/tp-agent/work/m15346`（master `fd81edf2`，含 #1794 合并）

---

## ① 案例与判据（先于取证的裁判定义）

| 项 | 值 | 出处 |
|---|---|---|
| 算例 | 2D 周期盒 Taylor-Green 衰减涡，N=128²，Re=100，U0=0.05，Ma=0.0866 | `runs/xval_taylor_green/case_spec.json` |
| 解析参考 | u=-A cos(kx)sin(ky), v=+A sin(kx)cos(ky), A=U0·e^{-2νk²t}；K=(U0²/4)e^{-4νk²t} | `src/tensorlbm/xval/case_spec.py` |
| 精确衰减率 | γ_K=4νk²=6.168503e-4 /step，γ_vel=2νk²=3.084251e-4 /step | 闭式，审计独立复算 |
| 预登记阈值（运行前写入代码） | 跨工具 γ_K **<5%**；vs 精确 γ_K **<8%**；相位 **<3% of π**；场 L2 **<8%** | `src/tensorlbm/xval/campaign.py`（`predeclared_thresholds` 随工件落盘） |
| 平台机器裁判 | **ABSENT**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-TAYLORGREEN'` = **0 行**（表内全库 15 行，无任何 TAYLOR/GREEN 条目） | 见 §④ F2 |

> 判据分两路：**平台侧机器裁判缺失**（无 benchmark_cases 行），因此本次以「闭式解析解 + 交付内预登记阈值」为裁判，并对阈值本身做独立性复核（§③）。

## ② 三件套逐项核验

### 2.1 run_artifacts（真实运行工件）

被审单工件目录 `/root/tp-agent/work/m15290/runs/xval_taylor_green/` 13 个文件全在（result.json 8050 B、场快照 4×~820 KB、k 历史 2 份、xval_table.csv、PNG 122 KB、run_info×2、case_spec.json），`runs/xval_taylor_green_boosted/` 亦在。
**缺陷（F6，低）**：`runs/` 未入 commit（`git ls-files runs/xval_taylor_green` = 0），交付只带源码。审计为耐久起见**独立重跑**并落盘于本分支 `artifacts/m15346/repro_full/`（3.4 MB，同 13 件）。

### 2.2 benchmark ref vs got（数值对照 + 独立复算）

审计**不看报告表**，直接以原始 CSV 重算（`artifacts/m15346/rederive_metrics.py`：log-线性拟合 K(t)/|u|max，场 L2 由 `field_*.csv` 与解析场逐点重算）：

| 量 | 工具 | 交付/报告 | 审计由原始数据重算 | Δ |
|---|---|---|---|---|
| γ_K | tensorlbm | 6.166753e-4 | 6.166753e-4 | **+0.0000%** |
| γ_K | palabos | 6.172737e-4 | 6.172737e-4 | **+0.0000%** |
| γ_vel | tensorlbm | 3.084849e-4 | 3.084849e-4 | **+0.0000%** |
| γ_vel | palabos | 3.062943e-4 | 3.062943e-4 | **+0.0000%** |
| field L2 t=1000 | tensorlbm / palabos | 0.300277% / 2.517154% | 0.300277% / 2.517154% | −4e-16 / −3e-15 |
| field L2 t=2500 | tensorlbm / palabos | 0.457453% / 3.091123% | 0.457453% / 3.091123% | 0 / −9e-16 |

对照闭式裁判（误差 = |got−ref|/ref）：

| 判据 | 最差偏差 | 阈值 | 裁定 |
|---|---|---|---|
| vs 精确 γ_K | **0.0686%** | <8% | PASS（<3%） |
| vs 精确 γ_vel | **0.691%** | <8% | PASS（<3%） |
| 跨工具 γ_K | **0.0970%** | <5% | PASS（<3%） |
| vs 精确 场 L2 (t=2500) | **3.091%** | <8% | PASS（**>3%**，在 8% 门内） |
| 相位误差 | 2.05e-13 % | <3% | PASS 但**判据空转**（见 F3） |
| 三方差分 | — | 三工具 | **FAIL(degraded)**：仅 2 工具（见 F1） |
| boost 变体（伴生算例） | 跨工具 35.2% / vs 精确 42.2% | 5% / 8% | **honest FAIL**（交付如实登记，未粉饰） |

### 2.3 repro_cmd（可执行性）

审计在工作树独立执行（非引用交付日志）：

```
python3 benchmarks/xval_taylor_green/run_xval.py --out artifacts/m15346/repro_full --threads 8
# 91.2 s wall（tensorlbm 73.6 s + palabos 5.5 s），无 OpenFOAM → frictions 留痕
```

对 208 个 result 字段逐键比对：**203 个数值字段逐位相同（Δ=0）**，5 处差异全为元数据（threads 16→8、wall_sec、palabos 二进制绝对路径、figure 路径）。→ **确定性可复现**（跨工作树、跨线程数）。

## ③ 实现真实性（防"报告好看"）

- 我方工具**真跑主求解器内核**：`runners.py:82-83` `from ..d2q9 import equilibrium, macroscopic` / `from ..solver import collide_bgk, stream` —— 非另写玩具求解器；IC 为 f=f_eq，float64，D2Q9 BGK。
- 第三方腿**确为 palabos 库**：`benchmarks/xval_taylor_green/palabos/bin/tg2d_palabos`（LoongArch64 ELF，1.04 MB）内含 `Palabos generic/memory/IO exception` 等 libpalabos 符号，`build.sh` 显式链接 `$PLB_BUILD/libpalabos.a`；源码 `tg2d_palabos.cpp`（228 行）在仓 → **独立实现，非自查重贴牌**。
- 数值接触面独立：三方同 case_spec（`case_spec.json` 一份三用，单元中心 x_i=i+0.5 显式写死），palabos 的 argv 由 `TG2DSpec.palabos_argv()` 生成（`tests/test_xval_taylor_green.py:53-59` 断言）。
- 测试质量（非仅"跑通"）：13 用例断言**不变量** —— 由闭式能量序列拟合能复原 γ（rel 1e-9，`test_exact_energy_fit_recovers_closed_form_rate`）、τ=0.5+3ν / k=2π/N / γ_K=2γ_vel 恒等式、场误差原语（自身比对=0、零场=100%）、**判据能抓出 20% 偏离的工具并定位疑似侧**（`test_table_and_criteria_flag_the_deviating_tool`）。`pytest -q` = **13 passed, 9.89 s**。

## ④ 审计发现（按严重度）

**F1｜中（阻塞"三方"规格，不阻塞 L4 实质）**：锚点名"三方横向"，实跑仅 **2 工具**；OpenFOAM icoFoam 在本镜像不可安装（适配器+算例生成器已落地但 `available:false`），`three_way_cross_check` 机器判据 **FAIL(degraded)** 并写 `frictions` 留痕（`result.json.frictions[0].reason="icoFoam not found on PATH..."`）。交付**如实降级、未伪装三方**；但锚点的第三方腿客观上未交付。

**F2｜中（裁判面缺口，平台侧问题）**：`tensor_benchmark_cases` 无本锚点行 → 无机器裁判，交付自带的 5%/8%/3% 阈值虽在运行前写入代码并随工件落盘（`result.json.predeclared_thresholds`），但**未登记进算例池**，下一轮无法被门机器比对。建议平台补行（ref=闭式 4νk²，tol 见上）。

**F3｜中（判据设计缺陷：不可失败的门）**：`phase_err_pct` 两工具分别为 2.05e-13%、7.9e-15%（即 0）——因 k=2π/N 与单元中心 x_i=i+0.5 的组合使离散采样点**恰好落在解析解上**，相位判据结构上不可能失败，不是有效证据。同类：`field_l2_pct`/`phase_err_pct` 的跨工具 `cross_diff`（148%、185%）在量值近 0 时无意义 —— 交付在报告 ⑤.2 已自述该坑，**未据此掩盖**，但判据表仍应把这两项降级为 informational。

**F4｜低-中（交付自认未根因的 7 倍场差，审计把它量化了）**：t=2500 我方场误差 0.457% vs palabos 3.091%。审计对差分场做 2D FFT（`artifacts/m15346/spectral_diagnosis.py`）：
- tensorlbm 误差主导模式 **k=(±2,0)，幅值 3.48e-5→3.66e-5**（t=1000→2500，**不衰减**）；
- palabos 误差主导模式 **k=(0,±1)，幅值 2.13e-4→2.07e-4**（同样不衰减），为前者 **5.9 倍**；
- 幅值比 2.07e-4/U0 ≈ 0.41%，量级与 Ma²≈0.75% 同阶，指向两驱动各自的**启动瞬态/非平衡初值处理差异**（双方 IC 均只给 f=f_eq，二阶项缺失），非粘性物理模式。
→ 报告 ⑤.3 的假设（"主导误差模 (kx=0,ky=1)、声学暂态+算子次序"）被审计**独立复现并量化**，但**根因仍未分离**（需对照：palabos compressor 开关 / 初值二阶项 / 采样口径）。

**F5｜低（诚实失败，不覆盖）**：boost（Galilean 平移）伴生算例 decay 判据全面不过（跨工具 35.2%、vs 精确 42.2%，相位率 1.0%），交付如实登记为 FAIL 并列出三条未分离假说，无粉饰。

**F6｜低（可复现性存续性）**：`runs/` 未入 commit；本次审计已把重跑工件落到 `artifacts/m15346/repro_full/`（将随本分支入库）。

**F7｜低（集成是代码级、非线上热态）**：`app/backend/main.py:47,214` 已注册 `xval_taylor_green` 路由（`/api/v1/xval/taylor-green`，n≤64/steps≤2000 限流），审计用 FastAPI TestClient **实跑 200** 并返回真实指标（`artifacts/m15346/api_xval_response.json`：n=32/steps=400 → overall=DEGRADED_PASS，粗网格下 decay 判据如预期不过）。但**部署中的 TensorLBM 应用实例（`/root/TensorLBM_test_waic`，:3000）仍是旧检出**，线上 GET `/api/v1/xval/...` 目前 404；平台主服务 :8010 属 TensorPlatform，不含此路由。→ 集成"接线真、热态未部署"。

**未发现**：MOCK/硬编码数字（所有指标由场数据程序化计算，审计逐点重算一致）、伪造运行（重跑逐位一致）、报告与工件不符（报告表 ③ 与 result.json 一致）。

## ⑤ 判级

**L4（基准验证）**——完整算例（128²/4900 步/50 采样/2 场快照）+ 误差分析（vs 解析解 0.069%、跨工具 0.097%、场 L2 0.46%/3.09% 并附模式归因）+ 可复现（审计独立重跑逐位一致）。

- **达 L1**：`tests/test_xval_taylor_green.py` 13 passed，且断言不变量。
- **达 L2**：真实现 1 492 行（`src/tensorlbm/xval/{case_spec,metrics,runners,campaign,cli}.py` + 228 行 palabos 驱动），输入→计算→输出闭环，非占位。
- **达 L3**：真调用主求解器内核（`d2q9.equilibrium`、`solver.collide_bgk/stream`）+ 仓内 API 层注册并经 TestClient 实跑 200（F7 说明线上未热部署，属部署侧）。
- **达 L4**：见上；F1（第三方腿缺失）与 F3（相位门空转）是**锚点规格/判据设计**的缺口，不是本单数值错误；F2 是平台裁判面缺口。
- **不达 L5**：无科研增量闭环——跨工具场差主导模式（F4）与 boost 失败（F5）均**未根因分离**，且平台算例池无登记、三方规格缺一腿。附 56 行六段报告与 1 图不改变级别。

## ⑥ 未做 / 局限

- 未修任何代码（交差核验单）；未补 pytest（既有 13 例已覆盖不变量）。
- 未尝试安装 OpenFOAM（镜像无包，交付已记 friction；审计不改环境）。
- F4 未做根因分离实验（compressor/初值二阶项对照需改 palabos 驱动重编，属返修/研究轮）。
- 线程数 8 vs 交付 16 的重跑：数值逐位一致（D2Q9 逐点算子，无归约），故不作独立性风险。
- 审计自身独立性限制：与交付同机同仓同镜像；未在 x86 CI 复跑（palabos 二进制为 loongarch64，CI 需重编）。