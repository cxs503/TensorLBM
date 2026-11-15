# REPRO-LBM-HOU1995 — Ghia (1982) Re = 3200 中心线参考数据准入（TensorLBM）

**锚点**：`REPRO-LBM-HOU1995`（Hou, Zou, Chen & Doolen, *J. Comput. Phys.* **118** (1995) 329-347,
"Simulation of Cavity Flow by the Lattice Boltzmann Method"）。该文用 LBM 把顶盖驱动方腔算到
**Re = 3200**，并与 Ghia et al. (1982) 的中心线速度表逐点比较。

**本单结论一句话**：仓内此前**根本不可能**做 Re = 3200 的判分——`GHIA_RE100/400/1000` 到 1000 为止，
Ghia Table I 的 Re = 3200 列又被 `build_reference.py` 的 ERRATA 判为「不可判分」。本单把这一列
**以机器见证 + 显式隔离（quarantine）** 的方式准入为 `ghia_re3200_reference.json`
（7 项门全绿，16/17 行可判分、1 行带区间隔离），并修好了两处使该列**不可加载/不可复现**的工程缺陷；
求解器侧的 Re = 3200 收敛解**仍在集群排队**（cj3652/cj3653/cj3654），因此本单**不含** LBM vs Ghia 的
误差数字——诚实状态见 ⑥。

## ① 案例介绍（物理背景 / 几何 / 参数）

顶盖驱动二维方腔：`L = 1` 晶格单位，顶盖速度 `U_lid = 0.1`（`Ma = 0.173`，压缩性误差 `O(Ma²) ≈ 3 %`），
运动学黏性 `ν = U_lid L / Re`，`tau = 3ν + 0.5`。

| 项 | 值 |
|---|---|
| Re | 3200（HOU1995 二维算例上限） |
| 几何 | 方腔，V3 边界：三壁 pre-streaming 半程反弹 + 顶盖 Zou/He 动壁 |
| 晶格 / 碰撞 | D2Q9 / regularised BGK（`collide_rlbm`, Latt & Chopard 2006） |
| tau（等 Re 配方） | n=128 → 0.5120，n=192 → 0.5180，n=256 → 0.5240 |
| 基准 | Ghia, Ghia & Shin, *JCP* **48** (1982) 387-411，Table I（u, x/L=0.5）/ Table II（v, y/L=0.5）的 **Re = 3200 列**，期刊 pp. 398-399 |
| 预注册容差 | 逐点 `|ref − got| ≤ 5 % U_lid`（并同时报「是否 < 3 %」） |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **见证链（零手抄）**：论文自身 OCR 文本层
   `benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt`
   （sha256 `3637ef9e…`，PDF sha256 `1d76e463…`）为**唯一**原始见证；两份独立数字化拷贝
   （`copy_tableI_u_gist.txt` / `copy_tableII_v_pinn.txt`）只解析不重打；第三份低 Re MATLAB 拷贝做交叉校验。
2. **把 builder 参数化**（`build_reference.py` 新增 `--re/--out`）：一份代码准入任意 Ghia 列，
   输出 `u_re<RE>` / `v_re<RE>` 键与 `re` 字段；`--re 5000` 的**内容校验和与既有文件完全一致**
   （测试 `test_committed_json_is_reproducible_in_content` 钉住）。
3. **显式隔离策略（本单新增 `QUARANTINE`）**：某格只有在「OCR token 可证伪 + 论文自身的相邻行给出区间」
   时才允许隔离，隔离格**不进判分行**，而是带**可接受区间**导出到 `provenance.quarantined`。
   本单唯一隔离格：Table I, Re = 3200, y = 0.4531 —— OCR 读作 `-0.86636`（丢小数点），
   相邻行 y=0.5 → `-0.04272`、y=0.2813 → `-0.24427` 给出区间 `[-0.24427, -0.04272]`；
   重建值 `-0.06636` 落在区间内但**不用于判分**。
4. **修加载器**：`re5000/run.py::load_reference` 原来硬编码 `u_re5000`/`v_re5000` 键，
   任何声明 `"re": 3200` 的参考文件都**加载不了**（`KeyError`）。现按文件自报的 `re` 选择键，
   键名不符时**报错而不是静默回退**（无库内 fallback，避免"未验证表冒充已验证表"）。
5. **判分与分析工具**：`benchmarks/verified/cavity/re3200/analyze.py`
   ——两种坐标约定（`node` 遗留 / `wall` 物理）+ 网格收敛 + **隔离敏感度**（把被隔离行按区间两端各判一次），
   程序化注入 JSON 与图，绝不手抄数字。
6. **工具链**：`python -m pytest tests/test_cavity_re3200_reference.py`（10 passed）；
   `python benchmarks/verified/cavity/re5000/run.py --re 3200 --ref-json .../ghia_re3200_reference.json`。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 准入门（`reference/ghia_re3200_reference.json`, sha256 `42470d4e2526…`）

| 门 | 结果 |
|---|---|
| witness Table I（每个值必须出现在论文 OCR 文本层） | ✅ |
| witness Table II | ✅ |
| target column Table I（除隔离格外无单字形差、无超 5 % 的区间越界） | ✅ |
| target column Table II | ✅ |
| vs library Re=100/400/1000（对仓内既有表交叉校验） | ✅ |
| third copy（低 Re MATLAB 独立拷贝） | ✅ |
| **判分行** | **u: 16 / 17（1 行隔离）· v: 17 / 17** |

### 3.2 准入的 Re = 3200 列（`u(y)` on `x/L=0.5`）

| y | 0.0 | 0.0547 | 0.0625 | 0.0703 | 0.1016 | 0.1719 | 0.2813 | 0.5 | 0.6172 | 0.7344 | 0.8516 | 0.9531 | 0.9609 | 0.9688 | 0.9766 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u/U_lid | 0.0 | −0.32407 | −0.35344 | −0.37827 | −0.41933 | −0.34323 | −0.24427 | −0.04272 | 0.07156 | 0.19791 | 0.34682 | 0.46101 | 0.46547 | 0.48296 | 0.53236 | 1.0 |

（`y = 0.4531` 一行被隔离，见 ②3。）

### 3.3 隔离敏感度（准则不因丢弃该格而改变——由 `analyze.py` 在拿到收敛解后自动给出；
本次因无收敛解未计算，此处只给几何事实）

| 量 | 值 |
|---|---|
| 区间宽度 | `0.20155`（= 20.2 % U_lid） |
| 该行是剖面最平坦段 | 是（相邻两行差 0.2016，为全表最大行差之一） |
| 结论 | 该行**不能**判分（区间宽度远超 5 % 容差），故必须排除；排除后判据仍是 16+17 行 |

## ④ 结果图

* `benchmarks/verified/cavity/re3200/figures/re3200_reference.png`
  —— 准入的 Re = 3200 中心线剖面 vs 已准入的 Re = 1000 / 5000 邻列，并标出被隔离格的区间与 OCR token。
* `figures/re3200_centerlines.png` / `figures/re3200_error.png`：**未生成**——它们需要收敛的求解器场，
  而三档网格作业仍在集群排队（见 ⑥）。`analyze.py` 一旦拿到 `out/re3200_n*.json` 即会自动产出这两张图；
  在此之前它输出 `status: "WITHHELD"`，不画空图、不编数字。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **丢小数点是真坑**：Re = 3200 列此前被整体拒收，只因一格 OCR 丢小数点；但「整列拒收」与
   「整列准入」都不是正确处置——正确处置是**区间化隔离 + 不判分**，本单落成代码策略而非一次性修补。
2. **`run.py` 的 `u_re5000` 硬编码**是隐藏的阻塞项：即使准入表就在手边，Re ≠ 5000 也加载不了。
3. **仓内存在两套包树**：`./src/tensorlbm`（`pyproject` 声明）与 `./TensorLBM/src/tensorlbm`
   （`benchmarks/verified/**/run.py` 通过 `sys.path.insert(REPO/"TensorLBM"/"src")` 实际加载）。
   `verification.lid_driven_cavity` **只在后者**。本单不改这一结构（风险高、非本单目标），
   但记录之：任何"改 `src/` 却看不到效果"的现象都源于此。
4. **既有 Re = 5000 参考 JSON 的 `vs_library` 段已过时**：库内 `GHIA_RE400` 的表 II 已知错值
   （`v(0.1563)=0.28003` 应为 `0.28124`、`x=0.2266/0.2344` 互换、`-0.38598` 重复）在后续任务里已修，
   重新生成会给出 0 差异；为不破坏文档/测试里钉住的 sha256，本单**没有**重写该文件。
5. **适用范围**：本单交付的是**判据本体**（参考数据 + 加载 + 隔离策略 + 分析器），不是"复现已通过"。

## ⑥ 误差判定（是否 < 3 %）

* **参考数据层面**：210 个表值 **210/210** 可在论文 OCR 文本层追到（`witness` 门全绿），
  无一个是手抄；Re = 3200 列有 1 格被证实腐败并隔离 → 数据侧误差判定 **通过（结构可复现）**。
* **求解器层面（LBM vs Ghia, Re = 3200）**：**无法回答**——没有收敛解。
  三档网格已递交集群 `cj3652`(n=128) / `cj3653`(n=192) / `cj3654`(n=256)，
  递交时队列位置 50/51/52、在飞作业 67，轮询 11.5 min 仍在 `queued`；本机重计算被调度令硬拦，
  故**不报任何误差数字**。`analysis.json` 当前为 `{"grids": {128/192/256: WITHHELD}, "grid_convergence": WITHHELD}`。
* 因此锚点 `REPRO-LBM-HOU1995` 状态：**参考前置条件已解除，复现本身仍未验证（open）**。
  作业产物落到 `benchmarks/verified/cavity/re3200/out/` 后，运行
  `python benchmarks/verified/cavity/re3200/analyze.py` 即得逐点误差与是否 < 3 % 的判定。