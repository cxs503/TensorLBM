# 差异清单：本仓 vs joconnor22/LIFE（compiletime-params-plus-runtime-geometry-config）

锚点 `XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY`。左=`joconnor22/LIFE`（源码 `/tmp/LIFE-master`，GPL-3.0），右=TensorLBM 新增 `src/tensorlbm/runtime_geometry.py`。

## A. 做法（机制）差异

| # | 维度 | LIFE | 本仓 | 取舍与理由 |
|---|---|---|---|---|
| A1 | 编译期块载体 | `inc/params.h` 的 `const`/`#define`，改需 `make clean && make` | `CaseParams` frozen dataclass；`SimulationSetup.apply_params(rebuild=True)` 才放行 | Python 无编译步骤；用 frozen + 显式 rebuild 门 + `fingerprint()` 入证把「编译期不可变」变成可审计的机制，而非纪律 |
| A2 | 运行期几何载体 | `input/geometry.config`（唯一约定路径，缺文件直接返回=零体） | `read_geometry_config()` 接受目录或显式文件路径；缺文件抛 `FileNotFoundError` | 缺体静默返回在批量算例里会变成「假成功」，改为显式失败 |
| A3 | 两遍扫描 | 先 `floor(2πr/dx)`/`floor(L/dx)+1` 计数预留 vector，再构造 | `Scene.counts(dx)` 保留同一计数口径（用于对表/预留） | 语义保留，便于逐项对表 |
| A4 | 注释/错误处理 | 仅跳过**头部**注释块；`>>` 读到脏数据即静默错读 | `#` 可在任意行；列数/关键字/数字错即 `ValueError` | 严格失败优先于静默错读 |
| A5 | 越界检查 | 无（越界标记被静默丢弃/代价函数错位） | `Scene.validate_domain()` 拒绝越界体 | 运行期几何最容易出的错就是越界，前置拦截 |
| A6 | 体到流场的耦合 | 纯 IBM：标记力 + ε 校正 + 3 点 δ 核 | 增加**栅格化** `Scene.mask()` → 半步反弹 mask（保留 marker 生成供 IBM/FEM 路径） | 本仓已验证算例是 mask 半步反弹；栅格化让运行期几何能直接驱动已验证求解器，见 A7 |
| A7 | 重启期改体 | 可就地增删体 | `Scene.add/remove` + `SimulationSetup.add_body/remove_body`，且**不改参数指纹** | 逐字保留「改几何不需重编译」这一核心性质，并被测试断言 |
| A8 | 结构求解（FLEXIBLE） | Newmark-β(0.25,0.5) + Newton-Raphson(1e-10, ≤20) + Aitken 子迭代 | **未移植**，只给 `FilamentBody.beam_properties()`（A=dx·h, I=dx·h³/12） | 属另一 slug（`NEW:LBM-FSI-BLOCK-GS-AITKEN`），本次对标范围外；如实标注 |

## B. 参数（约定）差异

| # | 项 | LIFE | 本仓 | 影响 |
|---|---|---|---|---|
| B1 | 角度单位 | 度，构造时一次性 ×π/180（`src/IBMBody.cpp:100`） | 同（`FilamentBody.angle_rad`）；实测 0/30/45/90° 跨度误差 ≤1.6e-14% | 无 |
| B2 | 旋转矩阵 | `[[c,-s],[s,c]]`（`inc/Utils.h:258`） | 同（`_ROT_SIGN=+1`，可切换） | 无 |
| B3 | 圆标记数 | `floor(2πr/dx)` | 同；D10/D20/D40 → 31/62/125，与公式逐项相等 | 无 |
| B4 | 梁标记数 | `floor(L/dx)+1`，间距 `L/(n-1) ≥ dx` | 同；并显式测试该不等式 | 无 |
| B5 | 相邻体间距 | `space=0` 表示单体，位置 `start+i*space` | 同 | 无 |
| B6 | 标记去重 | `<0.5dx` 删；柔性优先；柔性-柔性报错；刚性-刚性删较早者 | 逐条复刻（`_overlap_keep`） | 无 |
| B7 | 坐标语义 | geometry.config 用**物理**坐标（Turek–Hron：H=0.41 m，圆柱 (0.2,0.2) r=0.05） | 同；经 `x_lat=(x_phys-ox)/dx` 映射到格点 | 关键：正是这一步让 LIFE 行 ↔ 本仓已验证格点 mask 可精确对齐（Δ=0 格） |

## C. 边界条件差异
- LIFE 的边界类型写死在 `params.h`（`WALL_LEFT eVelocity` 等 `#define`），运行期不可改；本仓把它收进 `CaseParams`（`bc_left/right/bottom/top` 字符串）并纳入指纹 → 与 LIFE 同属「编译期/重建级」不可变项，但可被 provenance 记录与校验。
- LIFE 压力入口不携带 Δp（压差只能走体积力 `dpdx`）；本仓沿用已验证通道算例的 Zou/He 速度入口 + Zou/He 压力出口，未改边界物理（避免把配置对标污染成边界对标）。

## D. 移植取舍（结论）
1. **采纳**：两文件拆分（编译期块/运行期几何）、`floor` 标记口径、度→弧度、阵列展开、去重规则——全部可逐项对表，误差 0。
2. **改造**：静默错读→显式异常；新增越界校验；新增 marker→mask 栅格化（本仓已验证求解器的入口）。
3. **不采纳**：C++ 头文件常量机制（Python 用 frozen dataclass + rebuild 门 + 指纹）；FEM 结构求解（超范围，见 A8）。
