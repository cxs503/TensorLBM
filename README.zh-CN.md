# TensorLBM

[English](README.md) | [简体中文](README.zh-CN.md)

TensorLBM 是一个以 CPU 为首要目标的 PyTorch 格子玻尔兹曼方法 (LBM) 平台，专注于**可复现的研究实验**，并提供清晰的扩展接口。

## 文档

- **[软件说明书 / Software Manual](docs/software_manual.md)** – 完整的船舶与海洋工程算例说明、定量 benchmark 对比和 API 参考。
- **[SUBOFF 平台使用说明书](docs/suboff_platform_manual.md)** – 完整 SUBOFF 全附件案例的命令行 / 平台运行流程、精度判据与结果解读。
- **[HPC + AI：AI 湍流模型](docs/ai_turbulence.md)** – 数据生成 → SQLite 入库 → AI 湍流模型训练 → AI 模型嵌入 LBM 的端到端示范。
- **[开发工作流 / Development Workflow](docs/development_workflow.md)** – 环境配置、代码检查、平台启动与输出命名规范。
- **[可观测性说明 / Observability Notes](docs/observability.md)** – 任务生命周期、输出 Schema 与故障排查清单。

## TensorLBM 提供的功能

- `src/tensorlbm/__init__.py` 中精简、明确的公开 API
- **D2Q9**、**D3Q19** 和 **D3Q27** 格子原语（平衡态、宏观量、格子常数）
- **BGK**、**MRT**、**TRT**、**正则化 BGK (RLBM)**、**累积量（Cumulant）** 碰撞算子（支持二维和三维）
- **自适应网格细化（AMR）**：D2Q9/D3Q19 动态 Patch 管理，最多 5 级细化（Filippova–Hänel 界面交换），非平衡量/涡量/梯度/边界层等细化指示子
- **DG-LBM 混合求解器**：P1-Lobatto 节点型间断 Galerkin LBM，SSP-RK3 时间推进，DG↔LBM P0 界面耦合，支持 2D 圆柱/3D 球体/SUBOFF 潜艇流动
- **大涡模拟（LES）**：Smagorinsky、动态 Smagorinsky（Germano 标识）、WALE、Vreman（支持 D2Q9/D3Q19/D3Q27）
- **RANS 湍流模型**：k-ε（`KESolver`）和 k-ω SST（`KOmegaSSTSolver`）
- **非牛顿幂律 BGK**：提供剪切率估计、表观黏度计算与空间变 τ 碰撞
- **多相流模型**（D2Q9 & D3Q19）：Shan-Chen 单/双组分、Color-Gradient、自由能/Phase-Field
- **浸入边界法（IBM）**：二维和三维直接力施加（2 点帽函数 / 4 点余弦 Delta 核）
- **热流耦合**（D2Q9/D3Q19 + D2Q5/D3Q7 双分布函数，Boussinesq 浮力）
- **共轭传热（CHT）**：流体–固体导热耦合与界面边界条件
- **气动声学**：Ffowcs Williams–Hawkings (FWH) 远场求解器、SPL 频谱、OASPL 计算
- **AI 湍流模型**：MLP 涡粘模型、基于 Transformer 的自监督流场模型、DNS→LES 数据管道、AI 嵌入 LBM 碰撞
- **边界条件**：反弹、Zou-He 进/出口 BC、Bouzidi 插值反弹、移动壁（Ladd 1994）、远场 BC、海绵/吸收层出口 BC、粗糙壁面（等效砂粒）、JONSWAP 不规则波浪入口
- **湍流进口剖面**：对数律、幂律、抛物线、Blasius、Womersley、合成湍流 2D、DFSEM、数字滤波法
- **湍流统计**：`TurbulenceStatsAccumulator`、雷诺应力、湍流强度、湍流长度尺度
- **流线/路径线追踪**：二维和三维积分、均匀/直线播种点、驻留时间计算
- **面积分与体积分**：质量流量、面积平均、表面力/力矩、力/力矩系数、压降
- **多 GPU 域分解**：`MultiGPUSolver2D/3D`、halo 交换、自动分解
- **多后端分发**：`torch`（默认）、`paddle`、`mindspore`，通过 `get_backend`/`set_backend` 切换
- **船舶水动力**：Wigley / Series60 / KCS 船体、SUBOFF 潜艇 CAD、螺旋桨（KP-505）、Airy 和 JONSWAP 波浪入口边界
- **基准算例**：圆柱绕流、球体绕流（D3Q19/D3Q27）、船体绕流、SUBOFF 阻力、椭球体、翼型（NACA 4 位数）、螺旋桨、IBM 螺旋桨、盘式促动器、后台阶、顶盖驱动方腔流、旋转圆柱、湍流槽道、液舱晃动、近床管道、溃坝、多孔介质（2D/3D）、多相水入水
- **后处理**：Strouhal FFT、Q 准则、λ₂ 准则、涡量、VTK/HDF5/XDMF 导出、流线、力系数、尾迹剖面

## 可部署 Web 平台（中英双语）

`platform/` 目录提供了一个可部署的 B/S 平台，具备以下功能：

- **前处理**：多边形障碍物掩码生成、随机多孔介质掩码、LBM 单位换算
- **求解器**：提交 10 余种仿真任务（圆柱绕流、方腔流、湍流通道、多相流、船体阻力等）
- **后处理**：查看快照图片、下载输出文件、查看日志
- **基准测试**：与已发表参考数据进行定量比较
- **船体 CAD**：Wigley / Series 60 / KCS 船体预览、掩码生成与仿真提交
- **AI 助手**：自然语言驱动的仿真工作流（支持中文指令）

### 双语界面

平台原生支持**简体中文**和**英文**实时切换：

1. 打开平台后，点击右上角导航栏中的 **`中文`** 按钮即可切换至中文界面，点击 **`EN`** 切回英文。
2. 语言选择会持久化到浏览器的 `localStorage`（键名：`tensorlbm_lang`），刷新后保持。
3. 首次访问时根据浏览器语言（`navigator.language`）自动检测，中文浏览器默认显示中文，其他语言默认英文。

#### 技术方案说明

采用**轻量 JSON 字典前端方案**：

```
app/frontend/static/i18n/en.json   # 英文词典
app/frontend/static/i18n/zh.json   # 中文词典
app/frontend/static/js/i18n.js     # i18n 引擎（t() 函数）
app/i18n/GLOSSARY.md               # 术语对照表
app/i18n/check_keys.py             # 字典完整性校验脚本
```

HTML 通过 `data-i18n="key"` 属性绑定翻译键；动态生成的 DOM 通过 `t('key')` 函数取值。

### 启动平台

```bash
cd app
pip install -r requirements.txt
bash start.sh
# 打开浏览器访问 http://localhost:8000
```

### 校验 i18n 字典

```bash
python app/i18n/check_keys.py
```

## 快速开始

```bash
pip install -e ".[dev]"
PYTHONPATH=src python examples/cylinder_flow.py
```

### 运行 DG-LBM 混合求解器

```bash
# 2D 圆柱 DG 算例
PYTHONPATH=src python examples/dg_lbm_cylinder_hybrid.py

# 3D SUBOFF 潜艇高 Re MRT 算例
PYTHONPATH=src python examples/dg_suboff_highre_mrt.py
```

### 运行 AI 湍流管道

```bash
# DNS 数据生成 → SQLite 入库 → AI 模型训练 → AI 嵌入 LBM
PYTHONPATH=src python examples/ai_dns_case.py
PYTHONPATH=src python examples/ai_turbulence_pipeline.py
```

## 运行测试

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 授权。
