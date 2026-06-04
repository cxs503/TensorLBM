# TensorLBM

[English](README.md) | [简体中文](README.zh-CN.md)

TensorLBM 是一个以 CPU 为首要目标的 PyTorch 格子玻尔兹曼方法 (LBM) 平台，专注于**可复现的研究实验**，并提供清晰的扩展接口。

## 文档

- **[软件说明书 / Software Manual](docs/software_manual.md)** – 完整的船舶与海洋工程算例说明、定量 benchmark 对比和 API 参考。
- **[SUBOFF 平台使用说明书](docs/suboff_platform_manual.md)** – 完整 SUBOFF 全附件案例的命令行 / 平台运行流程、精度判据与结果解读。
- **[HPC + AI：AI 湍流模型](docs/ai_turbulence.md)** – 数据生成 → SQLite 入库 → AI 湍流模型训练 → AI 模型嵌入 LBM 的端到端示范。

## TensorLBM 提供的功能

- `src/tensorlbm/__init__.py` 中精简、明确的公开 API
- **D2Q9** 和 **D3Q19** 格子原语（平衡态、宏观量、格子常数）
- **BGK**、**MRT**、**TRT**、**正则化 BGK** 碰撞算子（支持二维和三维）
- **大涡模拟（LES）**：Smagorinsky、动态 Smagorinsky（Germano 标识）
- **多相流模型**（D2Q9 & D3Q19）：Shan-Chen 单/双组分、Color-Gradient、自由能/Phase-Field
- **浸入边界法（IBM）**：二维和三维直接力施加
- **热流耦合**（D2Q9/D3Q19 + D2Q5/D3Q7 双分布函数）
- **船舶水动力**：Wigley / Series 60 / KCS 船体、不规则 JONSWAP 波浪入口边界
- **后处理工具**：Strouhal FFT、附加质量、Bouzidi 曲面边界 q 因子、HDF5/VTK/XDMF 输出

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
platform/frontend/static/i18n/en.json   # 英文词典
platform/frontend/static/i18n/zh.json   # 中文词典
platform/frontend/static/js/i18n.js     # i18n 引擎（t() 函数）
platform/i18n/GLOSSARY.md               # 术语对照表
platform/i18n/check_keys.py             # 字典完整性校验脚本
```

HTML 通过 `data-i18n="key"` 属性绑定翻译键；动态生成的 DOM 通过 `t('key')` 函数取值。

### 启动平台

```bash
cd platform
pip install -r requirements.txt
bash start.sh
# 打开浏览器访问 http://localhost:8000
```

### 校验 i18n 字典

```bash
python platform/i18n/check_keys.py
```

## 快速开始

```bash
pip install -r requirements.txt
PYTHONPATH=src python examples/cylinder_flow.py
```

## 运行测试

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 授权。
