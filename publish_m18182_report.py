"""Build /tmp/report.json (paper-grade report) and POST it to tensor_reports."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

PNG = pathlib.Path("runs/m18182_nebb_force/nebb_bodyforce_xval.png")
OUT = pathlib.Path("/tmp/report.json")

MD = r"""# 力修正 NEBB 壁面 + 体力驱动：TensorLBM 真实现、入口、基准与主求解器集成（m18182）

**锚点** `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S` · 仓 `TensorLBM` · 返修轮 1（原单 m17759；审计单 m17963 仅判 L1）

## ① 案例介绍
二维平面槽道：D2Q9、x 方向周期、上下壁面无滑移、x 向常体力 `Gx`。定常充分发展解为精确抛物线
`ux(y) = Gx/(2ν)·y(H−y)`，`H = ny−1`（无滑移平面即壁面节点行）。

| 参数 | 值 |
|---|---|
| 格子 | D2Q9，`c_s² = 1/3` |
| `nx × ny` | 4 × {10, 20, 40}（x 周期） |
| `ν` | 0.2（`τ = 1.1`） |
| `Gx` | 1.0e-3 |
| `u_max`（解析） | 0.05（ny=10） |
| Re = u_max·H/ν | 2.25 |
| Ma = u_max/c_s | 0.087 |
| 精度/环境 | CPU float64，单进程；推进到稳态残差 < 1e-13 |

## ② 怎么做的
1. **真实现**：`src/tensorlbm/boundary/nebb.py` —— 壁面节点非平衡反弹（NEBB）`f_k = f_k^eq(ρ_w,u_w) − f_k̄^eq + f_k̄`，
   壁面速度取**力修正**值 `u_w = 0 − Gx/(2ρ_w)`，壁面密度由场量反算 `ρ_w = (f_0 + Σ_tan + 2Σ_out)/(1−u_n)`；
   体力用 Guo(2002) 格式（平衡态在 `u* = u + F/(2ρ)`、源项在真实 `u`）。方向集全部**由格矢推导**（`c_y·n > 0`），不手抄索引表；
   复用 `boundary/solver2d.py` 的 `equilibrium2d / macroscopic2d / stream2d`。
2. **入口**：`PYTHONPATH=src python -m tensorlbm.verification.nebb_bodyforce.run --out runs/m18182_nebb_force`（9 条臂 + 收敛表 + 图）。
3. **集成（L3）**：`gpu_core_path.py` 主求解器新增 `wall_bc="nebb"`，`core_step` 分派到 `apply_channel_walls_nebb`；默认 `halfway` 路径不变。
4. **测试**：`tests/test_nebb_bodyforce_force_corrected.py` 11 项（格矢不变量、one-hot 流式约定探针、壁密度一致性、力修正符号、
   二阶收敛、稳态残差/质量漂移、负控制、主求解器接线与默认路径不变）——11 passed in 70.8 s。

## ③ 计算结果（程序化注入，源 runs/m18182_nebb_force/result.json）

| 臂 | ny | 步数到稳态 | max\|got−ref\|/max\|ref\| | 判定 |
|---|---|---|---|---|
| nebb_guo | 10 | 1 151 | **4.40 %** | <15 % ✓ |
| nebb_guo | 20 | 5 051 | **0.978 %** | <3 % ✓ |
| nebb_guo | 40 | 21 001 | **0.232 %** | <3 % ✓ |
| nebb_guo_nocorr（去掉 −Gx/2ρ_w） | 20 | 5 051 | 1.20 % | 负控制 |
| halfway（半格反弹） | 10 / 20 | 951 / 4 501 | 34.9 % / 14.4 % | 约定控制 |
| nebb_edm（仅平衡态位移） | 20 | 4 901 | 54.1 % | 如实否定 |
| corepath_halfway（主求解器，H=ny−2 基准） | 18 | 6 600 | 0.361 % | 默认路径不变 |
| corepath_nebb（主求解器，H=ny−1 基准） | 18 | 7 000 | 4.70 % | 壁节点约定、未完全收敛，如实报 |

收敛比：ny10→20 = **4.50×**，ny20→40 = **4.22×**，与 `Δy²` 一致 ⇒ 二阶精度。

## ④ 结果图
左：ny=20 稳态 `ux(y)` 剖面 vs 解析抛物线与半格反弹控制臂叠加；右：误差–网格 log-log 收敛（含 `∝Δy²` 参考线与 15 %/3 % 阈值线）。

## ⑤ 注意点（坑与适用边界）
- **流式约定陷阱**：本仓 `stream2d` 用 `torch.roll(shifts=−c)`，即 `f_k` 沿 `−c_k` 平流；"域外来流"方向集因此是 `c_y·n > 0`（与教材符号相反）。已用 one-hot 探针测试锁定。
- **壁面约定**：NEBB 无滑移面在壁面节点行（`H=ny−1`）；半格反弹在节点之间（`H=ny−2`）。两者基准不同，不可直接横向排名。
- **EDM 臂失败**：仅靠平衡态位移（无显式源项）无法维持稳态力平衡，误差 54 %——不粉饰，主路径只用 Guo。
- **力修正幅度**：`Gx=1e-3` 下修正把误差 1.20 %→0.978 %、壁面滑移 0.0027→0.0022；低 Ma 影响小但方向正确。
- **未做**：本容器无 `/tmp/vivsim_src`，未重跑 vivsim 原码；跨臂比较引用 m17759 已提交工件（NY=10 时 5.0 %），非同机新鲜运行。
- 原 m17759 报告把 NEBB 归因于 `src/tensorlbm/wave_bc_common.py`，该文件 `nebb` 命中为 0——这正是当初只判 L1 的直接原因；本单补齐真实现、入口与工件。

## ⑥ 误差判定（是否 <3 %）
- ny=10：4.40 % → <15 % ✓，**≥3 % ✗**
- ny=20：0.978 % → **<3 % ✓**
- ny=40：0.232 % → **<3 % ✓**
- corepath_halfway：0.361 % → <3 % ✓；corepath_nebb：4.70 % → **>3 % ✗**（壁节点约定 + ny=18 未完全收敛，如实记）
- 结论：**二阶收敛成立、细化网格达标；粗网格 ny=10 不满足 3 % 生态拉伸线，属分辨率限制而非格式缺陷。**
"""


def main() -> int:
    png = PNG.read_bytes()
    rep = {
        "title": "力修正 NEBB 壁面 + 体力驱动：TensorLBM 真实现/入口/基准/主求解器集成（m18182 返修轮1）",
        "summary": (
            "TensorLBM 新增 force-corrected NEBB 壁面 + Guo 体力驱动真实现"
            "（src/tensorlbm/boundary/nebb.py）、CLI 入口、主求解器 wall_bc='nebb' 集成；"
            "对解析 Poiseuille 误差 ny=10/20/40 = 4.40%/0.98%/0.23%，收敛比 4.5x/4.2x（二阶），11 项测试全绿。"
        ),
        "scenario_key": "xval_lbm_force_corrected_nebb_body_force_driver",
        "mission_id": 18182,
        "tags": ["TensorLBM", "NEBB", "body-force", "Poiseuille", "XVAL", "m18182"],
        "markdown": MD,
        "figures": [
            {
                "name": "nebb_bodyforce_xval.png",
                "caption": (
                    "左：ny=20 稳态 ux(y) vs 解析抛物线与半格反弹控制臂；"
                    "右：误差-网格 log-log 收敛（dy^2 参考线 + 15%/3% 阈值）"
                ),
                "png_base64": base64.b64encode(png).decode(),
            }
        ],
    }
    OUT.write_text(json.dumps(rep))
    base = os.environ["TENSOR_API_BASE"]
    body = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(
        base + "/auth/login", data=body, headers={"Content-Type": "application/json"}
    )
    tok = json.load(urllib.request.urlopen(req))
    token = tok.get("access_token") or tok.get("token")
    req2 = urllib.request.Request(
        base + "/tensor/reports",
        data=OUT.read_bytes(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    resp = urllib.request.urlopen(req2).read().decode()
    print("report:", resp[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())