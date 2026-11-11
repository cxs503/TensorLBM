#!/usr/bin/env python
"""REPRO-LBM-GUO2002 —— 单跑 worker：差分加热方腔（DHC）定常平均 Nu。

复现靶标（外部锚）
------------------
Guo Z., Shi B., Zheng C. (2002) *A coupled lattice BGK model for the
Boussinesq equations*, Int. J. Numer. Meth. Fluids **39**:325-342.
该文用"速度分布 f + 温度分布 g 双分布耦合 LBGK + Boussinesq 浮力"求解
二维方腔自然对流，并以 de Vahl Davis (1983) 的平均 Nusselt 数为基准。

本工作**不另起炉灶**：全部物理（D2Q9 BGK + Guo 力格式浮力、D2Q5 温度
BGK、半程反弹 no-slip、anti-bounce-back Dirichlet 温度壁、壁面三点二次
插值 Nu）都取 ``tensorlbm.rb_convection_validation``（其底层为
``tensorlbm.thermal``）的既有实现，本脚本只做参数换算、时间推进编排与
对表。

预注册（判据先于动手，experimental design）
-------------------------------------------
* **决策**：TensorLBM 现有双分布耦合 LBGK（Boussinesq/Guo 力）能否在
  Pr=0.71 方腔内把平均 Nu 复现到 de Vahl Davis (1983) 基准的 **<5%**？
* **竞争方案**：
  A. 单网格直接对表（既有前作口径，ny=128）——已知 Ra=1e5 偏低 8.9%；
  B. **网格阶梯 + Richardson 外推**（Ra/Pr/ma_max 固定，h→0）——本脚本；
  C. 阶梯 + 马赫数轴灵敏度（ma_max=0.10 vs 0.05，检验压缩性偏差）；
* **主判据**：B 的 Richardson 外推 Nu 与基准相对误差 ≤5%（逐 Ra）；
  次判据：最细网格直接值 ≤5%；C 用于判定残余偏差是否来自压缩性。
* **停止规则**：阶梯到 Ny=320（Ra=1e5）/192（Ra=1e4）/64（Ra=1e3）为止，
  或 Richardson 外推与最细网格之差 <1% 且 |err|≤5% 即止（不再加密）。
* **混杂变量**：格子 Ma（压缩性误差 O(Ma²)）、τ_f→0.5 的 BGK 稳定性与
  边界层欠分辨、无量纲长度口径（本模块统一 H=ny，半程壁）、定常判定。
* **重复性**：LBM 时间推进是确定性过程；初场微扰用固定种子 12345，
  故同参数重跑逐位可复现（见 ``test_repro_guo2002_dhc.py`` 的位级断言）。

用法
----
    python scripts/repro_guo2002_dhc.py --ra 1e5 --ny 256 --out runs/.../ra1e5_ny256.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import rb_convection_validation as V  # noqa: E402

#: de Vahl Davis (1983) 平均 Nu（Pr=0.71，刚-刚方腔，DHC）
REF = V.DE_VAHL_DAVIS_NU


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--ma-max", type=float, default=0.10)
    ap.add_argument("--turnover", type=float, default=120.0)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=1_500_000)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--out", required=True, help="结果 JSON 路径")
    ap.add_argument("--fields", default="", help="末态场 .npz 路径（可选）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.dtype == "fp64" else torch.float32
    t0 = time.time()
    out = V.run_to_steady(
        case="dhc", ny=args.ny, ra=args.ra, pr=args.pr, ma_max=args.ma_max,
        dtype=dtype, turnover_target=args.turnover, min_steps=args.min_steps,
        max_steps=args.max_steps, report_every=args.report_every,
        verbose=args.verbose, return_fields=bool(args.fields),
    )
    fields = {k: out.pop(k) for k in ("T_field", "u_field", "v_field") if k in out}
    ref = REF[args.ra]
    out.update({
        "anchor": "Guo, Shi & Zheng 2002, Int. J. Numer. Meth. Fluids 39:325-342",
        "ref_source": "de Vahl Davis 1983, Int. J. Numer. Meth. Fluids 3:249-264",
        "nu_ref": ref, "err_pct": 100.0 * abs(out["nu"] - ref) / ref,
        "bias_pct": 100.0 * (out["nu"] - ref) / ref,
        "ra_effective_by_gap": args.ra * ((args.ny - 2) / args.ny) ** 3,
        "ma_max_target": args.ma_max, "dtype": args.dtype,
        "wall_s": round(time.time() - t0, 1),
    })
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if args.fields and fields:
        import numpy as np
        np.savez_compressed(args.fields, **{k: v.numpy() for k, v in fields.items()})
    print(f"[repro] Ra={args.ra:g} ny={args.ny} Nu={out['nu']:.5f} "
          f"ref={ref:.4f} err={out['err_pct']:.2f}% steps={out['steps']} "
          f"Ma={out['ma']:.3f} tau_f={out['tau_f']:.4f} wall={out['wall_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())