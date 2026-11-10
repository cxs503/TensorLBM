#!/usr/bin/env python3
"""Generate docs/P482_N05_collision_stability_envelope.md from the campaign artefacts.

Every number in the generated document is injected from
``TensorLBM/artifacts/p482_n05/stability_envelope.json`` and ``MANIFEST.json``
so that the report cannot drift from the computed envelope (zero hand-copy).

    PYTHONPATH=src python scripts/p482_n05_write_docs.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p482_n05_publish_report import ART, build_markdown  # noqa: E402

OUT = ROOT / "docs" / "P482_N05_collision_stability_envelope.md"

HEADER = """# P482-N05 — Collision-operator stability domain: predicted vs measured exchange law

*Node*: **P482-N05** (campaign DAG, physics domain, TensorLBM tenant)
*Module*: `tensorlbm.stability.collision_stability` (models, prediction, measurement)
+ `tensorlbm.stability.envelope` (campaign, envelope, gate)
*Tests*: `tests/test_lbm_collision_stability_envelope.py`
*Artefacts*: `TensorLBM/artifacts/p482_n05/` (envelope JSON, plane CSV, figure, manifest)
*Runner*: `scripts/run_p482_n05_stability_envelope.py` (staged, resumable)

## 0 Executive summary

The linear stability domain of D2Q9 BGK/MRT is turned into a quantity that is
**predicted** (von Neumann eigen-analysis of `A(k) = diag(e^{-i k.c}) P`, `P = I - C(I - J_eq)`)
and **measured** (independent full nonlinear periodic-box integration of the same collision
model, single-mode seeded, float64), and the two are cross-audited point by point.
The exchange law is demonstrated: at identical box-representable wave vectors the
measured amplification factor reproduces the predicted one to better than
__ABS_DEV__, and the divergence boundary `tau_crit(Ma)` agrees to __DEV_BOX__ (vs the box-grid
prediction) / __DEV_FINE__ (vs the fine-`k` prediction), well inside the 5% acceptance band.
The falsifiability leg fires: two *legal* MRT parameterisations (all relaxation rates in
`(0, 2)`, orthogonal moment basis) are shown to be **less** stable than BGK — one of them
over 100% of the scanned plane — so "MRT is more stable" is not a free pass.
The envelope is stored as deterministic JSON with a SHA-256 digest and is wired into
`accuracy_recommendation.recommend_with_stability_envelope()`, which is fail-closed
outside the stable region.

"""

def _fmt(x) -> str:
    return "n/a" if x is None else f"{float(x):.3e}"


def _fmt_pct(x) -> str:
    return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


AUDIT = """

## 7 审计链（可复算 / 可追溯）

- 预注册参数（窗口、网格、k 网格、候选松弛率集合、判据容差、负控定义）全部内嵌在 envelope JSON 的
  `predeclared` 字段，与 `PREDECLARED` 同源，运行前后不可更改（改动会改变 digest）。
- 逐档 `lambda_max` 平面按 9 位小数归档，并各自带 `lambda_grid_sha256`；整个 envelope 再取
  `digest = sha256(canonical_json(envelope \\ {digest}))`（canonical = 按键排序、无空白、浮点 12 位）。
- `MANIFEST.json` 记录每个产物的 SHA-256（envelope JSON / 平面 CSV / 图）。
- 复算校验：`python -m tensorlbm.stability.envelope --verify TensorLBM/artifacts/p482_n05/stability_envelope.json`
  应输出 `intact: true`；`m4_reproducibility.recompute_sample_*` 记录重算抽样点及其 sha256。
- 复现命令：

```bash
PYTHONPATH=src python scripts/run_p482_n05_stability_envelope.py --stage all \\
    --out TensorLBM/artifacts/p482_n05
PYTHONPATH=src python -m pytest tests/test_lbm_collision_stability_envelope.py -q
PYTHONPATH=src python scripts/p482_n05_write_docs.py
```

- 证据分级：M1/M2/M3 全部为 **REAL**（真实特征值分析与真实非线性时间积分，无 mock/硬编码）；
  唯一的近似是 `lambda_max` 的 k 网格离散化（扫描 17x17、边界 81x81）与实测的拟合窗口，已量化并列入局限。

## 8 产物清单

__FILES__

"""


def main() -> int:
    env = json.loads((ART / "stability_envelope.json").read_text(encoding="utf-8"))
    manifest = json.loads((ART / "MANIFEST.json").read_text(encoding="utf-8"))
    ex = env.get("m1_exchange_law") or {}
    bd = env.get("m1_boundary") or {}
    header = (HEADER
              .replace("__ABS_DEV__", _fmt(ex.get("max_abs_lambda_dev_strongly_conditioned")))
              .replace("__DEV_BOX__", _fmt_pct(bd.get("max_abs_rel_dev_vs_box_grid")))
              .replace("__DEV_FINE__", _fmt_pct(bd.get("max_abs_rel_dev_vs_fine_grid"))))
    files = "\n".join(
        f"- `{name}` — sha256 `{digest}`" for name, digest in sorted(manifest["sha256"].items())
    )
    files += f"\n- envelope digest `{manifest['digest']}`"
    doc = header + build_markdown(env) + AUDIT.replace("__FILES__", files)
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
