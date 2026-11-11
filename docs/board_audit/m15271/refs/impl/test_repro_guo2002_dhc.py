"""REPRO-LBM-GUO2002: 复现管线（DHC 平均 Nu 对表）的 pytest 用例。

覆盖：
1. ``run_to_steady`` 新增 ``return_fields`` 钩子（等温线后处理所需）——
   默认关闭时返回值必须仍可 JSON 序列化（不破坏既有调用方）；
2. DHC 短跑观测量自洽：壁面通量 Nu 与能量平衡 Nu 一致、"两侧通量不平衡"
   小、Nu 落在物理范围内（1 ≤ Nu ≤ 1.2 @ Ra=1e3 粗网格）；
3. 复现确定性：固定初场种子 ⇒ 同参数两次运行**逐位**相同（复现实验的可
   重跑性，不是"差不多"）；
4. 外推器（``scripts/repro_guo2002_analysis.richardson``）在合成数据上的
   正确性：二阶收敛的抛物/幂律序列必须给出正确观测阶与外推值。

重档网格阶梯由 ``scripts/repro_guo2002_ladder.py`` 在集群跑，产出入
``runs/repro_guo2002_dhc/``；本文件只保留分钟级可复现判据。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensorlbm.rb_convection_validation import run_to_steady

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _short(**kw):
    base = dict(case="dhc", ny=16, ra=1e3, pr=0.71, ma_max=0.10,
                turnover_target=1e9, min_steps=1500, max_steps=1500,
                report_every=500, dtype=torch.float64)
    base.update(kw)
    return run_to_steady(**base)


class TestFieldHook:
    def test_default_is_json_serializable(self):
        out = _short()
        assert "T_field" not in out
        json.dumps(out)                       # 不许有不可序列化对象

    def test_return_fields_shapes_and_range(self):
        out = _short(return_fields=True)
        T = out.pop("T_field")
        u = out.pop("u_field")
        v = out.pop("v_field")
        assert T.shape == (16, 16) and u.shape == T.shape and v.shape == T.shape
        assert float(T.min()) > -0.1 and float(T.max()) < 1.1
        json.dumps(out)                       # 取走场之后仍可序列化


class TestDhcSelfConsistency:
    def test_nusselt_estimators_agree(self):
        """壁面通量 Nu 与能量平衡 Nu 定常时必须一致（相对偏差 <2%）。"""
        out = _short()
        assert 1.0 <= out["nu"] <= 1.25, out["nu"]
        rel = abs(out["nu"] - out["nu_energy"]) / out["nu"]
        assert rel < 0.02, (out["nu"], out["nu_energy"])
        assert out["nu_imbalance"] < 0.02, out["nu_imbalance"]

    def test_deterministic_bitwise(self):
        a = _short()
        b = _short()
        assert a["steps"] == b["steps"]
        assert a["nu"] == b["nu"]             # 逐位相同（固定种子 12345）


class TestRichardson:
    def _mod(self):
        spec = importlib.util.spec_from_file_location(
            "repro_analysis", ROOT / "scripts" / "repro_guo2002_analysis.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)            # type: ignore[union-attr]
        return m

    def test_recovers_second_order_limit(self):
        m = self._mod()
        # Nu(h) = 2.0 - 0.5 h²，h = 1/8, 1/16, 1/32（与网格比 r=2 一致）
        hs = [1 / 32, 1 / 16, 1 / 8]
        ys = [2.0 - 0.5 * h * h for h in hs]
        out = m.richardson(ys, hs)
        assert out["ok"]
        assert abs(out["p_observed"] - 2.0) < 1e-6
        assert abs(out["nu_extrapolated"] - 2.0) < 1e-6

    def test_non_monotone_flagged(self):
        m = self._mod()
        out = m.richardson([1.0, 1.1, 1.05], [1 / 32, 1 / 16, 1 / 8])
        assert not out["ok"]