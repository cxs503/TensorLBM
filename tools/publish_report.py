#!/usr/bin/env python3
"""Publish the CASE-KARMAN-RE100 report to the platform (POST $TENSOR_API_BASE/tensor/reports).

If the simulation artifacts are missing (run still in flight / honest FAIL), a
minimal status report is published instead — never a fabricated result.
"""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
ROOT = Path(__file__).resolve().parents[1]


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read().decode()


def login():
    _, body = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                   "password": os.environ["TENSOR_AGENT_PASS"]})
    d = json.loads(body if body.strip().startswith("{") else "{}")
    return d.get("access_token")


def fallback_payload():
    runs = sorted((ROOT / "runs/karman_re100").glob("result_D*.json"))
    logs = sorted((ROOT / "runs/karman_re100").glob("log_D*.txt"))
    md = f"""# CASE-KARMAN-RE100（状态报告）

本轮**未产出可判定的数值结果**：全尺寸算例（D=48，1920² 网格、60000 步）在控制节点实测
2.48 s/步（≈41 h），集群作业 cj1114 排队 >300，故降尺寸到 D=16/20/24（域 20D×14D、24D×20D）
在控制节点后台重跑；报告生成时完成的档位数为 **{len(runs)}**。

## 已完成的工程件

- `benchmarks/case_pool/karman_re100/run_case.py`：算例入口（库原语链 collide_mrt → stream →
  compute_obstacle_forces(Ladd) → far_field_bc_2d + 海绵层 + 入口两周期播种），Cd/St 归约与仓内
  已验证案例同口径；一条命令可复现。
- `benchmarks/case_pool/karman_re100/make_report.py`：参考值机器解析（arXiv:1504.03949v2 第 7 页
  文献表，PDF sha256 存档）+ 对照表 + 图 + 六段报告生成。
- `artifacts/karman_re100/refs/`：参考值溯源自证（PDF + provenance.json）。

## 失败模式登记

- 失败阶段：`compute`（控制节点 CPU 吞吐不足 / 集群排队）——非模型问题。
- 生态摩擦：一手实验文献全被出版商墙（AIP/Elsevier/Cambridge 403，S2 API 429），参考锚值需由
  开放获取文献对照表间接取得；仓内已验证案例入口默认 `--device cuda:0`，无 GPU 时不自动回退。

复现命令见 `benchmarks/case_pool/karman_re100/README.md`（若已生成）。
"""
    return {
        "title": "CASE-KARMAN-RE100：圆柱卡门涡街 Re=100（状态/未完成报告）",
        "summary": "TensorLBM 圆柱 Re=100 算例池条目：本轮因控制节点 CPU 吞吐与集群排队未跑完，"
                   "工程件与参考溯源已入库，数值判定留待接棒轮。",
        "scenario_key": "CASE-KARMAN-RE100",
        "mission_id": 15407,
        "tags": ["TensorLBM", "case_pool", "cylinder", "Re100", "honest-FAIL"],
        "markdown": md,
        "figures": [],
    }


def main():
    payload_path = ROOT / "artifacts/karman_re100/report_payload.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text())
        kind = "result"
    else:
        payload = fallback_payload()
        kind = "status(fallback)"
    try:
        tok = login()
        if not tok:
            print("LOGIN FAILED")
            return 1
        st, body = post("/tensor/reports", payload, token=tok)
        print(f"published {kind}: status={st} resp={body[:300]}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"PUBLISH FAILED ({kind}): {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())