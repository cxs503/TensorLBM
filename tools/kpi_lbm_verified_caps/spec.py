#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KPI-LBM-VERIFIED-CAPS 计量口径（冻结 spec，只测量不改进）。

平台总账锚（tensorplatform.feature_ledger#176, slug=KPI-LBM-VERIFIED-CAPS）
  标题 : 周新增通过独立物理验证的能力数
  验收 : 【成果指标令｜计数=本周 gate confirmed L3+ 的物理能力锚新增数；不以文件/代码/PR数
          计成果；周计量走 kpi_meter】

这段文字是本 spec 的唯一输入。把文字落成机器可判的取数规则（不引入任何新数值目标）：

  单位 unit        = 「锚 (anchor)」——不是文件/代码/PR/任务单数。
  归属 repo        = tensor_maturity_verdicts.repo = 'TensorLBM'
  准入状态          = gate_state = 'confirmed'
                     （reopened / pending / unmapped / invalid / 空 一律不计：reopened 是审计判「不
                      达标待返修」，正是 gate 未确认的形态）
  准入门槛          = gate_level >= 3
                     （平台成熟度闸门：L1 交付+L1 数学验证 / L2 真实现 / L3 真集成 /
                       L4 机器裁判+收敛或对标 / L5 外部独立复现）
  物理锚判定        = anchor slug 不属于「基础管道/工程化」类（PHYSICS_EXCLUDE_KEYWORDS）；
                     命中者计入 sensitivity（all_anchors），不计入 primary。
     依据：KPI 文字明写「物理能力锚」；纯 XML/YML 模板、作业状态机、checkpoint、pytest 基线、
     数据布局模式、MPI 管道、kernel 选择器等锚点验证的是工程管道而非物理/数值能力。
     **该分类是规则化的、可逐条复核的（每条锚的命中理由落盘 data/anchors.tsv），非人工挑选。**
  新增判定          = 该 anchor 的 first_confirmed_at（窗口内最早一条 confirmed L3+ verdict 的
                     created_at）落在计量窗口内。同一 anchor 的多行 verdict 只计 1 次（取
                     max(gate_level)）。
  窗口 primary      = ISO 周（周一 00:00 UTC）→ 采集时刻 now
  窗口 sensitivity  = 滚动 7 天（now-7d → now）
  strict / loose    = loose  : 窗口内有 confirmed L3+ verdict 的 anchor
                     strict : loose ∩ 产出该锚的被审任务（evidence.audited_mission，缺省取
                              verdict.mission_id）finished_at 亦在窗口内
                              （= 本周新交付；排除「历史交付本周首次补审」的记账式增长）

  target            = 验收文字未给数值目标 → target=None，target_status='NOT_CALIBRATED'。
                      spec 不臆造目标数字；缺口只给「同一口径下的可提升口径」与派生锚建议。

冻结：任何数字都由 spec+snapshot 重算得出；spec_sha256 与 snapshot_sha256 一并入账。
"""
from __future__ import annotations

import hashlib
import json

SPEC_VERSION = "1"
ANCHOR_SLUG = "KPI-LBM-VERIFIED-CAPS"
FEATURE_LEDGER_ID = 176
UNIT = "anchors/week (physics capability anchors newly gate-confirmed at L3 or above)"
REPO = "TensorLBM"
CREDIT_GATE_STATE = "confirmed"
CREDIT_MIN_LEVEL = 3
LEVEL_MEANING = {
    1: "L1 delivery + L1 math verification",
    2: "L2 real implementation (input->compute->output)",
    3: "L3 real integration (a real caller consumes the capability)",
    4: "L4 machine referee + convergence/cross-validation",
    5: "L5 external independent reproduction",
}
# 非物理（工程管道/模板/作业/布局）锚的排除关键词，逐条命中理由写入 data/anchors.tsv
PHYSICS_EXCLUDE_KEYWORDS = (
    "TEMPLATE",
    "PIPELINE",
    "JOB-STATUS",
    "CHECKPOINT",
    "PYTEST",
    "LAYOUT",
    "XML",
    "YML",
    "YAML",
    "MACHINE",
    "SURVEY-EMPTY",
    "TAG-COMPRESSION",
    "KERNEL-TYPE-SELECTOR",
    "FUSED-KERNEL",
    "MPI",
)


def spec_dict() -> dict:
    return {
        "anchor_slug": ANCHOR_SLUG,
        "feature_ledger_id": FEATURE_LEDGER_ID,
        "spec_version": SPEC_VERSION,
        "unit": UNIT,
        "repo": REPO,
        "credit_gate_state": CREDIT_GATE_STATE,
        "credit_min_level": CREDIT_MIN_LEVEL,
        "level_meaning": {str(k): v for k, v in LEVEL_MEANING.items()},
        "physics_exclude_keywords": list(PHYSICS_EXCLUDE_KEYWORDS),
        "windows": ["iso_week_utc:primary", "rolling_7d:sensitivity"],
        "new_anchor_rule": "first_confirmed_at within window; dedupe by anchor (max gate_level)",
        "target": None,
        "target_status": "NOT_CALIBRATED",
    }


def spec_sha256() -> str:
    blob = json.dumps(spec_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()