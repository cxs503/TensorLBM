#!/usr/bin/env python3
"""本地模拟 maturity_gate.judge() 对本单 evidence 的机器复核（adversarial self-check）。

直接 import 守护进程同款函数（/root/tp-agent/maturity_gate.py），audit_mid=17735（verdict.mission_id），
repo=TensorLBM，level=4。预期 gate_level=4 → confirmed。
"""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("maturity_gate", "/root/tp-agent/maturity_gate.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

sys.path.insert(0, "/root/tp-agent/work/m17735/docs/board_audit/m17735")
import insert_verdict as iv  # noqa: E402  (只取 evidence，不执行落表——见下)

ev = iv.evidence
gl, why = mg.judge(iv.LEVEL, ev, "TensorLBM", 17735)
print("claimed_level:", iv.LEVEL)
print("gate_level:", gl)
print("why:", why)
print("PASS" if gl == iv.LEVEL else "MISMATCH")
