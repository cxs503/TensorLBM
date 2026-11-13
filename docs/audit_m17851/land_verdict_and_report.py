import base64
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path("/root/tp-agent/work/m17851")
AUD = ROOT / "artifacts" / "m17851_audit"
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)
BASE = ENV["TENSOR_API_BASE"].rstrip("/")
W = "/root/tp-agent/work/m17720/src/tensorlbm/xval_cartesian_mpi"

evidence = {
    "verdict_basis": (
        "L2：真实现（8 模块 1316 行）+ CLI 入口可跑 + 6 组配置工件逐数 bit-identical 复现；"
        "但核心能力（2x2 分块 halo 交换）在交付默认配置下产生 25.35% 中心线误差（容差 15%），"
        "2882 步×2 长积分恶化到 36.5%，交付件自判 status=fail ⇒ XVAL 判据未过，不授 L3/L4/L5。"),
    "claimed": "XVAL-LBM-CARTESIAN-MPI 对标实现（result.check 文字自认 decomposed 25% error needs debugging）",
    "delivered_level": 2,
    "gap_to_claim": (
        "『对标通过』不成立：m17720/runs/xval_cartesian_mpi/result.json status=fail, "
        "decomp_umax_error_pct=25.3525 vs accept_threshold_pct=15.0（超 1.69 倍）；"
        "4000 步 36.5066%、256x64 25.1073% 说明非分辨率/瞬态；mpi4py 未安装，MPI 通信路径从未执行（单进程模拟）"),
    "impl_files": [
        W + "/runners.py",
        W + "/domain_decomp.py",
        W + "/campaign.py",
        W + "/metrics.py",
        W + "/case_spec.py",
        W + "/cli.py",
    ],
    "impl_notes": {
        "defect_site": W + "/runners.py:418-452 (_halo_exchange: y 面交换写整行，覆盖本轮 x 面已写好的角点)",
        "dead_code_defect": W + "/domain_decomp.py:198-226 (halo_slices send/recv 同址，全仓无调用)",
        "duplication": "src/tensorlbm/domain_decomposition.py（m3379/m368 既有 slab 分解 + HaloExchange）未复用",
        "dead_param": "case_spec.warmup_steps 在 runners 中从未读取；交付工件 warmup_steps=2000 > steps=1000",
    },
    "entry": W + "/cli.py",
    "entry_cmd": "PYTHONPATH=src python3 -m tensorlbm.xval_cartesian_mpi --out-dir <out> --nx 128 --ny 32 --steps 1000",
    "run_artifacts": [
        "/root/tp-agent/work/m17720/runs/xval_cartesian_mpi/result.json",
        "/root/tp-agent/work/m17720/runs/xval_cartesian_mpi/xval_table.csv",
        "/root/tp-agent/work/m17720/runs/xval_cartesian_mpi/xval_cartesian_mpi.png",
        str(AUD / "repro_matrix.json"),
        str(AUD / "fix_probe.json"),
        str(AUD / "fig_m17851.png"),
    ],
    "repro_cmd": ("cd /root/tp-agent/work/m17720 && PYTHONPATH=src python3 -m tensorlbm.xval_cartesian_mpi "
                  "--out-dir /tmp/m17851_repro_1000 --nx 128 --ny 32 --steps 1000 "
                  "# 复现交付数值 mono 0.2334% / decomp 25.3525%，status=fail（本次实跑逐数一致）"),
    "shipped_vs_reproduced": {
        "mono_umax_centreline": [0.04988328383918091, 0.04988328383918091],
        "decomp_umax_centreline": [0.03732374758250377, 0.03732374758250377],
        "identical": True,
    },
    "benchmark": {
        "registered_cases_for_anchor": 0,
        "tensor_benchmark_cases_total": 59,
        "tensor_xval_cases_rows_for_anchor": 0,
        "ref": "analytic Poiseuille u(y)=4*Umax/H^2*y*(H-y) (case_spec.py:105-119)",
        "shipped_status": "fail",
        "shipped_accept_threshold_pct": 15.0,
        "shipped_mono_umax_error_pct": 0.2334,
        "shipped_decomp_umax_error_pct": 25.3525,
        "shipped_decomp_l2_error_pct": 6.282,
        "margin_over_tolerance": 1.69,
        "ablation_2x2_pct": {"1x1": 0.2334, "1x2": 1.7038, "2x1": 0.2334, "2x2": 25.3525},
        "long_integration_4000_steps_pct": {"mono": 2.2378, "decomp": 36.5066},
        "mesh_256x64_1000_steps_pct": {"mono": 0.0405, "decomp": 25.1073},
        "interface_localization": {"max_col_mean_abs_err": 0.005322, "at_x": 63,
                                   "interior_col_mean_abs_err": 0.002436,
                                   "interior_pct_of_umax": 4.872, "max_abs_du_field": 0.01764},
        "root_cause_fix_probe": {"control_pct": 25.3525, "corner_aware_halo_pct": 0.2334,
                                 "delta_pct_points": 25.119, "script": "docs/audit_m17851/fix_probe.py"},
        "pass": False,
    },
    "tests": ["none: tests/ benchmarks/ docs/ examples/ 对 xval_cartesian_mpi 零命中（grep 0 命中）",
              "审计侧验证：6 组配置重跑 + 分解消融(1x1/1x2/2x1/2x2) + 长积分 + 网格加密 + 角点修复探针"],
    "integration_callers": "none: src/tensorlbm 其余模块零引用；git status 仅 '?? src/tensorlbm/xval_cartesian_mpi/'（未 commit、零 tracked 改动）",
    "external_anchor": ("pylbm/pylbm mpi_topology.py 实体源码在 /root/oss-cache/pylbm（存在，出处真实）；"
                        "但 mpi4py 缺失 → OSS/MPI 运行未执行，仅代码级移植"),
    "report_registration": "tensor_reports WHERE mission_id=17720 -> 0 行（m17720 result 记 API 401），报告纪律未达",
    "recommended_action": ("板面按 L2 记账同时开缺陷单：改 runners._halo_exchange 为角点感知交换"
                           "（面只写内部跨距 + 四对角角块交换），已验证 2x2 25.35%→0.2334%；"
                           "补 tests/ 与 REPORT.md，并把 xval_table 登记 tensor_xval_cases"),
}
ev = json.dumps(evidence, ensure_ascii=False)
chk = subprocess.run(["runuser","-u","postgres","--","psql","-d","tensorplatform","-tA","-c",
    "SELECT id FROM tensor_maturity_verdicts WHERE mission_id=17851 AND scope LIKE 'm17720%'"],
    capture_output=True, text=True).stdout.strip()
sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       "VALUES ('TensorLBM', 'm17720 XVAL-LBM-CARTESIAN-MPI', 17851, 2, $json$" + ev + "$json$, 'board-audit') "
       "RETURNING id;")
if chk:
    print("verdict already landed id=", chk)
else:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-c", sql],
                       capture_output=True, text=True)
    print("INSERT:", r.returncode, r.stdout.strip(), r.stderr.strip()[:300])

# login + report
def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as resp:
        return resp.status, json.loads(resp.read() or b"{}")

st, d = api("POST", "/auth/login",
            {"username": ENV["TENSOR_AGENT_USER"], "password": ENV["TENSOR_AGENT_PASS"]})
tok = d.get("access_token") or d.get("token")
print("login", st, bool(tok))

fig = AUD / "fig_m17851.png"
md = (ROOT / "docs/audit_m17851/AUDIT_m17851.md").read_text()
report = {
    "title": "m17851 交差审计：m17720 XVAL-LBM-CARTESIAN-MPI 判 L2，2x2 分解 25.35% 超容差（根因已定位并验证修复）",
    "summary": ("逐项核验 m17720：工件真值可 bit-identical 复现（非伪造），但交付件自判 status=fail，"
                "decomposed 中心线误差 25.35%（容差 15%），4000 步 36.5%、256x64 25.1% 不收敛；"
                "平台无登记基准/无报告/无测试/零集成调用。判级 L2；根因=runners._halo_exchange 角点被覆盖，"
                "角点感知交换后 25.35%→0.2334%。"),
    "scenario_key": "xval_audit",
    "mission_id": 17851,
    "tags": ["XVAL", "TensorLBM", "cartesian-MPI", "board-audit", "verdict", "L2", "halo-exchange"],
    "markdown": md,
    "figures": [{"name": "fig_m17851", "caption": "三联图：(a) 分解配置消融（仅 2x2 越 15% 线）；(b) 中心行 u_x 沿 x —— x 分块界面 x=64 与周期缝合列塌陷；(c) x 平均剖面 vs 解析解",
                 "png_base64": base64.b64encode(fig.read_bytes()).decode()}],
}
try:
    st, d = api("POST", "/tensor/reports", report, tok)
    print("report", st, str(d)[:300])
except Exception as e:
    print("report FAILED", type(e).__name__, str(e)[:300])
# judge 预演（不触发守护动作）：确认 gate 会判到 L2
import sys
sys.path.insert(0, "/root/tp-agent")
try:
    import maturity_gate as MG
    lvl, why = MG.judge(2, evidence, "TensorLBM", 17851)
    print("gate_dry_run judge(2) ->", lvl, why)
except Exception as e:
    print("gate dry run skipped:", type(e).__name__, str(e)[:120])
