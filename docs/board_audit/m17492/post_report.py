import base64, json, os, urllib.request, urllib.error
from pathlib import Path

AUD = Path('/root/tp-agent/work/m17492/docs/board_audit/m17492')
BASE = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8010/api/v1')
USER = os.environ['TENSOR_AGENT_USER']; PWD = os.environ['TENSOR_AGENT_PASS']


def api(path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method='POST' if payload is not None else 'GET')
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', 'Bearer ' + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=180) as r:
            return r.status, json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:1500]


st, d = api('/auth/login', {'username': USER, 'password': PWD})
tok = d.get('access_token') if isinstance(d, dict) else None
print('login', st, bool(tok))

md = (AUD / 'AUDIT_m17450.md').read_text()
figs = json.loads((AUD / 'figs_b64.json').read_text())
fig_payload = [
    {'name': 'fig1_profile_H40.png',
     'caption': 'H=40 标量 Couette 剖面：m17450 交付核（与 equilibrium 湿节点逐位相同）vs 按 film_antibb.cpp 重导出的 OSS ABB（与报告参考解机器零）vs 两套参考解；右图显示偏差是纯斜率/有效壁面位置偏移',
     'png_base64': figs['fig1_profile_H40.png']},
    {'name': 'fig2_convergence.png',
     'caption': '双对数误差-分辨率：交付核 vs 节点参考（阶 0.83→0.44）、vs 报告半格参考（阶 1.06 = 参考错配伪阶）、OSS ABB（机器零）',
     'png_base64': figs['fig2_convergence.png']},
    {'name': 'fig3_kernel_identity.png',
     'caption': '核函数身份：|交付 − equilibrium|=0（三个网格全部）vs |交付 − OSS ABB|；右图壁面首流体行浓度随 1/H 线性 → 单侧 ≈0.28 格子壁面内移',
     'png_base64': figs['fig3_kernel_identity.png']},
]

summary = ('交差审计 m17450（XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE）判 L1：交付的 inamuro_wetnode 核与 '
           'equilibrium 湿节点逐位相同（max diff 0.0e+00 @H=20/40/80），OSS ABB（film_antibb.cpp 写法）在同一循环里 '
           '可达 0.0000% 误差却未实现；被引 film_inamuro.cpp 实为 Inamuro 质量拟合 BC；报告参考解与壁面约定错配使 '
           '1.45%/0.70% 与"阶≈1.05"成为伪值；tensor_benchmark_cases 该锚点 0 行、tensor_reports mission 17450 0 行、'
           '分支未并入 main → reopen 返修。')

body = md + '\n\n---\n\n### 复核命令\n```\ncd docs/board_audit/m17492\nOMP_NUM_THREADS=1 python3 audit_xval_abb.py\nOMP_NUM_THREADS=1 python3 audit_p6_kernel_swap.py\nOMP_NUM_THREADS=1 python3 make_figs.py\n```\n'

payload = {
    'title': '交差审计 m17450：XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE 判 L1（交付核非 ABB，OSS 锚误引，参考解错配，返修）',
    'summary': summary,
    'scenario_key': 'board-audit/xval-lbm-abb-inamuro-wetnode/m17450',
    'mission_id': 17492,
    'tags': ['board-audit', 'TensorLBM', 'XVAL', 'LBM', 'anti-bounce-back', 'm17450', 'reopened', 'L1'],
    'markdown': body,
    'figures': fig_payload,
}
st, d = api('/tensor/reports', payload, tok)
print('post', st, str(d)[:400])