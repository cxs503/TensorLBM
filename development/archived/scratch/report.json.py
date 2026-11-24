import base64, json, os, urllib.request, pathlib
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
U, P = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]

def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()

st, body = post(BASE + "/auth/login", {"username": U, "password": P})
tok = json.loads(body)["access_token"]

md = pathlib.Path("benchmarks/xval_akai2018_wetting_rotation/REPORT.md").read_text()
png = pathlib.Path("benchmarks/xval_akai2018_wetting_rotation/fig_estimator_and_round4.png")
payload = {
    "title": "XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING R6：接触角量测门与旋转润湿路径",
    "summary": ("色梯度坐滴接触角验证的量测门：81 例解析圆冠上帽反演中位误差 0.45°/p95 1.69°，"
                "证明 R4 用作判定量的圆拟合结构性失效（θ=30° 得 141–146°，θ=120° 得补角 57.6°）；"
                "用新判定量重测 R4 存档场：中性壁 −5.0%、θ=90° −3.9% 通过，θ=30° +17.0%，"
                "θ=60° 检出卫星液滴（液体团簇 [66,55]）。旋转润湿（Akai/MF-LBM 保模长写法）收敛扫描已递交 cj4251 但排队未跑，不作端到端达标声明。"),
    "scenario_key": "xval_lbm_wetting_contact_angle",
    "mission_id": 22599,
    "tags": ["TensorLBM", "XVAL", "wetting", "contact-angle", "Akai2018", "MF-LBM",
             "color-gradient", "measurement-gate", "honest-partial"],
    "markdown": md,
    "figures": [{
        "name": "fig_estimator_and_round4.png",
        "caption": ("左：解析 tanh 圆冠（无边界条件）上 theta_cap（帽面积+接触宽度反演）与圆拟合的误差 vs 规定角；"
                    "圆拟合在 θ<90° 与 θ>90° 两侧分别给出伪像与补角。右：R4 存档相位场用新判定量重测 vs 目标，"
                    "θ=60° 臂标注检出的卫星液滴。"),
        "png_base64": base64.b64encode(png.read_bytes()).decode(),
    }],
}
st, body = post(BASE + "/tensor/reports", payload, tok)
print("POST /tensor/reports", st, body[:300])
