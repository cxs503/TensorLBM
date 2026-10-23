# benchmarks/archived/ — 冻结战役证据区（append-only）

本区存放多相流引擎战役的冻结记录（X campaign 2026-10-04/05 + XH hybrid
2026-10-06）。内容为 append-only 证据归档，**不是**晋级的 benchmark 案例：
不进 verified/pending 案例注册、不进 CI、不进包 import 路径。md5 冻结工件
逐字节保存、永不重排（含未过 lint 门的源文件——它们仅以 sha256 记录在
per-track MANIFEST.md，原件留在暂存树）。出处与逐文件 sha256 证明见各
track 的 MANIFEST.md / MANIFEST_evidence.csv，终态判决见各 track 的
CLOSURE.md（xh_hybrid 为 README.md，只有域声明无总判）。

- x_campaign_20261004/xa_rehab/ — C1-G2 晋级包 4/4 PASS、XA-2..XA-5 康复臂、X1 跨模型归因电池（RC=0 定稿）
- x_campaign_20261004/xb_engine/ — XB-1 iso4 电池（A1/A3/A4 honest FAIL）、XB-2 多量程电池 v3（总判 FAIL）
- x_campaign_20261004/xc_freesurface/ — W1 壁包（已验证库资产）+ 溃坝 Z 结构性 FAIL 定案 + τ 阶梯诊断
- xh_hybrid_20261006/ — XH-1 混合 CG×CAC 引擎（证据+复活种子；已验证域=低密度比 λ=2.0 族）
