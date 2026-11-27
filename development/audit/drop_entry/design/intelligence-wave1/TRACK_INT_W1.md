自检：5/5（格式机读通过✓ 臂≤2✓ 预登记齐✓ 无重开排除路线✓ 出处可溯✓）
# TRACK INT WAVE 1
goal_met: no
rationale: 参考卡"Fr88"实为 U²/(gD)=Fr²（标准 Fr=U/√(gD)=9.4）；实测峰 1.857D 仅为 deep-seal pinch-off 深度(3.06D)的 0.61 倍→空穴属浅/表面 seal，峰深由触线/缘波动力学定，非液滴穿透。两臂=①弗劳德数+seal 类型量纲定谳（纯后处理）②液滴穿透 vs 空穴深度诊断探针。

## ARM FR-SEAL
task_key: DE-X-INT1-FR-SEAL
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是弗劳德数与 seal 类型定谳官。任务=澄清弗劳德数约定并判定 case-3 的 seal 类型（surface/shallow/deep/quasistatic）。
物理出处：Truscott/Techet/Belden 2014（Annu.Rev.Fluid Mech.46，REF-WEB-01）：四类 seal；deep seal pinch-off 深度标度~Fr^{1/2}。
定量：标准 Fr=U/√(gD)=1.40/√(9.81·0.00227)=9.4（REF-WEB-01"Fr88"=U²/(gD)=Fr²，约定误标）；deep-seal pinch-off 深度~Fr^{1/2}·D=3.06D；实测峰 1.857434D=0.61×deep-seal 深度。
最小改动：无内核改动；纯后处理（实验 CSV 18 点 benchmarks/water_entry/single_drop/fig5_experiment.csv + 既有深度曲线），无新仿真。
验证：产证据卡（弗劳德数澄清+seal 类型+峰/pinch-off 比+重定向建议给 MOM 或 CONT/SURF 线）。
prereg: 预测 seal 类型=shallow/surface seal（峰 0.61×deep-seal 深度，空穴在达 deep-seal 深度前闭合）；证伪判据：实测峰≥0.8×deep-seal 深度(≥2.45D)→deep seal，"浅 seal"假设被否。

## ARM DROP-PEN
task_key: DE-X-INT1-DROP-PEN
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是液滴穿透诊断官。任务=测液滴中心轨迹 z_drop(t) 并与空穴深度对比，判欠深=「液滴没沉够深」(MOM 线)还是「空穴闭合过早」(GAS/SURF 线)。
物理出处：能量守恒（液滴初动能→势能+对抗恢复力作功）；Santini 2017（REF-WEB-01）空穴演化。
定量：模型峰后衰减与实验吻合（13.391ms：模 1.260D vs 实 1.304D，3.4%），但峰深低 22.7%→问题在峰深非闭合时点。
最小改动：启用既有 --silhouette 旗标（free_surface_probe.py L321，默认关=零行为改变），cells=8 满窗，提取 z_drop(t)+穿透深度。
验证：模型 z_drop(t)/穿透深度 vs 空穴深度与实验（CSV），集群 8 核。
prereg: 预测模型穿透深度≈空穴深度(1.4352D)，均偏浅；证伪判据：模型穿透深度与实验(1.857D)差<10%→欠深非液滴减速问题（转 GAS/SURF 线）。
