# MMC4-BENCH18 草稿（2026-11-21）
## 已读
- fig5_experiment.csv: 18 测点，Santini case-3，D=2.27mm, U=1.40m/s, Re=3715, We=62.1, Fr=88
- KERN-A-R2 result.json: 2017 steps, peak=1.3701D@22.947ms, window_complete 24.552ms
- 基线锚 ctrl=1.4352D@11.091ms (abs) / 10.543ms (pc)
- 实验 peak=1.857434D@10.394ms (pc) → 误差 -22.73%
- MMC3-RUNSDIFF.md: 时轴口径澄清 (abs vs pc, 差 0.547ms)
- KERN-A-R2.md: ABB Option3 证伪 (1.3701D 变浅)
- REF-WEB-01.md: Santini 2017 文献

## 已得
- 实验数据: 18 点 (t=0.39~17.40ms, depth=0.039~1.857D)
- 仿真数据: 2017 点 (t=0~24.55ms, depth=0~1.370D)
- KERN-A-R2 run.log: arm=KERN-A-R2 card=0 commit=badf6b70c steps=10000

## 下一步
1. 写 Python 脚本：插值仿真→实验时间点，算残差
2. 分段分解：早(0-3ms)/中(3-7ms)/晚(7ms+)
3. 生成残差图 + Markdown 报告
4. 回答：-22.7% 缺口集中在哪段？哪段已符合？
