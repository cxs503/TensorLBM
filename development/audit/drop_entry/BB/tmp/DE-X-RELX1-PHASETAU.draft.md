# DE-X-RELX1-PHASETAU Draft · 两相弛豫设计官 Wave 1

## 已读
- drop_entry/BB/NEXT/TRACK_DIAG_W1.md, TRACK_EV_W1.md, TRACK_FORCE_W1.md, TRACK_TENS_W1.md（各波处方卡）
- drop_entry/BB/tmp/DE-NXT-DIAG-W1.draft.md, DE-NXT-EV-W1.draft.md, DE-NXT-FORCE-W1.draft.md, DE-NXT-TENS-W1.draft.md
- src/lbm/collision/d3q27.py（D3Q27 碰撞算子：BGK/MRT/Cumulant）

## 已得结论
- 定谳前提：峰深 1.4352D@11.091ms vs 实验 1.857434D@10.394ms（-22.73%）
- 网格/参数/u_limit/CAVSWEEP 全排除 → 气穴欠深=物理内核问题
- 本任务=RELX1-PHASETAU：界面胞(fill<1)用不同弛豫率 τ_iface=τ·g(fill)，液相胞保持物理τ
- 物理出处=液气界面有效动量输运异于体相（部分填充胞+气相边界）
- KERN-B 证伪 p_cav↓变浅提示界面闭包对弛豫敏感
- 这是结构(空间)分离非全局τ_s微调

## 下一步
1. 找到碰撞段代码中 fill 掩码/弛豫率应用位置
2. 实现界面胞修正弛豫率（fill 加权）
3. 跑验证（cells=8 满窗三元组同 ARM1）
4. 交付 RELX1-PHASETAU.md
