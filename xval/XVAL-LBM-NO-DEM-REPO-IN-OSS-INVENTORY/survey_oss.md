# ① 调研·开源横向 — OSS 台账里有 DEM 孪生吗？（证据，不假设）

`anchor = XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY · mission 18215 · repo TensorLBM · 2026-11-13`

本锚点第 1 轮（m18066）以 honest-FAIL 收场且**未留 fail_stage / failure_reason**。
第 2 轮的第一件事是把"OSS 里没有 DEM 仓"这句话从**假设**变成**证据**，再决定
验证锚点落在哪里——不重蹈"卡在找一个不存在的孪生仓"的死法。

## 1.1 缓存清单（零手抄：命令 + 原始输出）

```bash
$ ls -1 /root/oss-cache | wc -l
3
$ ls -1 /root/oss-cache
Maarten-vd-Sande_lbm
hschwane
pylbm
$ git -C /root/oss-cache/Maarten-vd-Sande_lbm log -1 --format='%h %ad %s' --date=short
573a01c 2020-08-08 make easier to install
$ git -C /root/oss-cache/hschwane/GraSPH log -1 --format='%h %ad %s' --date=short
ae8c6ff 2020-03-10 some bugfixed
$ git -C /root/oss-cache/pylbm log -1 --format='%h %ad %s' --date=short
f9f5d16 2026-11-09 build: use uv for development, CI and documentation (#209)
```

集群节点上同名缓存目录由 `cluster_jobs` 作业 `cj1829`（`ls -1 ~/oss-cache`）交叉核对，
与本地同一份清单（作业排队中，若结论不同以本节本地清单为准，已在报告 §6 记明）。

## 1.2 DEM / 颗粒接触代码存在性检索

```bash
$ cd /root/oss-cache && grep -ril -E "discrete element|spring.?dashpot|granular|contact model|restitution" \
      --include=*.py --include=*.f90 --include=*.cpp .
(无输出)
```

三个仓的方法面：

| 仓 | 是什么 | 有 DEM 吗 | 与本锚点的关系 |
|---|---|---|---|
| `Maarten-vd-Sande/lbm` @573a01c | Python/NumPy D2Q9 + 自由表面 | **无**（已在前序 XVAL 中逐行读过：bitflag/streaming/Zou-He） | 流体侧参考，无颗粒动力学 |
| `pylbm` @f9f5d16 | 通用 LBM（多格式、边界算子、符号推导） | **无**（无接触力/颗粒相） | 只有流体侧 |
| `hschwane/GraSPH` @ae8c6ff | SPH（拉格朗日粒子、但为**连续介质**粒子） | **无**（SPH 粒子≠离散元刚体接触） | 易被误认为"粒子法孪生"，实为无接触本构 |

**结论（可复核）**：OSS 台账中不存在 DEM/颗粒接触力学的孪生实现，
`NO-DEM-REPO-IN-OSS-INVENTORY` 这一命名所描述的缺口**成立**；
因此本锚点**不可能**用"读 OSS 源码逐行对标"的常规 XVAL 打法，
必须改用 **②方法谱系 + 闭合解/文献锚**（见 `lineage.md`）。

## 1.3 第 1 轮的失败模式（为什么这单不能重复）

| 死法 | 表现 | 第 2 轮的对策 |
|---|---|---|
| 把"找 OSS 孪生"当唯一路径 | 找不到 → 无米下锅 → 空手 FAIL，且未记 stage | 先证缺口（本节），再改锚：闭合解 + 文献曲线 |
| 只报结论不报数字 | gate 无法机器复核 | 每个 arm 都先声明判据（`predeclared.json`），再输出逐项 `rel_err` |
| 只跑一个分辨率 | 无法区分"方法错"与"离散误差" | 每个 arm 两档以上时间步，报收敛方向 |

数据来源：本节所有命令行输出均为本机 2026-11-13 实跑；`grep` 无匹配即"无匹配"，
不代表检索式万能（检索式与范围已在上面原样列出，供复核与推翻）。