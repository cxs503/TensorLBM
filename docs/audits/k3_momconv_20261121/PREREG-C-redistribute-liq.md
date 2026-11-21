# PREREG-C · K3-MOMCONV-C：`redistribute_liq` 变体（跑前预登记）

> 本文件在**该臂派车之前**写入；PREREG.md（三臂主赛）与 PREREG-B（inherit 修复臂）**一字未改**。

## 1. 为什么需要这一臂（主赛的实测结果，不是事后改门）

主赛 `redistribute` 臂跑完 2020 步（window_complete），其 history 与 ctrl **逐字段零差异**
（2020 行 × 9 字段，最大绝对差 0.0；峰深同为 1.435220474823 D @ 11.090571429 ms）。
根因（代码级）：该臂的接收胞集合是**纯 bulk LIQUID 胞**，而本工况里每一次 I→G 转换都发生在
界面层（转换胞的邻居全是 INTERFACE 胞），**bulk LIQUID 接收胞集合恒为空** ⇒ 动量从未被交付，
该臂是位级 no-op。

`redistribute_liq` 变体把接收胞集合改为**整个液相**（bulk LIQUID + INTERFACE 胞），权重仍为
受追踪质量份额（任务书「动量按质量比重分配给邻液胞」的字面读法：液相胞 = 持液相质量的胞）。

## 2. 臂定义

| 臂 | momconv | 接收胞集合 | 权重 | 步数 |
|---|---|---|---|---|
| k3_redistribute_liq | `redistribute_liq` | LIQUID + INTERFACE（除转换胞） | 受追踪质量份额 | 1200（覆盖深度峰步 912 与判决时刻 10.4 ms = 步 854） |

其余参数与主赛完全相同：cells = 8，gap = 0.18 D，lattice speed = 0.06，release = 1.53 m/s，
pool depth = 3.0 D，u_limit = 0.15，capacity + simple，SDAA，`--diag-budget`。

## 3. 判决门（与主赛同一套门，不放宽）

- **命中**：峰深相对实验 1.857434 D 的 |err| < 12% 且存活 > 10.4 ms；
- **证伪**：|err| > 15%，或峰深 ≤ ctrl 1.435220474823 D（更浅），或早死（存活 < 10.4 ms）。

## 4. 声明

- 本臂是**解释性补充**，不改变主赛判决；若它早死，则「把被转动量交回液相」在本模型上不可用，
  与 `inherit`（等分界面注入，早死）共同构成对处方的 honest FAIL。
