# 3D 矩形管 Poiseuille 流（D3Q19）— 双奇正弦级数解析验证（Shah-London）

Wave-3（W3-B）交付。方形 1:1（W=32/64/128）与 2:1 矩形（W=64/128，H=W/2）两组
网格阶梯，直接模拟 vs **自推解析解**（矩形截面 Poiseuille 的双重奇正弦级数），
Shah-London 摩擦因子常数作独立交叉。真实模拟、无外推、无修正因子
（`extrap: "none"`）。

## 解析解（自推，无文献数据依赖）

矩形截面 $[-a,a]\times[-b,b]$ 上 $u$ 满足 $\nu\nabla^2 u = -G$、壁面无滑移：

$$u(y,z) = \frac{16G}{\nu\pi^2}\sum_{m\,\text{odd}}\sum_{n\,\text{odd}}
\frac{\sin\frac{m\pi(y+a)}{2a}\,\sin\frac{n\pi(z+b)}{2b}}
{mn\left[\left(\frac{m\pi}{2a}\right)^2+\left(\frac{n\pi}{2b}\right)^2\right]}$$

独立交叉（书表常数，知识域核对）：方形 $Po = f\cdot Re = 56.9083$、
2:1（H=W/2）$Po = 62.1922$（Shah & London 1978）。

## 判定方法（NOTES.md 预注册）

1. **CH1 主通道 = 全剖面反演有效几何**：数字阶梯/半程 BB 平壁的真实无滑移位置
   不在名义格点（BB 半程反射对角链法向落点 0.707≠0.5 → 平壁有效位移 ~+0.45 格，
   格单位常数）。用解析解族对整个中心测量剖面（$u>0.2u_{\max}$）做加权 LSQ，
   反演两个有效半宽偏移 (da, db)，梯度锚定到施加流量（Zou-He 入口质量守恒给
   $Q_{\text{meas}}$ → $G_Q$）；参考解 = (有效几何, $G_Q$) 的精确级数解。
2. **CH2 名义帧直评**：名义几何 + $G_{\text{nom}}$（由 $\bar u=u_{\text{in}}$，
   无任何测量量）。单独报告，不作主判据（管道严格评审先例）。
3. **独立交叉**：$Po_{\text{sim}}$（纯测量：内区压力梯度 + 实测均值 + 名义
   $D_h$）vs Shah-London；$G_{\text{fit}}/G_Q$；$G_{\text{int}}/G_Q$。

判决规则：主通道每档 ≤3% 且随加密单调下降；反演偏移量跨档 spread ≤0.05 格。

## 配置

| 项 | 值 |
|---|---|
| 格子/碰撞 | D3Q19 BGK（τ=0.8），库原语 + compile_route |
| 驱动 | Zou-He 速度入口 $u_{\text{in}}$=**0.005** / 压力出口 ρ=1 |
| 壁 | 库 `bounce_back_cells_3d`（post-stream 半程） |
| 域 | L/W=4，测点 x=nx/2（发展度入口诊断见披露 3） |
| 档位 | 方形 W=32/64/128；2:1 矩形 W=64/128（H=W/2） |

## 结果

### 方形 1:1（`result.json`）

| W | 步数 | eff max % | eff L2 % | nom max % | da | db | $Po_{\text{sim}}$ err % |
|---|---|---|---|---|---|---|---|
| 32 | 250000 | **0.1232** | 0.1270 | 4.6867 | +0.4285 | +0.4285 | −9.49 |
| 64 | 250000 | **0.0518** | 0.0377 | 2.4938 | +0.4502 | +0.4502 | −4.86 |
| 128 | 268000 | **0.0336** | 0.0225 | 1.2340 | +0.4640 | +0.4642 | −2.74 |

### 矩形 2:1（`duct_ar05/result.json`）

| W×H | 步数 | eff max % | eff L2 % | nom max % | da | db | $Po_{\text{sim}}$ err % |
|---|---|---|---|---|---|---|---|
| 64×32 | 250000 | **0.0725** | 0.0785 | 3.2094 | +0.4319 | +0.4214 | −7.65 |
| 128×64 | 255000 | **0.0231** | 0.0227 | 1.7293 | +0.4413 | +0.4261 | −4.36 |

- 主通道全档 ≤3%（富余 30–100×）且随加密单调下降 ✓。
- da≈db（方形自洽）；da/db 跨档 spread 0.0355/0.0357（方形）、0.0094/0.0047
  （2:1）≤ 0.05 ✓ —— **平壁有效位移格无关**（与环隙曲壁的 O(1/R) 漂移对照，
  见 pending/poiseuille_3d_annulus）。
- $Po_{\text{sim}}$ 随加密单调收敛到 Shah-London（56.9083 / 62.1922），
  一阶 $\propto da/W$；$G_{\text{fit}}/G_Q$ = 1.0013/1.0003/1.0001（方形）。

## 判定

**verified**（预注册口径）：主通道全档 ≤3% + 单调 ✓，反演偏移格无关 ✓，
`extrap: "none"`，入口零手写核（grep 铁律零命中）。

## 披露

1. **名义帧粗档超标**：nom max 4.69%（W32）、3.21%（2:1 W64）> 3%；根因 = 平壁
   有效位移 +0.45 格（BB 几何常数，被 CH1 反演吸收）。直接可观测口径（nom 剖面
   + $Po_{\text{sim}}$）下 W64/W128 nom 已 ≤3%（2.49/1.23、1.73%），
   $Po_{\text{sim}}$ 仅 W128 过线 —— 与 thermal_cavity 粗档披露同构，入库扫描
   含全部档位、粗档失败如实披露（合并即默认接受该判法）。
2. **掩模敏感性**：nominal 通道 W64 在窄中心盒（H/4）2.49% vs 宽盒/阈值掩模
   3.02–4.53%（判据不受影响：名义帧为披露通道）。
3. **入口长度诊断与协议修订**：首轮 u_in=0.02 的 W128 出现入口发展污染
   （eff 0.55%、da=+0.77）；x 站点扫描证实 $L_e\propto u_{\text{in}}W^2$，
   修订 u_in→0.005（余量 22/10.8/10.9 个 $L_e$）后重跑全档；旧工件完整保留于
   staging `duct/uin02_archive/`（修订先写 NOTES 再重跑，时间戳可查）。
4. 质量漂移同环隙披露（Zou-He 稳态密度抬升），不影响反演稳定性。

## 运行

```bash
python run.py scan <out_dir> --W 32 64 128 --u-in 0.005 \
  --min-steps 250000 --max-steps 600000 --device cuda:0
python run.py scan <out_dir>/ar05 --W 64 128 --aspect 0.5 --u-in 0.005 \
  --min-steps 250000 --max-steps 600000 --device cuda:0   # 2:1（duct_ar05/）
python run.py summarize <out_dir> --W 32 64 128
```

## 参考

- Shah, R.K. & London, A.L. (1978). *Laminar Flow Forced Convection in Ducts*,
  Supplement 1, Academic Press（方形 Po=56.9083、2:1 Po=62.1922）。
- Zou, Q. & He, X. (1997). *Phys. Fluids* 9, 1591（Zou-He 入口/出口）。
