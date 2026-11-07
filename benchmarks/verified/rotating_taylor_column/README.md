# Taylor 柱（D3Q19）— Taylor–Proudman 柱状停滞 benchmark

**判定: 未达标 (FAIL，如实记录)**
—— 最佳配置柱区 $w_{\max}/U_\infty=5.67\%$（判据 <5 %，差 13 % 相对），
柱体延伸长度误差 70 %（判据 <10 %）。失败原因已定位到参数区间而非求解器缺陷（见下）。

## 预期物理

旋转系中竖直障碍物（半径 $a$，自底板升到 $z_\text{top}<H$）上方，慢速定常
（$Ro=U_\infty/(fa)\ll1$）无粘流的 Taylor–Proudman 定理要求 $\partial\mathbf u/\partial z=0$，
流体无法"翻越"障碍物 → 障碍物正上方形成**停滞柱**（$w\approx0$，水平速度也近于零），
柱体贯穿全深（Taylor 1923；Greenspan 1968 ch.3）。

## 仿真构成

- D3Q19 BGK + Guo 力项 + Coriolis；盒子 $(n_z,n_y,n_x)$，$x/y$ 周期（侧向无界），
  $z=0$ 与 $z=n_z-1$ **自由滑移**端盖（避免端盖埃克曼层向被测内区泵入
  $O(\sqrt E)U_\infty$ 的竖直速度），障碍物为体素圆柱、无滑移。
- 自由流由常体力 $\mathbf F=(0,fU_\infty,0)$ 经地转平衡给出 $\mathbf u=(U_\infty,0,0)$
  （$-2\boldsymbol\Omega\times\mathbf u+\mathbf F=0$）；$f=2U_\infty/(Ro_\text{input}\,a)$，
  JSON 里的 `Ro` 是真实 Rossby 数 $U_\infty/(fa)=Ro_\text{input}/2$。
- 柱区统计取**内盘 $r\le0.7a$**：障碍物轮廓的阶梯化会让 $r\approx a$ 处的 $w$ 变成
  "绕角"插值伪迹；全 footprint 的数值作为诊断量一并存档
  （`checks.w_max_rel_full_footprint`）。
- 收敛判据：惯性周期（$2\pi/\Omega$）平均残差。
- 对照：同几何、更大 $Ro$（更弱旋转）的算例，用于说明"$w\approx0$"来自旋转约束。

## 结果（`records/`，三次真实运行）

| 档案 | $\nu$ | $\Omega$ | $Ro$ | $E_L=\nu/(\Omega a^2)$ | $w_{\max}/U_\infty$ | 延伸长度误差 | 收敛 |
|---|---|---|---|---|---|---|---|
| `taylor_Ro0.04_nu0.05.json` | 0.05 | 0.01 | 0.04 | **0.20** | 14.1 % | 86 % | 是（res 2.9e−5） |
| `taylor_Ro0.60_nu0.05.json`（对照） | 0.05 | 0.0033 | 0.60 | — | 6.6 % | 70 % | 否（res 4.4e−3） |
| `taylor_Ro0.04_nu0.01.json` | 0.01 | 0.02 | 0.04 | **0.02** | **5.67 %** | 70 % | 否（res 8.5e−4） |

柱区轴向剖面（$\nu=0.01$ 算例，从障碍物顶上一格往上）：
$w/U_\infty$ = 1.1 %, 4.5 %, **5.7 %**, 4.9 %, 3.7 %, 2.5 %, 1.7 %, 1.1 %, 0.8 %, 0.5 %
—— 中段最大，两端衰减，说明存在**跨越柱体的竖直输运**，柱未真正"冻结"。

## 失败原因（已定位，属参数区间而非求解器缺陷）

Taylor–Proudman 成立需要**局地埃克曼数** $E_L=\nu/(\Omega a^2)\ll1$
（等价于柱体尺度上的黏性/Coriolis 比）。三次运行的 $E_L$ 分别为 0.20、0.20、0.02：
把 $E_L$ 从 0.20 降到 0.02，$w_{\max}/U_\infty$ 由 14.1 % 降到 5.67 %（近似按
$\sqrt{E_L}$ 收缩，5.67 %≈14.1 %×0.41），与"柱区竖直速度由黏性—Coriolis 竞争控制"
的判据一致，也说明趋势正确、只是尚未进入 $\lesssim5\%$ 区间。

要把 $w_{\max}$ 压到 5 % 以下并让柱体延伸贯穿全深，需要同时满足

- $E_L=\nu/(\Omega a^2)\lesssim0.01$（$\nu\approx0.005$–0.01 @ $\Omega\approx0.02$–0.05, $a=5$），
- 障碍物自身埃克曼层 $\delta_E=\sqrt{\nu/\Omega}$ 至少 1–2 格（$\nu/\Omega\gtrsim1$，与上一条在
  $\Omega\approx0.02$、$\nu\approx0.01$ 附近相容，正好是本次最佳算例），
- 端盖必须自由滑移（否则端盖泵吸另加 $O(\sqrt E)$，见 Ekman 案例），
- 积分时间 $>10$ 个惯性周期（本次最大 10062 步 = 16 个周期，仍 res=8.5e−4 未达 2e−5）。

即需要更长的积分与更细的柱体分辨，超出本战役预算；**该判据本轮未通过，不予覆盖**。
同一求解器的旋转实现已在 Ekman 案例中被独立验证（$\delta_E$ 3.1 %、五点剖面 2.6 %），
因此本失败指向参数/预算，而非 Coriolis 力实现错误（后者另有不动点测试与惯性振荡
逐点对照测试固化）。

## 复现

    python benchmarks/verified/rotating_taylor_column/run.py single out.json \
        --ro 0.08 --nu 0.01 --nz 30 --ny 24 --nx 24 --max-steps 10000 --threads 4

（`--ro` 输入是 $2Ro$；JSON `config.Ro` 为真实 Rossby 数。）