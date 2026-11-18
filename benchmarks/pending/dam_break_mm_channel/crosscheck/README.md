# 溃坝自由面数值对照与前锋跳变诊断

本次诊断衔接 [PR #2767](https://github.com/cxs503/TensorLBM/pull/2767)。原 Martin–Moyce 通道实验的 41 点验收仍为 pending。这里的数值对照使用 15a × 4a 二维平面、四壁自由滑移、水柱初始尺寸 a × 2a，重力相同，时间零点统一为模拟瞬时撤板。TensorLBM 使用 D3Q19 和 3 格展向周期；waLBerla 原生算例使用 D2Q9。Basilisk 求解水、气两相的不可压 Navier–Stokes 方程，用几何 VOF 推进界面。

## 自由滑移结果

| 求解器 | T=4 前锋 Z | T=4 高度 H | T=6 前锋 Z |
|---|---:|---:|---:|
| TensorLBM BGK N=12 | 4.417 | 0.292 | — |
| TensorLBM MRT N=12 | 4.750 | 0.250 | 7.500 |
| waLBerla N=12 | 5.250 | 0.333 | 8.750 |
| waLBerla 论文配置 N=50 | 5.727 | 0.320 | 9.150 |
| Basilisk N=16 | 5.613 | 0.316 | 9.109 |
| Basilisk N=32 | 5.806 | 0.317 | 9.136 |

TensorLBM BGK 在 T=4.865 超过 Mach=0.30 上限，MRT 完成 T=11.5。MRT 的全程峰值体积漂移为 1.092%，Basilisk N=16 和 N=32 截至 T=7 的输出时刻峰值分别为 0.00143% 和 0.00090%。同档网格和同一读数下，TensorLBM MRT 在 T=4 比 waLBerla 的前锋短 9.52%，到 T=6 仍短约 14.29%。整体传播差距在真实通道的晚期跳变之前已经出现。

TensorLBM 和 waLBerla 都用地板第一行的最远界面格和上游壁第一列的最高界面格读数，采用零基索引。waLBerla 输出时间比实际推进少一个时间步，比较脚本补齐这一固定差。Basilisk 在相同位置读半填充分数交点，包含亚格插值。参数生成器将无量纲物性和时间步映射到 waLBerla 的参数。仅作数值诊断。表格聚焦 T=4 和 T=6，避开接近下游壁面时的前锋识别与反射。这一几何同实际实验的通道外宽、固壁摩擦和释放条件有差异。

waLBerla 论文配置开启 Smagorinsky（0.1），使用其发布的松弛频率。匹配配置关闭 SGS，分别使用 N=12、50 与相同的格子速度尺度。匹配的 N=50 在后续推进中反复发出质量丢失告警且高度降为零，保留在数据中作为失败工况，未放进曲线图。匹配的 N=12 到前锋接近右边界后按上游停止条件退出。两套 LBM 还存在界面受力权重、质量通量、补充分布函数和剩余质量分配的实现差异。

[曲线](free_slip.png) · [定点对比 JSON](free_slip.json) · [数据包](observations.zip)

## 真实通道的跳变

第 1802→1803 步，前锋读数从 11.248 跳到 10.012。全场最远半填充格的零基位置为 144→144。在 x≥121 格的下游，仍有 246 个半填充格，与水柱底部连通的半填充格由 180 个变为 0 个。该区域液体体积为 251.164→251.912 格体积。这一步全域液体库存变化为 +0.025198 格质量。

在两个时刻，按 0.25 填充率建立连通区域都能到达第 134 格。把连通阈值和等值线阈值分开后，0.5 等值线的读数为 11.248282→11.250346，两步之间连续向前。这次跳变可归因于测量连通性丢失，原始液体仍在下游。这项原型保留为诊断，其跨网格和喷溅工况的适用性仍待测。

[跳变诊断 JSON](front_event.json)包含两帧连通性和逐阶段质量账本。原有阶段账本记录的是每个阶段的合计量，其中各阶段可能包含多个操作。诊断中保留 0.05、0.25、0.5 三个连通阈值仅用于归因。Martin–Moyce 验收仍使用固定的 0.5 观测口径。

## 复算

以下命令在 TensorLBM 仓库根目录执行。设置 `PYTHONPATH=src:.`，参考 [原场景依赖](../README.md)。本次 waLBerla 锁定论文附录所用的 [01a2816](https://github.com/lssfau/walberla/tree/01a28162ae1aacf7b96152c9f886ce54cc7f53ff)，GCC 14 下在上游 `src/core/RandomUUID.h` 添加 `#include <cstdint>`，求解代码按该版本构建。Basilisk 从 [b84c670](https://github.com/comphy-lab/basilisk-C/tree/b84c6700c3840d19d3965fd5897a0eaf5f765562) 的源码镜像构建 `qcc`，使用本目录的无量纲算例。

```bash
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case_dir=benchmarks/pending/dam_break_mm_channel

python "$case_dir/run.py" --cells 12 --collision mrt --capture 1799 1806 --out /tmp/channel
python "$case_dir/front_audit.py" /tmp/channel --step 1802 --front-cell 121 --cells 12

for collision in bgk mrt; do
  python "$case_dir/run.py" --cells 12 --collision "$collision" --span periodic \
    --paper-geometry --out "/tmp/tensor-$collision"
done

# wa_src 是上述锁定版 waLBerla 源码的绝对路径。
# 添加 <cstdint> 后，按需开启 MPI。以下为本次单进程编译方式。
cmake -S "$wa_src" -B /tmp/wa-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DWALBERLA_BUILD_SHOWCASES=ON \
  -DWALBERLA_BUILD_WITH_MPI=OFF -DWALBERLA_BUILD_BENCHMARKS=OFF \
  -DWALBERLA_BUILD_TOOLS=OFF -DWALBERLA_BUILD_TUTORIALS=OFF \
  -DWALBERLA_BUILD_DOC=OFF -DWALBERLA_GIT_SUBMODULE_AUTO=OFF
cmake --build /tmp/wa-build --target DamBreakRectangular -j 6
template="$wa_src/apps/showcases/FreeSurface/DamBreakRectangular.prm"
wa_exe=/tmp/wa-build/apps/showcases/FreeSurface/DamBreakRectangular
for n in 12 50; do
  python "$case_dir/walberla_case.py" "$template" "/tmp/wa$n/case.prm" --cells "$n"
  (cd "/tmp/wa$n" && "$wa_exe" case.prm)
done
mkdir -p /tmp/wa-paper
sed -E 's/(writeFrequency +)[0-9]+;/\10;/' "$template" >/tmp/wa-paper/case.prm
(cd /tmp/wa-paper && "$wa_exe" case.prm)

(cd "$case_dir" && qcc -disable-dimensions -O2 -Wall basilisk_slip.c -o /tmp/basilisk-slip -lm)
/tmp/basilisk-slip 16 >/tmp/basilisk16.csv
/tmp/basilisk-slip 32 >/tmp/basilisk32.csv
python "$case_dir/compare_slip.py" --tensor /tmp/tensor-bgk/wall_observations.csv \
  --tensor-mrt /tmp/tensor-mrt/wall_observations.csv --wa12 /tmp/wa12/breaking-dam.txt \
  --wa50 /tmp/wa50/breaking-dam.txt --wa-paper /tmp/wa-paper/breaking-dam.txt \
  --basilisk16 /tmp/basilisk16.csv --basilisk32 /tmp/basilisk32.csv --plot /tmp/free_slip.png
```

上游资料 [waLBerla 论文](https://doi.org/10.1002/fld.5173) · [Basilisk 几何 VOF](https://basilisk.fr/src/vof.h)。
