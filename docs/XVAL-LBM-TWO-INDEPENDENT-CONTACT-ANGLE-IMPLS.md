# XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — 两套独立接触角实现的交叉验证（第 2 轮）

**入口**：`xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/`（`predeclared.json` 预注册判据、
`run_xval.py` 矩阵 + 汇总、`make_report.py` 图与报告、`runs/*/result.json` 原始数据、
`REPORT.md` 六段论文级报告；本文件只是指针，不复制任何测量数字）。

**一句话结论**：同一自由能上的**两套互不共享代码的壁面润湿实现**——壁面通量
(`kappa d_n psi = (3/4) sigma cos(theta)(1-psi^2)`) 与**纯几何 ghost**
(`psi_ghost = psi_0 + cot(theta)|d_t psi|`，无壁面自由能)——在解析 Young 律基准上
**可测行 10/10 + 10/10 全部 <15%**（中位误差 ~2.7% / ~3.0%），且两者在可测角上互差 < 3%；
第三臂（壁面取值 Dirichlet，`cos(theta)=(3p-p^3)/2`）在 `theta>=90` 侧 5/5 通过。
强润湿 `theta<=45` 与 Dirichlet 臂 `theta<90` 的失效**如实登记为不可测行**（见 REPORT §6），
不放宽判据。

## 新增能力

* `src/tensorlbm/wetting/two_impl_contact_angle.py`
  * `wall_ghost_rows(arm, ...)`：`dirichlet` / `neumann_flux` / `geometric_ghost` 三种壁面条件，
    都作用在同一个 `mu = a psi(psi^2-1) - kappa lap psi` 自由能上；
  * `grad_laplacian_ghost`：支持**逐列 ghost** 的各向同性 9 点 stencil（与库内
    `grad_laplacian` 逐位一致，单测锁定）；
  * `measure_contact_angle`：两套实现无关的外形估计器（守恒面积 Laplace 帽反演 + 等值线圆拟合），
    带气相偏移校正；
  * `initialise_matched_cap`：体积匹配初值（消除上一轮 "初值过肥 → 气相过饱和 → 面积估计器
    系统性偏移 ~18°@90°" 的坑）。
* 缺陷修复：`wall_order_parameter_2d.contact_angle_from_field` 的 `theta_circle` 符号
  （旧式 `acos((y_c-y_wall)/R)` 对所有 `theta != 90` 返回补角），回归测试
  `test_library_circle_estimator_is_not_supplementary`。

## 与 OSS 谱系的对应

| 臂 | OSS 出处（`oss_solutions`） |
|---|---|
| `geometric_ghost` | 281 walberla 自由面侧 `ContactAngle.h`（几何 ghost，只缓存 sin/cos）；5 openLBMPM 几何润湿 |
| `dirichlet` | 281 walberla 相场侧 codegen `ContactAngle`（按界面厚度与角度反解壁面 C）；91 sailfish `bc_wall_grad_phase` |
| `neumann_flux` | 同一壁面自由能的自然边界条件（Jacqmin 1999；Briant–Wagner–Yeomans 2004） |

## 复现

```bash
python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py sweep
python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py summarize
python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/make_report.py
python3 -m pytest tests/test_two_impl_contact_angle.py -q
```
