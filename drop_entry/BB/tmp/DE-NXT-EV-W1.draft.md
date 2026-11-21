# DE-NXT-EV-W1 Draft

## 自检
自检：1/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）

## 背景
Santini case 3 单滴入水，峰深 1.8574D@10.39ms。当前 CACSim 与 free_surface_probe 均未达 ≤3% 门限。

## 排除路线
- 网格纯收敛
- 参数微调
- u_limit
- ABB 整体移除
- rho_g 简单缩放
- 空穴压亏

## 设计臂
### ARM ABB_DIAG
- 诊断 ABB 气相压力重建保真度
- 开启日志，cells=8
- 预测：净漂移 <0.1%，r<0.3
- 证伪：净漂移>1% 或 r>0.5

### ARM GAS_FLOW
- 启用外部势流耦合
- --gas-flow 开关，cells=8
- 预测：深度变化 <2%
- 证伪：峰值偏差>3% 或相位偏移>0.5ms

## 下一步
- 提交集群运行
- 分析结果
- 更新 TRACK_EV_W1.md
