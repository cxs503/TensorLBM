"""精确对比: 1 卡 vs 2 卡, dump f_leaf 分段的均值 (找差异段)。"""
import sys, json, subprocess, torch
sys.path.insert(0, 'src')

# 1 卡跑 3 步, dump 完整 f_leaf 各段均值
code1 = '''
import sys; sys.path.insert(0, "src")
import torch, torch.distributed as dist, json
import examples.octree_distributed_validate as M
# 直接 import 会跑 main? 不, main 只在 __main__ 跑
'''
# 简化: 在脚本里加 dump 逻辑 (env 控制)
print("Use script instrumentation instead")
