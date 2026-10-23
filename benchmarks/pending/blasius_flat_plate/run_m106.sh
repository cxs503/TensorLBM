#!/bin/bash
# m106: 延长板长放大有效 Re_x 测试脚本

export CUDA_VISIBLE_DEVICES=5

PYTHON=/nfs/wangxi/venvs/tensorlbm/bin/python
RUNDIR=/nfs/wangxi/runs/bm_close/blasius_flat_plate
CASEDIR=$(pwd)

echo "=== m106: 延长板长测试 ==="
echo "开始时间: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "工作目录: $CASEDIR"
echo "结果目录: $RUNDIR"

# plate800: Re_x = 2100 (probe at x_eff=420)
echo ""
echo ">>> 运行 plate800 (Re_x=2100)..."
$PYTHON run.py $RUNDIR/R7p_plate800_bgk.json --grid plate800 --collision bgk --U 0.05 --nu 0.01 --steps 30000 --device cuda
if [ $? -eq 0 ]; then
    echo "✓ plate800 BGK 完成"
else
    echo "✗ plate800 BGK 失败"
fi

# plate1600: Re_x = 4100 (probe at x_eff=820)
echo ""
echo ">>> 运行 plate1600 (Re_x=4100)..."
$PYTHON run.py $RUNDIR/R8p_plate1600_bgk.json --grid plate1600 --collision bgk --U 0.05 --nu 0.01 --steps 30000 --device cuda
if [ $? -eq 0 ]; then
    echo "✓ plate1600 BGK 完成"
else
    echo "✗ plate1600 BGK 失败"
fi

echo ""
echo "=== 全部完成 ==="
echo "结束时间: $(date)"
