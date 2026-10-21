#!/bin/bash
# W11-A2 六档全跑脚本
# 分离运行：nohup 后台，GPU5 独占

set -e

PYTHON=/nfs/wangxi/venvs/tensorlbm/bin/python
SCRIPT=benchmarks/pending/cylinder_re40_st/w11/src/run_formal_w11a2.py
OUT_BASE=/nfs/wangxi/runs/bm_close/cylinder_re40_st/w11a2
LOG=$OUT_BASE/run_all_w11a2.log

mkdir -p $OUT_BASE

echo "=== W11-A2 六档全跑 ===" > $LOG
echo "开始时间: $(date -Iseconds)" >> $LOG
echo "Python: $PYTHON" >> $LOG
echo "脚本: $SCRIPT" >> $LOG
echo "输出目录: $OUT_BASE/formal_w11a2/" >> $LOG
echo "" >> $LOG

export CUDA_VISIBLE_DEVICES=5

# 六档：Re={20,40} × D={40,80,160}
for re in 20 40; do
  for D in 40 80 160; do
    echo "--- re=$re D=$D ---" >> $LOG
    echo "[$(date -Iseconds)] 启动 re=$re D=$D" >> $LOG
    $PYTHON $SCRIPT --re $re --D $D >> $LOG 2>&1
    echo "[$(date -Iseconds)] 完成 re=$re D=$D" >> $LOG
    echo "" >> $LOG
  done
done

echo "=== 全部完成 ===" >> $LOG
echo "结束时间: $(date -Iseconds)" >> $LOG

# 写 DONE.json
cat > $OUT_BASE/DONE.json <<EOF
{
  "case": "cylinder_re40_st_w11a2",
  "mission": "m120",
  "status": "completed",
  "timestamp": "$(date -Iseconds)",
  "log": "$LOG",
  "output_dir": "$OUT_BASE/formal_w11a2",
  "runs": [
    {"re": 20, "D": 40},
    {"re": 20, "D": 80},
    {"re": 20, "D": 160},
    {"re": 40, "D": 40},
    {"re": 40, "D": 80},
    {"re": 40, "D": 160}
  ]
}
EOF

echo "DONE.json 已写入: $OUT_BASE/DONE.json" >> $LOG
